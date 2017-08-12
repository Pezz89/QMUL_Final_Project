from __future__ import division
from sklearn.model_selection import cross_val_score, GroupKFold
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics.scorer import make_scorer
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression

from sklearn.decomposition import KernelPCA as KPCA, PCA


# k nearest neighbours
from sklearn.neighbors import KNeighborsClassifier
# support vector machine classifier
from sklearn.svm import SVC
# Naive Bayes
from sklearn.naive_bayes import GaussianNB
# Random Forest
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.pipeline import Pipeline

from mlxtend.feature_selection import SequentialFeatureSelector as SFS
import optunity
import optunity.search_spaces as search_spaces
import optunity.constraints as constraints

from multiscorer import multiscorer as ms
from physionetscore import score, sensitivity, specificity
from group import generateGroups

import re
import numpy as np
import pandas as pd
import pdb
import loggerops
import logging
import sys
import multiprocessing
import six

logger = logging.getLogger(__name__)
random_state = np.random.RandomState(42)

def buildModel(
    features,
    classifications,
    algorithm,
    worker_log = logging.getLogger(__name__),
    **kwargs
):

    # Filter out None arguments
    kwargs = {k:v for k,v in kwargs.iteritems() if v is not None}

    fr_method = kwargs.pop('feature_reduction')
    # Strip fr prefix substring used for distinguishing reduction parameters
    # from model parameters in optimization code
    reduction_parameters = {k.replace('fr_', ''): v.replace('fr_', '') if isinstance(v, six.string_types) else v for k, v in kwargs.iteritems() if 'fr_' in k}
    model_parameters = {k: v for k, v in kwargs.iteritems() if 'fr_' not in k and k != ''}

    interger_params = ['n_neighbors', 'max_features', 'max_depth', 'n_estimators']

    # Round any float parameters that should be ints
    for p in set(interger_params).intersection(reduction_parameters):
        reduction_parameters[p] = int(round(reduction_parameters[p]))
    for p in set(interger_params).intersection(model_parameters):
        model_parameters[p] = int(round(model_parameters[p]))

    reduction_methods = {
        "PCA": PCA,
        "None": None
    }


    worker_log.info("--------------------------------------------------------------------------------------------")
    model_methods = {
        'k-nn': KNeighborsClassifier,
        'SVM': SVC,
        'naive-bayes': GaussianNB,
        'random-forest': RandomForestClassifier,
        'adaboost': AdaBoostClassifier
    }

    worker_log.debug("Building model: {}, with parameters:".format(algorithm).ljust(92))
    worker_log.debug(" ".join(['{0}: {1}'.format(k, v) for k, v in model_parameters.iteritems()]).ljust(92))
    # Build model
    model = model_methods[algorithm](**model_parameters)

    worker_log.debug("Feature reduction method: {}, with parameters:".format(fr_method).ljust(92))
    worker_log.debug(" ".join(['{0}: {1}'.format(k, v) for k, v in reduction_parameters.iteritems()]).ljust(92))


    pipe_components = [("scaler", preprocessing.MinMaxScaler()), ("model", model)]
    if reduction_methods[fr_method]:
        pipe_components.insert(1, ("reduction", reduction_methods[fr_method](**reduction_parameters)))
    pipe = Pipeline(pipe_components)

    return pipe


def modelFeatureSelection(features, classifications, gkf, model, worker_log=logging.getLogger(__name__)):
    physionetScorer = make_scorer(score)
    worker_log.info("--------------------------------------------------------------------------------------------")
    worker_log.info("Running feature selection...".ljust(92))
    worker_log.info("--------------------------------------------------------------------------------------------")

    sfs1 = SFS(
        model,
        k_features=(10, 40),
        forward=True,
        floating=True,
        verbose=2,
        scoring=physionetScorer,
        cv=gkf,
        logger=worker_log
    )

    sfs1 = sfs1.fit(features.as_matrix(), classifications.as_matrix())

    np.set_printoptions(precision=4)

    worker_log.info("--------------------------------------------------------------------------------------------")
    #logging.info("Cross-validation scores:                   {}".format(scr).ljust(92))
    #logging.info("Sensitivity:                               {}".format(sens).ljust(92))
    #logging.info("Specificity:                               {}".format(spec).ljust(92))
    #logging.info("Average Cross-validation score:            {}".format(np.mean(scr)).ljust(92))
    #logging.info("Standard-dev Cross-validation score:       {}".format(np.std(scr)).ljust(92))
    worker_log.info("Selected features: {}".format(" ".join([str(x) for x in features.columns[np.array(sfs1.k_feature_idx_)]])).ljust(92))
    worker_log.info("k-score score:                             {}".format(sfs1.k_score_).ljust(92))
    worker_log.info("--------------------------------------------------------------------------------------------")

    return sfs1.k_score_, features.columns[np.array(sfs1.k_feature_idx_)]


def scoreOptimizedModel(train_features, test_features, train_classifications, test_classifications, optimization_fpath, **kwargs):
    # load model information from file
    groups = generateGroups(train_features)
    with pd.HDFStore(optimization_fpath) as hdf:
        iterations = [extract_number(x)[0] for x in hdf.keys()]
        try:
            latestIteration = max(iterations)
        except ValueError:
            raise ValueError("Models have not been optimized, please run with the --optimize flag")
        latestSolution = hdf["/solution{}".format(latestIteration)]
        latestFeatures = pd.Index(hdf["/bestFeatures{}".format(latestIteration)])
    latestSolution = latestSolution.dropna()
    algorithm = latestSolution.pop('algorithm')
    train_features = train_features.ix[:, latestFeatures]
    test_features = test_features.ix[:, latestFeatures]
    model = buildModel(train_features, train_classifications, algorithm, **latestSolution)
    model.fit(train_features, train_classifications)
    physionetScorer = make_scorer(score)
    finalScore = physionetScorer(model, test_features, test_classifications)
    logging.info("--------------------------------------------------------------------------------------------")
    logging.info("Final optimized score:                      {}".format(finalScore).ljust(92))
    logging.info("--------------------------------------------------------------------------------------------")


def group_train_test_split(features, classifications, groups):
    logo = LeaveOneGroupOut()
    # Split data into test and training sets by database
    train_inds, test_inds = logo.split(features, classifications, groups=groups).next()
    train_features = features.ix[train_inds]
    test_features = features.ix[test_inds]
    train_classifications = classifications.ix[train_inds]
    test_classifications = classifications.ix[test_inds]
    train_groups = generateGroups(train_features)
    test_groups = generateGroups(test_features)

    return (train_features, test_features, train_classifications, test_classifications, train_groups, test_groups)


def optimizeClassifierModel(features, classifications, groups, optimization_fpath, parallelize=False):
    gkf = list(GroupKFold(n_splits=3).split(features,classifications,groups))#int(np.max(groups)+1))

    def dummyWrapper(algorithm, **kwargs):
        '''Dummy function created for debugging optimization quickly'''
        return (0.5, pd.Index(['diaCent', 'diaDur', 'diaFlat']))

    def optimizationWrapper(algorithm, **kwargs):
        process_name = multiprocessing.current_process().name
        worker_log = loggerops.create_logger(
            process_name,
            log_filename="{}.log".format(process_name),
            use_stream_handler=False
        )
        worker_log.propagate = False
        model = buildModel(features, classifications, algorithm, worker_log=worker_log, **kwargs)
        scr, featureLabels = modelFeatureSelection(features, classifications, gkf, model, worker_log=worker_log)
        return scr, featureLabels
    # TODO: Used for quickly debugging particle swarm optimization, remove for
    # production
    #optimizationWrapper = dummyWrapper

    # Define search space, providing model names and parameter ranges to search
    # for best solution
    search = {
        'algorithm': {
            'SVM': {
                'kernel': {
                    'linear': {'C': [0, 2]},
                    'rbf': {'gamma': [0, 1], 'C': [0, 10]},
                    'poly': {'degree': [2, 5], 'C': [0, 50], 'coef0': [0, 1]}
                }
            },
            'adaboost': {
                'n_estimators': [10, 100],
            },
            'naive-bayes': None,
        },
        'feature_reduction': {
            'PCA': None,
            'None': None
        }
    }
    '''
            'random-forest': {
                'n_estimators': [10, 30],
                'max_features': [5, 20],
                'max_depth': [1, 10]
            }
    '''

    if parallelize:
        pmap = optunity.pmap
    else:
        pmap = map

    num_evals=50
    tree = search_spaces.SearchTree(search)
    box = tree.to_box()

    # we need to position the call log here
    # because the function signature used later on is internal logic
    #f = optunity.functions.logged(optimizationWrapper)

    # wrap the decoder and constraints for the internal search space representation
    f = tree.wrap_decoder(optimizationWrapper)
    f = constraints.wrap_constraints(f, -sys.float_info.max, range_oo=box)

    # Create solver keyword args based on number of evaluations and box
    # constraints
    suggestion = optunity.suggest_solver(num_evals, "particle swarm", **box)
    solver = optunity.make_solver(**suggestion)
    solution, details = optunity.optimize(
        solver,
        f,
        maximize=True,
        max_evals=num_evals,
        pmap=pmap,
        decoder=tree.decode,
        solutionFPath=optimization_fpath
    )

    # TODO: Remove this...
    optimal_configuration, info, solverInfo = solution, details, suggestion

    # Create dictionary of all parameters that have values
    logging.info("Solution:".ljust(92))
    for item in optimal_configuration.iteritems():
        logging.info("{:20.20}{:72.72}".format(item[0], str(item[1])))


def extract_number(f):
    s = re.findall("\d+$",f)
    return (int(s[0]) if s else -1,f)


