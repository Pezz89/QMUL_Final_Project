from __future__ import division
from sklearn.model_selection import cross_val_score, GroupKFold, train_test_split, StratifiedKFold, GroupShuffleSplit, StratifiedShuffleSplit
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics.scorer import make_scorer
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import KernelPCA as KPCA, PCA

from sklearn.tree import DecisionTreeClassifier
# k nearest neighbours
from sklearn.neighbors import KNeighborsClassifier
# support vector machine classifier
from sklearn.svm import SVC
# Naive Bayes
from sklearn.naive_bayes import GaussianNB
# Random Forest
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.pipeline import Pipeline

from mlxtend.classifier import StackingCVClassifier

from mlxtend.feature_selection import SequentialFeatureSelector as SFS
import optunity
import optunity.search_spaces as search_spaces
import optunity.constraints as constraints

from multiscorer import multiscorer as ms
from physionetscore import score, sensitivity, specificity
from group import generateGroups
from tabulate import tabulate

import re
import numpy as np
import pandas as pd
import pdb
import loggerops
import logging
import sys
import multiprocessing
import six
import textwrap

logger = logging.getLogger(__name__)
random_state = np.random.RandomState(42)
np.random.seed(42)

def buildModel(
    worker_log = logging.getLogger(__name__),
    **kwargs
):
    # Filter out None arguments
    kwargs = {k:v for k,v in kwargs.iteritems() if v is not None}
    #fr_method = kwargs.pop('feature_reduction')

    cl1_parameters = {k: v for k, v in kwargs.iteritems() if 'cl1_' in k}
    cl2_parameters = {k: v for k, v in kwargs.iteritems() if 'cl2_' in k}
    cl3_parameters = {k: v for k, v in kwargs.iteritems() if 'cl3_' in k}
    lr_parameters = {k: v for k, v in kwargs.iteritems() if 'lr_' in k}

    interger_params = ['n_neighbors', 'max_features', 'max_depth', 'n_estimators', 'penalty']

    cl1_parameters = {k.replace('cl1_', ''): v.replace('cl1_', '') if isinstance(v, six.string_types) else v for k, v in cl1_parameters.iteritems() if 'cl1_' in k}
    cl2_parameters = {k.replace('cl2_', ''): v.replace('cl2_', '') if isinstance(v, six.string_types) else v for k, v in cl2_parameters.iteritems() if 'cl2_' in k}
    cl3_parameters = {k.replace('cl3_', ''): v.replace('cl3_', '') if isinstance(v, six.string_types) else v for k, v in cl3_parameters.iteritems() if 'cl3_' in k}
    lr_parameters = {k.replace('lr_', ''): v.replace('lr_', '') if isinstance(v, six.string_types) else v for k, v in lr_parameters.iteritems() if 'lr_' in k}

    cl1_algorithm = cl1_parameters.pop('algorithm')
    cl2_algorithm = cl2_parameters.pop('algorithm')
    cl3_algorithm = cl3_parameters.pop('algorithm')

    # Round any float parameters that should be ints
    for p in set(interger_params).intersection(cl1_parameters):
        cl1_parameters[p] = int(round(cl1_parameters[p]))
    for p in set(interger_params).intersection(cl2_parameters):
        cl2_parameters[p] = int(round(cl2_parameters[p]))
    for p in set(interger_params).intersection(cl3_parameters):
        cl3_parameters[p] = int(round(cl3_parameters[p]))
    for p in set(interger_params).intersection(lr_parameters):
        lr_parameters[p] = int(round(lr_parameters[p]))

    penalty_str = ['l1', 'l2']
    lr_parameters['penalty'] = penalty_str[lr_parameters['penalty']]


    worker_log.info("--------------------------------------------------------------------------------------------")
    model_methods = {
        'k-nn': KNeighborsClassifier,
        'SVM': SVC,
        'naive-bayes': GaussianNB,
        'random-forest': RandomForestClassifier,
        'adaboost': AdaBoostClassifier,
        'logistic-regression': LogisticRegression
    }

    worker_log.debug("Building Stacking Classifier, with parameters:".ljust(92))
    worker_log.debug(("Algorithm: {} ".format(cl1_algorithm) + " ".join(['{0}: {1}'.format(k, v) for k, v in cl1_parameters.iteritems()])).ljust(92))
    worker_log.debug(("Algorithm: {} ".format(cl2_algorithm) + " ".join(['{0}: {1}'.format(k, v) for k, v in cl2_parameters.iteritems()])).ljust(92))
    worker_log.debug(("Algorithm: {} ".format(cl3_algorithm) + " ".join(['{0}: {1}'.format(k, v) for k, v in cl3_parameters.iteritems()])).ljust(92))
    worker_log.debug("Meta-classifier (Logistic regression) parameters:".ljust(92))
    worker_log.debug(" ".join(['{0}: {1}'.format(k, v) for k, v in lr_parameters.iteritems()]).ljust(92))


    if cl1_algorithm == 'adaboost':
        cl1_parameters['base_estimator'] = DecisionTreeClassifier(max_depth=1, min_samples_leaf=1)
    if cl2_algorithm == 'adaboost':
        cl2_parameters['base_estimator'] = DecisionTreeClassifier(max_depth=1, min_samples_leaf=1)
    if cl3_algorithm == 'adaboost':
        cl3_parameters['base_estimator'] = DecisionTreeClassifier(max_depth=1, min_samples_leaf=1)
    if cl1_algorithm == 'SVM':
        cl1_parameters['probability'] = True
    if cl2_algorithm == 'SVM':
        cl2_parameters['probability'] = True
    if cl3_algorithm == 'SVM':
        cl3_parameters['probability'] = True

    # Build model
    clf1 = model_methods[cl1_algorithm](**cl1_parameters)
    clf2 = model_methods[cl2_algorithm](**cl2_parameters)
    clf3 = model_methods[cl3_algorithm](**cl3_parameters)
    lr = LogisticRegression(random_state=42, **lr_parameters)

    pipe_components = [("imputer", preprocessing.Imputer()), ("scaler", preprocessing.MinMaxScaler()), ("model", StackingCVClassifier(classifiers=[clf1, clf2, clf3], meta_classifier=lr, use_probas=True))]
    pipe = Pipeline(pipe_components)

    return pipe


def modelFeatureSelection(features, classifications, optimization_fpath, max_features=50, worker_log=logging.getLogger(__name__), **kwargs):
    # load model information from file
    with pd.HDFStore(optimization_fpath) as hdf:
        iterations = [extract_number(x)[0] for x in hdf.keys()]
        try:
            latestIteration = max(iterations)
        except ValueError:
            raise ValueError("Models have not been optimized, please run with the --optimize flag")
        latestSolution = hdf["/solution{}".format(latestIteration)]
    latestSolution = latestSolution.dropna()

    model = buildModel(**latestSolution)
    worker_log.info("--------------------------------------------------------------------------------------------")
    worker_log.info("Running feature selection...".ljust(92))
    worker_log.info("--------------------------------------------------------------------------------------------")

    physionetScorer = make_scorer(score)

    model = SFS(
        model,
        k_features=(1, max_features),
        forward=True,
        floating=True,
        verbose=2,
        scoring=physionetScorer,
        cv=3,#gkf,
        logger=worker_log,
        n_jobs=-1
    )

    model.fit(features.as_matrix(), classifications.as_matrix())
    bestLabel = features.columns[np.array(model.k_feature_idx_)]
    bestLabel.to_series().to_hdf(optimization_fpath, key="bestFeatures")


def scoreModel(features, classifications, gkf, model, worker_log=logging.getLogger(__name__), **kwargs):
    physionetScorer = make_scorer(score)
    pipe = model

    sfs1 = pipe.fit(features.as_matrix(), classifications.as_matrix())

    np.set_printoptions(precision=4)

    worker_log.info("--------------------------------------------------------------------------------------------")
    skf = StratifiedKFold(n_splits=10, random_state=42)
    logo_scores = cross_val_score(model, features, classifications, scoring=physionetScorer, cv=skf)

    worker_log.info("Score:                             {}".format(np.mean(logo_scores)).ljust(92))
    worker_log.info("--------------------------------------------------------------------------------------------")

    return np.mean(logo_scores)


def scoreOptimizedModel(features, classifications, groups, train_features, test_features, train_classifications, test_classifications, optimization_fpath, **kwargs):
    # load model information from file
    with pd.HDFStore(optimization_fpath) as hdf:
        iterations = [extract_number(x)[0] for x in hdf.keys()]
        try:
            latestIteration = max(iterations)
        except ValueError:
            raise ValueError("Models have not been optimized, please run with the --optimize flag")
        latestSolution = hdf["/solution{}".format(latestIteration)]
        latestFeatures = pd.Index(hdf["/bestFeatures"])
    latestSolution = latestSolution.dropna()

    train_features = train_features.ix[:, latestFeatures]
    test_features = test_features.ix[:, latestFeatures]

    model = buildModel(**latestSolution)
    model.fit(train_features, train_classifications)
    physionetScorer = make_scorer(score)
    finalScore = physionetScorer(model, test_features, test_classifications)
    logging.info("--------------------------------------------------------------------------------------------")
    logging.info("Final optimized score:                      {}".format(finalScore).ljust(92))
    logging.info("Selected features:".ljust(92))
    for line in textwrap.fill(" ".join([str(x) for x in features.columns]), width=89).split('\n'):
        logging.info(line.ljust(92))
    logging.info("--------------------------------------------------------------------------------------------")

    logo = LeaveOneGroupOut()
    skf = StratifiedKFold(n_splits=10, random_state=42)

    logo_scores = cross_val_score(model, features, classifications, groups, physionetScorer, logo)

    table_header = ['a', 'b', 'c', 'd', 'e', 'f', 'Mean']
    logo_scores = np.append(logo_scores, np.mean(logo_scores))
    logo_table = tabulate(list([logo_scores]), headers=table_header, tablefmt='grid', floatfmt=".4f")
    logging.info("Leave-one-out cross-validation score:".format(logo_scores).ljust(92))
    for line in logo_table.split('\n'):
        logging.info(line.ljust(92))

    table_header = list(np.arange(1, 11))
    table_header.append('Mean')
    skf_scores = cross_val_score(model, features, classifications, groups, physionetScorer, skf)
    skf_scores = np.append(skf_scores, np.mean(skf_scores))
    skf_table = tabulate(list([skf_scores]), headers=table_header, tablefmt='grid', floatfmt=".4f")
    logging.info("Startifier K-fold cross-validation score:    {}".format(np.mean(skf_scores)).ljust(92))
    for line in skf_table.split('\n'):
        logging.info(line.ljust(92))


def group_train_test_split(features, classifications, groups):
    gss = StratifiedShuffleSplit(test_size=0.33, n_splits=10, random_state=42)

    train_features = pd.DataFrame()
    test_features = pd.DataFrame()
    train_classifications = pd.Series()
    test_classifications = pd.Series()
    train_groups = np.array([])
    test_groups = np.array([])
    for i in xrange(np.max(groups)+1):
        g_feat = features.ix[groups == i]
        g_class = classifications.ix[groups == i]
        g_group = groups[groups == i]

        train_inds, test_inds = gss.split(g_feat, g_class, g_group).next()

        train_features = train_features.append(g_feat.ix[train_inds])
        test_features = test_features.append(g_feat.ix[test_inds])
        train_classifications = train_classifications.append(g_class.ix[train_inds])
        test_classifications = test_classifications.append(g_class.ix[test_inds])
        train_groups = np.append(train_groups, g_group[train_inds])
        test_groups = np.append(test_groups, g_group[test_inds])

    return (train_features, test_features, train_classifications, test_classifications, train_groups, test_groups)


def optimizeClassifierModel(features, classifications, groups, optimization_fpath, num_evals=50, parallelize=False):
    cross_validator = StratifiedKFold(n_splits=10, random_state=42)

    def dummyWrapper(**kwargs):
        '''Dummy function created for debugging optimization quickly'''
        return (0.5, pd.Index(['diaCent', 'diaDur', 'diaFlat']))

    def optimizationWrapper(**kwargs):
        process_name = multiprocessing.current_process().name
        worker_log = loggerops.create_logger(
            process_name,
            log_filename="{}.log".format(process_name),
            use_stream_handler=False
        )
        worker_log.propagate = False
        model = buildModel(worker_log=worker_log, **kwargs)
        scr = scoreModel(features, classifications, cross_validator, model, worker_log=worker_log, **kwargs)
        return scr
    # FIXME: Used for quickly debugging particle swarm optimization, remove for
    # production
    #optimizationWrapper = dummyWrapper

    # Define search space, providing model names and parameter ranges to search
    # for best solution
    search = {
        'cl1_algorithm': {
            'cl1_SVM': {
                'cl1_kernel': {
                    'cl1_linear': {'cl1_C': [0, 10]},
                    'cl1_rbf': {'cl1_gamma': [0, 1], 'cl1_C': [0, 10]},
                    'cl1_poly': {'cl1_degree': [2, 5], 'cl1_C': [0, 50], 'cl1_coef0': [0, 1]}
                }
            },
            'cl1_naive-bayes': None,
        },
        'cl2_algorithm': {
            'cl2_SVM': {
                'cl2_kernel': {
                    'cl2_linear': {'cl2_C': [0, 10]},
                    'cl2_rbf': {'cl2_gamma': [0, 1], 'cl2_C': [0, 10]},
                    'cl2_poly': {'cl2_degree': [2, 5], 'cl2_C': [0, 50], 'cl2_coef0': [0, 1]}
                }
            },
            'cl2_naive-bayes': None,
        },
        'cl3_algorithm': {
            'cl3_SVM': {
                'cl3_kernel': {
                    'cl3_linear': {'cl3_C': [0, 10]},
                    'cl3_rbf': {'cl3_gamma': [0, 1], 'cl3_C': [0, 10]},
                    'cl3_poly': {'cl3_degree': [2, 5], 'cl3_C': [0, 50], 'cl3_coef0': [0, 1]}
                }
            },
            'cl3_naive-bayes': None,
        },
        'lr_meta_algorithm': {
            'lr_C': [0.1, 20],
            'lr_penalty': [0, 1]

        }
    }


    if parallelize:
        pmap = optunity.pmap
    else:
        pmap = map

    tree = search_spaces.SearchTree(search)
    box = tree.to_box()

    # wrap the decoder and constraints for the internal search space representation
    f = tree.wrap_decoder(optimizationWrapper)
    f = constraints.wrap_constraints(f, (-sys.float_info.max, pd.Index(['test', 'test2'])), range_oo=box)

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

    # Create dictionary of all parameters that have values
    logging.info("Solution:".ljust(92))
    for item in solution.iteritems():
        logging.info("{:20.20}{:72.72}".format(item[0], str(item[1])))


def extract_number(f):
    s = re.findall("\d+$",f)
    return (int(s[0]) if s else -1,f)

# Code taken from: https://stackoverflow.com/questions/250357/truncate-a-string-without-ending-in-the-middle-of-a-word
def smart_truncate1(text, max_length=100, suffix='...'):
    """Returns a string of at most `max_length` characters, cutting
    only at word-boundaries. If the string was truncated, `suffix`
    will be appended.
    """

    if len(text) > max_length:
        pattern = r'^(.{0,%d}\S)\s.*' % (max_length-len(suffix)-1)
        return re.sub(pattern, r'\1' + suffix, text)
    else:
        return text
