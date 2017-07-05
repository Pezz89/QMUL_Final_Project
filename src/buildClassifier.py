from __future__ import division
from sklearn.model_selection import cross_val_score, GroupKFold
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics.scorer import make_scorer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC


# k nearest neighbours
from sklearn.neighbors import KNeighborsClassifier
# support vector machine classifier
from sklearn.svm import SVC
# Naive Bayes
from sklearn.naive_bayes import GaussianNB
# Random Forest
from sklearn.ensemble import RandomForestClassifier
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
import logging
import sys

logger = logging.getLogger(__name__)
random_state = np.random.RandomState(42)

def buildModel(features, classifications, algorithm, n_neighbors=None, n_estimators=None, max_features=None,
                kernel=None, C=None, gamma=None, degree=None, coef0=None, max_depth=None):
    if algorithm == 'k-nn':
        n_neighbors = int(np.round(n_neighbors))
        logger.debug("Building k-NN Model with parameters:".ljust(92))
        logger.debug("n_neighbors={}".format(n_neighbors).ljust(92))
        model = KNeighborsClassifier(n_neighbors=int(n_neighbors))
    elif algorithm == 'SVM':
        logger.debug("Building SVM Model with parameters:".ljust(92))
        logger.debug("kernel={0}, C={1}, gamma={2}, degree={3}, coef0={4}".format(kernel, C, gamma, degree, coef0).ljust(92))
        model = train_svm(kernel, C, gamma, degree, coef0)
    elif algorithm == 'naive-bayes':
        logger.debug("Building Gaussian NB Model (no parameters)".ljust(92))
        model = GaussianNB()
    elif algorithm == 'random-forest':
        max_features = int(np.round(max_features))
        n_estimators = int(np.round(n_estimators))
        max_depth=int(np.round(max_depth))
        logger.debug("Building Random Forest Model with parameters:".ljust(92))
        logger.debug("n_estimators={0}, max_features={1}, max_depth={2}".format(n_estimators, max_features, max_depth).ljust(92))
        model = RandomForestClassifier(n_estimators=int(n_estimators),
                                    #max_features=int(max_features), random_state=42)
                                       max_depth=max_depth,
                                    random_state=42)
    else:
        raise ValueError('Unknown algorithm: {}'.format(algorithm))

    return model


def modelFeatureSelection(features, classifications, gkf, model):
    physionetScorer = make_scorer(score)
    '''
    scorer = ms.MultiScorer({
        'score': (score, {}),
        'sensitivity': (sensitivity, {}),
        'specificity': (specificity, {})
    })
    '''

    sfs1 = SFS(
        model,
        k_features=(30, 50),
        forward=False,
        floating=True,
        verbose=2,
        scoring=physionetScorer,
        #cv=gkf
        cv=0
    )

    sfs1 = sfs1.fit(features.as_matrix(), classifications.as_matrix())

    # Evaluate model using stratified cross-validation
    '''
    cross_val_score(
        model,
        features,
        classifications,
        groups,
        cv=GroupKFold(n_splits=int(np.max(groups)+1)),
        scoring=scorer
    )
    '''

    np.set_printoptions(precision=4)
    '''
    results = scorer.get_results()
    scr = np.array(results['score'])
    sens = np.array(results['sensitivity'])
    spec = np.array(results['specificity'])
    '''

    logging.info("--------------------------------------------------------------------------------------------")
    #logging.info("Cross-validation scores:                   {}".format(scr).ljust(92))
    #logging.info("Sensitivity:                               {}".format(sens).ljust(92))
    #logging.info("Specificity:                               {}".format(spec).ljust(92))
    #logging.info("Average Cross-validation score:            {}".format(np.mean(scr)).ljust(92))
    #logging.info("Standard-dev Cross-validation score:       {}".format(np.std(scr)).ljust(92))
    logging.info("k-score score:                             {}".format(sfs1.k_score_).ljust(92))
    logging.info("--------------------------------------------------------------------------------------------")
    pdb.set_trace()

    return sfs1.k_score_, features.columns[np.array(sfs1.k_feature_idx_)]


def scoreOptimizedModel(train_features, test_features, train_classifications, test_classifications, optimization_fpath, **kwargs):
    # load model information from file
    groups = generateGroups(train_features)
    with pd.HDFStore(optimization_fpath) as hdf:
        iterations = [extract_number(x)[0] for x in hdf.keys()]
        latestIteration = max(iterations)
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
    #logging.info("Cross-validation scores:                   {}".format(scr).ljust(92))
    #logging.info("Sensitivity:                               {}".format(sens).ljust(92))
    #logging.info("Specificity:                               {}".format(spec).ljust(92))
    #logging.info("Average Cross-validation score:            {}".format(np.mean(scr)).ljust(92))
    #logging.info("Standard-dev Cross-validation score:       {}".format(np.std(scr)).ljust(92))
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
        model = buildModel(features, classifications, algorithm, **kwargs)
        scr, featureLabels = modelFeatureSelection(features, classifications, gkf, model)
        return scr, featureLabels
    # TODO: Used for quickly debugging particle swarm optimization, remove for
    # production
    optimizationWrapper = dummyWrapper

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
            'random-forest': {
                'n_estimators': [10, 30],
                'max_features': [5, 20],
                'max_depth': [1, 10]
            }
        }
    }

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

    # TODO: Remove this...
    optimal_configuration, info, solverInfo = solution, details, suggestion

    # Create dictionary of all parameters that have values
    logging.info("Solution:".ljust(92))
    for item in optimal_configuration.iteritems():
        logging.info("{:20.20}{:72.72}".format(item[0], str(item[1])))


def extract_number(f):
    s = re.findall("\d+$",f)
    return (int(s[0]) if s else -1,f)


def train_svm(kernel, C, gamma, degree, coef0):
    """A generic SVM training function, with arguments based on the chosen kernel."""
    if kernel == 'linear':
        model = SVC(kernel=kernel, C=C, random_state=42)
    elif kernel == 'poly':
        model = SVC(kernel=kernel, C=C, degree=degree, coef0=coef0, random_state=42)
    elif kernel == 'rbf':
        model = SVC(kernel=kernel, C=C, gamma=gamma, random_state=42)
    else:
        raise ArgumentError("Unknown kernel function: %s" % kernel)
    return model
