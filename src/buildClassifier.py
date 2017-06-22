from __future__ import division
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, GroupKFold
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
# Recursive Feature Elimination with Cross-Validation
from sequential_feature_selector_group import SequentialFeatureSelectorGroup as SFS
import optunity

from multiscorer import multiscorer as ms
from physionetscore import score, sensitivity, specificity
from group import generateGroups

import numpy as np
import pandas as pd
import pdb
import logging

logger = logging.getLogger(__name__)
random_state = np.random.RandomState(42)

def generateModel(features, classifications, groups, algorithm, n_neighbors=None, n_estimators=None, max_features=None,
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
        raise ArgumentError('Unknown algorithm: {}'.format(algorithm))

    return model


def evaluateModel(features, classifications, groups, model):
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
        k_features=(1, 20),
        forward=True,
        floating=True,
        verbose=2,
        scoring=physionetScorer,
        cv=GroupKFold(n_splits=int(np.max(groups)+1)),
    )

    sfs1 = sfs1.fit(features.as_matrix(), classifications.as_matrix(), groups=groups)

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

    return sfs1.k_score_


def fitOptimizedModel(features, classifications, optimization_fpath, **kwargs):
    groups = generateGroups(features)
    # load model information from file
    modelInfo = pd.read_pickle(optimization_fpath)
    algorithm = modelInfo.pop('algorithm')
    model = generateModel(features, classifications, groups, algorithm, **modelInfo)
    scr = evaluateModel(features, classifications, groups, model)
    return scr


def optimizeClassifierModel(features, classifications, optimization_fpath):
    groups = generateGroups(features)

    def optimizationWrapper(algorithm, **kwargs):
        model = generateModel(features, classifications, groups, algorithm, **kwargs)
        scr = evaluateModel(features, classifications, groups, model)
        return scr

    search = {'algorithm': {
                            'SVM': {'kernel': {'linear': {'C': [0, 2]},
                                            'rbf': {'gamma': [0, 1], 'C': [0, 10]},
                                            'poly': {'degree': [2, 5], 'C': [0, 50], 'coef0': [0, 1]}
                                            }
                                    },
                            'random-forest': {'n_estimators': [10, 30],
                                           'max_features': [5, 20],
                                              'max_depth': [1, 10]
                                              }
                            }
            }

    optimal_configuration, info, solverInfo = optunity.maximize_structured(
        optimizationWrapper,
        search_space=search,
        num_evals=50,
        pmap=optunity.pmap
    )

    # Create dictionary of all parameters that have values
    solution = pd.Series({k: v for k, v in optimal_configuration.items() if v is not None})
    logging.info("Solution:".ljust(92))
    for item in solution.iteritems():
        logging.info("{:20.20}{:72.72}".format(item[0], str(item[1])))

    solution.to_pickle(optimization_fpath)



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
