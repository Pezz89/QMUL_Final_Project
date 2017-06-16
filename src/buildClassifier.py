from __future__ import division
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, GroupKFold
from sklearn.metrics.scorer import make_scorer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from multiscorer import multiscorer as ms
from physionetscore import score, sensitivity, specificity

import numpy as np
import pdb
import logging

logger = logging.getLogger(__name__)

def buildClassifier(features, classifications):
    forest = RandomForestClassifier(n_estimators=1000, random_state=42, n_jobs=-1)
    linSVM = LinearSVC()
    lr = LogisticRegression()
    logging.info("--------------------------------------------------------------------------------------------")
    evaluateModel(lr, features, classifications, "Logistic Regression")
    evaluateModel(linSVM, features, classifications, "Linear SVM")
    evaluateModel(forest, features, classifications, "Random Forrest")
    '''
    forest.fit(X_train, y_train)

    logger.info("Accuracy on training set: {:.3f}".format(forest.score(X_train, y_train)))
    logger.info("Accuracy on test set: {:.3f}".format(forest.score(X_test, y_test)))
    '''
def evaluateModel(model, features, classifications, name=''):
    logger.info("Generating {} model from training data...".format(name).ljust(92))
    groups = generateGroups(features)

    #physionetScorer = make_scorer(physionetScore)
    scorer = ms.MultiScorer({
        'score': (score, {}),
        'sensitivity': (sensitivity, {}),
        'specificity': (specificity, {})
    })

    # Evaluate model using startified cross-validation
    cross_val_score(
        model,
        features,
        classifications,
        groups,
        cv=GroupKFold(n_splits=int(np.max(groups)+1)),
        scoring=scorer
    )

    results = scorer.get_results()
    np.set_printoptions(precision=4)
    scr = np.array(results['score'])
    sens = np.array(results['sensitivity'])
    spec = np.array(results['specificity'])

    logging.info("--------------------------------------------------------------------------------------------")
    logging.info("Cross-validation scores:                   {}".format(scr).ljust(92))
    logging.info("Sensitivity:                               {}".format(sens).ljust(92))
    logging.info("Specificity:                               {}".format(spec).ljust(92))
    logging.info("Average Cross-validation score:            {}".format(np.mean(scr)).ljust(92))
    logging.info("Standard-dev Cross-validation score:       {}".format(np.std(scr)).ljust(92))
    logging.info("--------------------------------------------------------------------------------------------")


'''
Generate groups for a pandas DataFrame using keys for splitting
'''
def generateGroups(frame):
    groups = np.empty(frame.index.size, dtype=int)

    groupKeys = {}
    i = 0
    for ind, key in enumerate(frame.index):
        groupChar = key[0]
        if groupChar not in groupKeys:
            groupKeys[groupChar] = i
            i += 1
        groups[ind] = groupKeys[groupChar]
    return groups

