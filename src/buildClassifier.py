from __future__ import division
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, GroupKFold
from sklearn.metrics.scorer import make_scorer

from multiscorer import multiscorer as ms
from physionetscore import score, sensitivity, specificity

import numpy as np
import pdb
import logging

logger = logging.getLogger(__name__)

def buildClassifier(features, classifications):
    forest = RandomForestClassifier(n_estimators=1000, random_state=42, n_jobs=-1)
    logger.info("Generating model from training data...")
    '''
    forest.fit(X_train, y_train)

    logger.info("Accuracy on training set: {:.3f}".format(forest.score(X_train, y_train)))
    logger.info("Accuracy on test set: {:.3f}".format(forest.score(X_test, y_test)))
    '''
    groups = generateGroups(features)

    #physionetScorer = make_scorer(physionetScore)
    scorer = ms.MultiScorer({
        'score': (score, {}),
        'sensitivity': (sensitivity, {}),
        'specificity': (specificity, {})
    })

    # Evaluate model using startified cross-validation
    cross_val_score(
        forest,
        features,
        classifications,
        groups,
        cv=GroupKFold(n_splits=int(np.max(groups)+1)),
        scoring=scorer
    )

    results = scorer.get_results()

    logging.info("Cross-validation scores: {}".format(results['score']))
    logging.info("Sensitivity {}".format(results['specificity']))
    logging.info("Specificity: {}".format(results['sensitivity']))
    logging.info("Average Cross-validation score: {}".format(np.mean(results['score'])))
    logging.info("Standard-dev Cross-validation score: {}".format(np.std(results['score'])))


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

