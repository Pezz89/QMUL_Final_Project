from __future__ import division
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, GroupKFold
from sklearn.metrics.scorer import make_scorer

import numpy as np
import pdb
import logging

logger = logging.getLogger(__name__)

def buildClassifier(features, classifications):
    #X_train, X_test, y_train, y_test = train_test_split(features, classifications, test_size=0.33, random_state=42)

    forest = RandomForestClassifier(n_estimators=1000, random_state=42, n_jobs=-1)
    logger.info("Generating model from training data...")
    '''
    forest.fit(X_train, y_train)

    logger.info("Accuracy on training set: {:.3f}".format(forest.score(X_train, y_train)))
    logger.info("Accuracy on test set: {:.3f}".format(forest.score(X_test, y_test)))
    '''
    groups = generateGroups(features)

    physionetScorer = make_scorer(physionetScore)
    # Evaluate model using startified cross-validation
    scores = cross_val_score(
        forest,
        features,
        classifications,
        groups,
        cv=GroupKFold(n_splits=int(np.max(groups)+1)),
        scoring=physionetScorer
    )
    logging.info("Cross-validation scores: {}".format(scores))
    logging.info("Average Cross-validation score: {}".format(np.mean(scores)))


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

def physionetScore(y, y_pred):
    unsure_weight=0.5
    y = np.array(y)
    truePositive = np.sum(np.logical_and((y==y_pred), (y==1)))
    falsePositive= np.sum(np.logical_and((y!=y_pred), (y==-1)))
    trueNegative = np.sum(np.logical_and((y==y_pred), (y==-1)))
    falseNegative= np.sum(np.logical_and((y!=y_pred), (y==1)))
    positiveUnsure = np.sum(np.logical_and((y==0), (y_pred==1)))
    negativeUnsure = np.sum(np.logical_and((y==0), (y_pred==-1)))

    truePositive+=unsure_weight*positiveUnsure
    trueNegative+=unsure_weight*negativeUnsure
    truePositive+=negativeUnsure-(unsure_weight*negativeUnsure)
    trueNegative+=positiveUnsure-(unsure_weight*positiveUnsure)

    # Sensitivity
    se = truePositive/(truePositive+falseNegative)
    # Specificity
    sp = trueNegative/(trueNegative+falsePositive)

    # Accuracy
    macc = (se+sp)/2

    return macc
