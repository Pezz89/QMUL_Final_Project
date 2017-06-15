from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, GroupKFold

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
    # Evaluate model using startified cross-validation
    scores = cross_val_score(forest, features, classifications, groups, cv=GroupKFold(n_splits=int(np.max(groups)+1)))
    logging.info("Cross-validation scores: {}".format(scores))
    logging.info("Average Cross-validation score: {}".format(np.mean(scores)))


'''
Generate groups for a pandas DataFrame using keys for splitting
'''
def generateGroups(frame):
    groups = np.empty(frame.index.size)

    groupKeys = {}
    i = 0
    for ind, key in enumerate(frame.index):
        groupChar = key[0]
        if groupChar not in groupKeys:
            groupKeys[groupChar] = i
            i += 1
        groups[ind] = groupKeys[groupChar]
    return groups
