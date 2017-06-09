from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pdb
import logging

logger = logging.getLogger(__name__)

def buildClassifier(features, classifications):
    X_train, X_test, y_train, y_test = train_test_split(features, classifications, test_size=0.33, random_state=42)

    forest = RandomForestClassifier(n_estimators=1000, random_state=42)
    logger.info("Generating model from training data...")
    forest.fit(X_train, y_train)

    logger.info("Accuracy on training set: {:.3f}".format(forest.score(X_train, y_train)))
    logger.info("Accuracy on test set: {:.3f}".format(forest.score(X_test, y_test)))
