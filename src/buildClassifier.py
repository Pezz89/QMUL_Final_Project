from sklearn.model_selection import train_test_split
import pdb

def buildClassifier(features, classifications):
    pdb.set_trace()

    X_train, X_test, y_train, y_test = train_test_split(features, classifications, test_size=0.33, random_state=42)
