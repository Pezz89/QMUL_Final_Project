import numpy as np
def score(y, y_pred):
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


def sensitivity(y, y_pred):
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

    return se


def specificity(y, y_pred):
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

    # Specificity
    sp = trueNegative/(trueNegative+falsePositive)
    return sp
