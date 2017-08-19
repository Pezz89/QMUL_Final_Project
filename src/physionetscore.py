import numpy as np
from sklearn.metrics.scorer import make_scorer as MS

'''
Calculate modified accuracy score as defined by Liu et. al
'''
def score(y, y_pred, custom_y):
    custom_y = custom_y[~custom_y.index.duplicated(keep='first')]
    unsure_weight=0.5
    inds = y.index
    y = custom_y.ix[inds, 'class']
    y = np.array(y)
    truePositive = np.sum(np.logical_and((y_pred==1), (y==1)))
    trueNegative = np.sum(np.logical_and((y_pred==-1), (y==-1)))
    falseNegative= np.sum(np.logical_and((y_pred==-1), (y==1)))
    falsePositive= np.sum(np.logical_and((y_pred==1), (y==-1)))
    positiveUnsure = np.sum(np.logical_and((y==1), (y_pred==0)))
    negativeUnsure = np.sum(np.logical_and((y==-1), (y_pred==0)))

    truePositive+=unsure_weight*positiveUnsure
    trueNegative+=unsure_weight*negativeUnsure
    falsePositive+=negativeUnsure-(unsure_weight*negativeUnsure)
    falseNegative+=positiveUnsure-(unsure_weight*positiveUnsure)

    # Sensitivity
    se = truePositive/(truePositive+falseNegative)
    # Specificity
    sp = trueNegative/(trueNegative+falsePositive)

    # Accuracy
    macc = (se+sp)/2

    return macc


'''
Calculate sensitivity score as defined by Liu et. al
'''
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


'''
Calculate specificity score as defined by Liu et. al
'''
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
