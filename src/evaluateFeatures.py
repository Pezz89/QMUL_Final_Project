import pdb
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import chi2, f_classif
from sklearn import preprocessing
from sklearn.ensemble import ExtraTreesClassifier
import numpy as np
sns.set()

plot = True

def evaluateFeatures(features, classifications):
    # Find classification for each series of features
    featureNames = features.keys()

    #sns.pairplot(features.ix[:, 0:3])
    minmax_scale = preprocessing.MinMaxScaler().fit(features)
    features = minmax_scale.transform(features)
    chi2score = chi2(features, classifications)[0]

    chi2score = chi2score.reshape(1, -1).T
    minmax_scale = preprocessing.MinMaxScaler().fit(chi2score)
    chi2score = minmax_scale.transform(chi2score)

    model = ExtraTreesClassifier()
    model.fit(features, classifications)
    feature_importance_ = model.feature_importances_.reshape(1, -1).T
    minmax_scale = preprocessing.MinMaxScaler().fit(feature_importance_)
    feature_importance_ = minmax_scale.transform(feature_importance_)

    anovaFScore = f_classif(features, classifications)[0]
    anovaFScore = anovaFScore.reshape(1, -1).T
    minmax_scale = preprocessing.MinMaxScaler().fit(anovaFScore)
    anovaFScore = minmax_scale.transform(anovaFScore)


    # Chi-Squared plotting code adapted from:
    # http://mark-kay.net/2014/03/10/using-chi-square-tfid-vectorization/

    if plot:
        plt.figure(figsize=(16,30))
        wscores = zip(featureNames, chi2score)
        wchi2 = sorted(wscores, key=lambda x:x[1])
        topchi2 = zip(*wchi2[-100:])
        x = range(len(topchi2[1]))
        labels = topchi2[0]
        plt.barh(x,topchi2[1], align='center', alpha=0.2, color='g')
        plt.plot(topchi2[1], x, '-o', markersize=5, alpha=0.8, color='g')
        plt.yticks(x, labels)
        plt.xlabel('$\chi^2$')

        #plt.figure(figsize=(16,30))
        wscores = zip(featureNames, feature_importance_)
        wchi2 = wscores#wchi2 = sorted(wscores, key=lambda x:x[1])
        topchi2 = zip(*wchi2[-100:])
        x = range(len(topchi2[1]))
        labels = topchi2[0]
        plt.barh(x,topchi2[1], align='center', alpha=0.2, color='b')
        plt.plot(topchi2[1], x, '-o', markersize=5, alpha=0.8, color='b')
        plt.yticks(x, labels)
        plt.xlabel('$\chi^2$')

        wscores = zip(featureNames, anovaFScore)
        wchi2 = wscores#wchi2 = sorted(wscores, key=lambda x:x[1])
        topchi2 = zip(*wchi2[-100:])
        x = range(len(topchi2[1]))
        labels = topchi2[0]
        plt.barh(x,topchi2[1], align='center', alpha=0.2, color='r')
        plt.plot(topchi2[1], x, '-o', markersize=5, alpha=0.8, color='r')
        plt.yticks(x, labels)
        plt.xlabel('$\chi^2$')
        plt.show()
