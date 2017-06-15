import numpy as np
import pandas as pd
import pdb
'''
Function for resampling rows with "abnormal" classification of a pandas
dataframe using bootstrap resampling
'''
def bootstrapResample(features, classification):
    # Count number of normal classifications
    n = np.sum(classification == -1)
    # Get features for all abnormal records
    abnormal = features.ix[classification[classification == 1].keys()]
    normal = features.ix[classification[classification == -1].keys()]

    # Resample all abnormal samples with replacement to balance the dataset
    abnormal = abnormal.sample(n=n, replace=True)
    # Join resampled abnormal records with normal records
    resampled_features = pd.concat([abnormal, normal])
    # Sort records in-place
    resampled_features.sort_index(inplace=True)
    # Get classification for all resampled features
    classification = classification[resampled_features.index]
    return resampled_features, classification


'''
Function for resampling rows with "normal" classification of a pandas dataframe
using jacknife resampling
'''
def jacknifeResample(features, classification):
    # Count number of abnormal classifications
    n = np.sum(classification == 1)
    # Get features for all abnormal records
    abnormal = features.ix[classification[classification == 1].keys()]
    normal = features.ix[classification[classification == -1].keys()]

    # Resample all normal samples with replacement to balance the dataset
    normal = normal.sample(n=n)
    # Join resampled normal records with abnormal records
    resampled_features = pd.concat([abnormal, normal])
    # Sort records in-place
    resampled_features.sort_index(inplace=True)
    # Get classification for all resampled features
    classification = classification[resampled_features.index]
    return resampled_features, classification


'''
Function for resampling rows with "normal" and "abnormal" classification of a
pandas dataframe using a combination of jacknife and bootstrap resampling
Bootstrap resampling is used to increase number of abnormal samples
Jacknife is used to decrease number of normal samples
Mix determines the amount of each method to use to balance the database.
'''
def combinationResample(features, classification, mix=0.5):
    if not 0.0 <= mix <= 1.0:
        raise ValueError("mix must be in range 0.0-1.0. mix={0}".format(mix))
    # Count number of abnormal classifications
    abnormal_n = np.sum(classification == 1)

    # Count number of normal classifications
    normal_n = np.sum(classification == -1)

    diff = normal_n - abnormal_n

    n = int(abnormal_n + (diff * mix))
    # Get features for all abnormal records
    abnormal = features.ix[classification[classification == 1].keys()]
    normal = features.ix[classification[classification == -1].keys()]

    # Resample all abnormal samples with replacement to balance the dataset
    abnormal = abnormal.sample(n=n, replace=True)
    # Resample all normal samples with replacement to balance the dataset
    normal = normal.sample(n=n)
    # Join resampled abnormal records with normal records
    resampled_features = pd.concat([abnormal, normal])
    # Sort records in-place
    resampled_features.sort_index(inplace=True)
    # Get classification for all resampled features
    classification = classification[resampled_features.index]
    return resampled_features, classification

