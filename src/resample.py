import numpy as np
import pandas as pd
import pdb
'''
Function for resampling rows with "abnormal" classification of a pandas dataframe using bootstrap resampling
'''
def bootstrapResample(features, classification):
    # Count number of normal classifications
    # Count number of abnormal classifications
    # Abnormal
    classification[classification == 1]
    n = classification[classification == -1].size
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
