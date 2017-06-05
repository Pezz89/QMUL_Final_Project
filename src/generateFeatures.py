import numpy as np
import pandas
import pdb

def parse_segmentation_file(segPath):
    '''
    Returns a dictionary with elements for each header value in csv file and a
    data element for segmentation values, stored ina numpy array
    '''
    with open(segPath, 'r') as csvfile:
        originalSR, downsampledSR, heartRate = np.genfromtxt(csvfile, delimiter=",", max_rows=1, dtype=float, unpack=True)
        csvData = np.genfromtxt(csvfile, delimiter=",",  dtype=int)
        return {
            "originalSR": originalSR,
            "downsampledSR": downsampledSR,
            "data": csvData
        }


def calculateFeatures(name, audioPath, segPath):
    segmentation = parse_segmentation_file(segPath)
    pdb.set_trace()

def generateFeatures(dataFilepaths, output_dir):
    '''
    Processes filepath dictionary to generate a set of features for each file
    '''
    for pcgData in dataFilepaths:
        calculateFeatures(pcgData['name'],pcgData['audio'],pcgData['seg'])


