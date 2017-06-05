import numpy as np
import glob
import pdb

def calculateFeatures(name, audioPath, segPath):
    csv = np.genfromtxt(segPath, delimiter=",", dtype=int)

def generateFeatures(dataFilepaths, output_dir):
    '''
    Processes filepath dictionary to generate a set of features for each file
    '''
    pdb.set_trace()
    for pcgData in dataFilepaths:
        calculateFeatures(pcgData['name'],pcgData['audio'],pcgData['seg'])


