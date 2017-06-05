from __future__ import division
import numpy as np
import pandas
import pysndfile
import matplotlib.pyplot as plt
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
    # Get audio data from PCG file
    audioFile = pysndfile.PySndfile(audioPath, 'r')
    audioData = audioFile.read_frames()
    segData = segmentation['data']

    # Organise segments into a 4*N array, where each column represents the S1,
    # Systole, S2 and Diastole segments for N rows of complete heart cycles
    firstS1 = np.where(segData[:, 1] == 1)[0][0]
    lastDiastole = np.where(segData[:, 1] == 4)[0][-1]
    segs = segData[firstS1:lastDiastole+1, 0].reshape(segData[firstS1:lastDiastole+1, 1].size // 4, 4)
    '''
    plt.plot(audioData)
    plt.step(segData[:, 0]*int(segmentation['originalSR']/segmentation['downsampledSR']), segData[:, 1])
    plt.show()
    '''
    # Calculate basic features. This code was adapted directly from the
    # Physionet challenge 2016 example entry: https://www.physionet.org/challenge/2016/sample2016.zip
    m_RR        = np.round(np.mean(np.diff(segs[:,0])))             # mean value of RR intervals
    sd_RR       = np.round(np.std(np.diff(segs[:,0])))              # standard deviation (SD) value of RR intervals
    mean_IntS1  = np.round(np.mean(segs[:,1]-segs[:,0]))            # mean value of S1 intervals
    sd_IntS1    = np.round(np.std(segs[:,1]-segs[:,0]))             # SD value of S1 intervals
    mean_IntS2  = np.round(np.mean(segs[:,3]-segs[:,2]))            # mean value of S2 intervals
    sd_IntS2    = np.round(np.std(segs[:,3]-segs[:,2]))             # SD value of S2 intervals
    mean_IntSys = np.round(np.mean(segs[:,2]-segs[:,1]))            # mean value of systole intervals
    sd_IntSys   = np.round(np.std(segs[:,2]-segs[:,1]))             # SD value of systole intervals
    mean_IntDia = np.round(np.mean(segs[1:,0]-segs[0:-2,3]))        # mean value of diastole intervals
    sd_IntDia   = np.round(np.std(segs[1:,0]-segs[0:-2,3]))         # SD value of diastole intervals


def generateFeatures(dataFilepaths, output_dir):
    '''
    Processes filepath dictionary to generate a set of features for each file
    '''
    for pcgData in dataFilepaths:
        calculateFeatures(pcgData['name'],pcgData['audio'],pcgData['seg'])


