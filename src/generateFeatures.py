from __future__ import division
import numpy as np
import pysndfile
import matplotlib.pyplot as plt
import pdb
import logging
from multiprocessing import Pool, cpu_count
from scipy.stats import skew


logger = logging.getLogger(__name__)

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
    logger.debug("Calculating features for: {0}".format(audioPath))
    segmentation = parse_segmentation_file(segPath)
    # Get audio data from PCG file
    audioFile = pysndfile.PySndfile(audioPath, 'r')
    audioData = audioFile.read_frames()
    # TODO: Apply downsampling here...
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
    m_RR        = np.round(np.mean(np.diff(segs[:,0])))                     # mean value of RR intervals
    sd_RR       = np.round(np.std(np.diff(segs[:,0])))                      # standard deviation (SD) value of RR intervals
    mean_IntS1  = np.round(np.mean(segs[:,1]-segs[:,0]))                    # mean value of S1 intervals
    sd_IntS1    = np.round(np.std(segs[:,1]-segs[:,0]))                     # SD value of S1 intervals
    mean_IntS2  = np.round(np.mean(segs[:,3]-segs[:,2]))                    # mean value of S2 intervals
    sd_IntS2    = np.round(np.std(segs[:,3]-segs[:,2]))                     # SD value of S2 intervals
    mean_IntSys = np.round(np.mean(segs[:,2]-segs[:,1]))                    # mean value of systole intervals
    sd_IntSys   = np.round(np.std(segs[:,2]-segs[:,1]))                     # SD value of systole intervals

    mean_IntDia = np.round(np.mean((np.roll(segs[:,0],-1)-segs[:,3])[:-1])) # mean value of diastole intervals
    mean_IntDia = np.round(np.mean((np.roll(segs[:,0],-1)-segs[:,3])[:-1])) # SD value of diastole intervals

    # If the signal exceeds 60 heart cycles, analyse only the first 60
    maxCycles = 60 if segs.shape[0]-1 > 60 else segs.shape[0]-1

    # Allocate memory for segment specific features
    s1ZeroX = np.empty(maxCycles)
    sysZeroX = np.empty(maxCycles)
    s2ZeroX = np.empty(maxCycles)
    diaZeroX = np.empty(maxCycles)

    s1RMS = np.empty(maxCycles)
    sysRMS = np.empty(maxCycles)
    s2RMS = np.empty(maxCycles)
    diaRMS = np.empty(maxCycles)

    i = 0
    while i < maxCycles:
        # Get audio signal for each segment
        s1 = audioData[segs[i,0]:segs[i,1]]
        sys = audioData[segs[i,1]:segs[i,2]]
        s2 = audioData[segs[i,2]:segs[i,3]]
        dia = audioData[segs[i,3]:segs[i+1,0]]

        # Zero-crossing
        s1ZeroX[i] = np.sum(np.abs(np.diff(s1)>0))/s1.size
        sysZeroX[i] = np.sum(np.abs(np.diff(sys)>0))/sys.size
        s2ZeroX[i] = np.sum(np.abs(np.diff(s2)>0))/s2.size
        diaZeroX[i] = np.sum(np.abs(np.diff(dia)>0))/dia.size

        # RMS
        s1RMS[i] = np.sqrt(np.mean(s1**2))
        sysRMS[i] = np.sqrt(np.mean(sys**2))
        s2RMS[i] = np.sqrt(np.mean(s2**2))
        diaRMS[i] = np.sqrt(np.mean(dia**2))

        # Shannon Energy (Directly on PCG signal)
        s1SEngy = (s1**2)*np.log(s1**2)
        sysSEngy = (sys**2)*np.log(sys**2)
        s2SEngy = (s2**2)*np.log(s2**2)
        diaSEngy = (dia**2)*np.log(dia**2)

        # Time duration
        s1Dur = s1.size
        sysDur = sys.size
        s2Dur = s2.size
        diaDur = dia.size

        # Skewness
        s1Skew = skew(s1)
        sysSkew = skew(sys)
        s2Skew = skew(s2)
        diaSkew = skew(dia)

        # Variance
        # Spectral Spread
        # Spectral Flatness
        # Spectral Centroid
        i += 1

    avrS1ZeroX = np.nanmean(s1ZeroX)
    avrSys1ZeroX = np.nanmean(sysZeroX)
    avrS2ZeroX = np.nanmean(s2ZeroX)
    avrDiaZeroX = np.nanmean(diaZeroX)

    return 0

def calculateFeatures_helper(args):
    '''
    Helper function to allow for parallelization of feature extraction
    '''
    return calculateFeatures(*args)

def generateFeatures(dataFilepaths, output_dir, parallelize=False):
    '''
    Processes filepath dictionary to generate a set of features for each file
    '''
    if parallelize:
        args = []
        pool = Pool(cpu_count())
        for pcgData in dataFilepaths:
            args.append((pcgData['name'],pcgData['audio'],pcgData['seg']))
        result = pool.map(calculateFeatures_helper, args)
        print result
    else:
        for pcgData in dataFilepaths:
            calculateFeatures(pcgData['name'],pcgData['audio'],pcgData['seg'])


