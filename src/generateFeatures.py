from __future__ import division
import numpy as np
import pysndfile
import matplotlib.pyplot as plt
import pdb
import logging
from pathos.multiprocessing import Pool, cpu_count
from scipy.stats import skew, tvar, kurtosis
from scipy.signal import decimate
from sklearn import preprocessing
import collections
import os
import pandas as pd
import pathops
from scipy.stats import entropy
from pyeeg import samp_entropy


logger = logging.getLogger(__name__)

from scipy.signal import butter, lfilter

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

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
            "data": csvData,
            "heartRate": heartRate
        }


def calculateFeatures(name, audioPath, segPath):
    logger.debug("Calculating features for: {0}".format(os.path.relpath(audioPath)))
    segmentation = parse_segmentation_file(segPath)
    # Get audio data from PCG file
    audioFile = pysndfile.PySndfile(audioPath, 'r')
    audioData = audioFile.read_frames()
    audioSamplerate = audioFile.samplerate()
    # TODO: Apply downsampling here...
    resampleRatio = segmentation['originalSR'] / segmentation['downsampledSR']
    if resampleRatio % 1.0:
        raise ValueError("Resample ratio is not an integer for audio file {0}".format(audioPath))
    audioData = decimate(audioData, int(resampleRatio), zero_phase=True)
    audioSamplerate = audioSamplerate // resampleRatio
    audioData = butter_bandpass_filter(audioData, 25, 400, audioSamplerate, order=4)
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

    # =========================================================================
    # Global Features - Calculated over entire signal
    # =========================================================================

    features = collections.defaultdict(lambda: None)

    # Get heart rate feature calculated during segmentation
    features['heartRate'] = segmentation['heartRate']

    # Calculate basic features. This code was adapted directly from the
    # Physionet challenge 2016 example entry: https://www.physionet.org/challenge/2016/sample2016.zip
    features['m_RR']        = np.round(np.mean(np.diff(segs[:,0])))                     # mean value of RR intervals
    features['sd_RR']       = np.round(np.std(np.diff(segs[:,0])))                      # standard deviation (SD) value of RR intervals
    features['mean_IntS1']  = np.round(np.mean(segs[:,1]-segs[:,0]))                    # mean value of S1 intervals
    features['sd_IntS1']    = np.round(np.std(segs[:,1]-segs[:,0]))                     # SD value of S1 intervals
    features['mean_IntS2']  = np.round(np.mean(segs[:,3]-segs[:,2]))                    # mean value of S2 intervals
    features['sd_IntS2']    = np.round(np.std(segs[:,3]-segs[:,2]))                     # SD value of S2 intervals
    features['mean_IntSys'] = np.round(np.mean(segs[:,2]-segs[:,1]))                    # mean value of systole intervals
    features['sd_IntSys']   = np.round(np.std(segs[:,2]-segs[:,1]))                     # SD value of systole intervals

    features['mean_IntDia'] = np.round(np.mean((np.roll(segs[:,0],-1)-segs[:,3])[:-1])) # mean value of diastole intervals
    features['mean_IntDia'] = np.round(np.mean((np.roll(segs[:,0],-1)-segs[:,3])[:-1])) # SD value of diastole intervals


    # =========================================================================
    # Local Features - Calculated per segment
    # =========================================================================

    # If the signal exceeds 60 heart cycles, analyse only the first 60
    maxCycles = 60 if segs.shape[0]-1 > 60 else segs.shape[0]-1

    # Automatically allocate memory for segment specific features as they are
    # created
    perSegFeatures = collections.defaultdict(lambda: np.empty(maxCycles))

    i = 0
    while i < maxCycles:
        # Get audio signal for each segment
        s1 = audioData[segs[i,0]:segs[i,1]]
        sys = audioData[segs[i,1]:segs[i,2]]
        s2 = audioData[segs[i,2]:segs[i,3]]
        dia = audioData[segs[i,3]:segs[i+1,0]]

        perSegFeatures['R_SysRR'][i]  = (segs[i,2]-segs[i,1])/(segs[i+1,0]-segs[i,0])*100;
        perSegFeatures['R_DiaRR'][i]  = (segs[i+1,0]-segs[i,3])/(segs[i+1,0]-segs[i,0])*100;
        perSegFeatures['R_SysDia'][i] = perSegFeatures['R_SysRR'][i]/perSegFeatures['R_DiaRR'][i]*100;
        # =====================================================================
        # Time-domain Features
        # =====================================================================

        # Zero-crossing
        perSegFeatures['s1ZeroX'][i] = np.sum(np.abs(np.diff(s1)>0))/s1.size
        perSegFeatures['sysZeroX'][i] = np.sum(np.abs(np.diff(sys)>0))/sys.size
        perSegFeatures['s2ZeroX'][i] = np.sum(np.abs(np.diff(s2)>0))/s2.size
        perSegFeatures['diaZeroX'][i] = np.sum(np.abs(np.diff(dia)>0))/dia.size

        # RMS
        perSegFeatures['s1RMS'][i] = np.sqrt(np.mean(s1**2))
        perSegFeatures['sysRMS'][i] = np.sqrt(np.mean(sys**2))
        perSegFeatures['s2RMS'][i] = np.sqrt(np.mean(s2**2))
        perSegFeatures['diaRMS'][i] = np.sqrt(np.mean(dia**2))

        # Shannon Energy
        perSegFeatures['s1ShanEngy'][i] = (-1/s1.size) * np.sum((s1**2)*np.log(s1**2))
        perSegFeatures['sysShanEngy'][i] = (-1/s1.size) * np.sum((sys**2)*np.log(sys**2))
        perSegFeatures['s2ShanEngy'][i] = (-1/s1.size) * np.sum((s2**2)*np.log(s2**2))
        perSegFeatures['diaShanEngy'][i] = (-1/s1.size) * np.sum((dia**2)*np.log(dia**2))

        # Time duration
        perSegFeatures['s1Dur'][i] = s1.size
        perSegFeatures['sysDur'][i] = sys.size
        perSegFeatures['s2Dur'][i] = s2.size
        perSegFeatures['diaDur'][i] = dia.size

        # Skewness
        perSegFeatures['s1Skew'][i] = skew(s1)
        perSegFeatures['sysSkew'][i] = skew(sys)
        perSegFeatures['s2Skew'][i] = skew(s2)
        perSegFeatures['diaSkew'][i] = skew(dia)

        # Kurtosis
        perSegFeatures['s1Kurt'][i] = kurtosis(s1)
        perSegFeatures['sysKurt'][i] = kurtosis(sys)
        perSegFeatures['s2Kurt'][i] = kurtosis(s2)
        perSegFeatures['diaKurt'][i] = kurtosis(dia)

        # Variance
        perSegFeatures['s1Var'][i] = tvar(s1)
        perSegFeatures['sysVar'][i] = tvar(sys)
        perSegFeatures['s2Var'][i] = tvar(s2)
        perSegFeatures['diaVar'][i] = tvar(dia)

        # Sample Entropy
        perSegFeatures['s1SampEnt'][i] = samp_entropy(s1, 1, 0.2*np.std(s1))
        perSegFeatures['sysSampEnt'][i] = samp_entropy(sys, 1, 0.2*np.std(sys))
        perSegFeatures['s2SampEnt'][i] = samp_entropy(s2, 1, 0.2*np.std(s2))
        perSegFeatures['diaSampEnt'][i] = samp_entropy(dia, 1, 0.2*np.std(dia))

        # Sample Entropy
        perSegFeatures['s1ShanEnt'][i] = entropy(s1**2)
        perSegFeatures['sysShanEnt'][i] = entropy(sys**2)
        perSegFeatures['s2ShanEnt'][i] = entropy(s2**2)
        perSegFeatures['diaShanEnt'][i] = entropy(dia**2)

        # =====================================================================
        # Frequency-domain Features
        # =====================================================================

        # Calculate closest power of 2 to the segment size to use as FFT size
        s1FFTLength = nextpow2(s1.size)
        s1FFT= (np.fft.fft(s1,s1FFTLength)/s1.size)[:s1FFTLength//2];
        # Calculate center frequency of each bin
        fS1=audioSamplerate/2*np.linspace(0,1,s1FFTLength/2);

        sysFFTLength = nextpow2(sys.size)
        sysFFT= (np.fft.fft(sys,sysFFTLength)/sys.size)[:sysFFTLength//2];
        # Calculate center frequency of each bin
        fSys=audioSamplerate/2*np.linspace(0,1,sysFFTLength/2);

        s2FFTLength = nextpow2(s2.size)
        s2FFT= (np.fft.fft(s2,s2FFTLength)/s2.size)[:s2FFTLength//2];
        # Calculate center frequency of each bin
        fS2=audioSamplerate/2*np.linspace(0,1,s2FFTLength/2);

        diaFFTLength = nextpow2(dia.size)
        diaFFT= (np.fft.fft(dia,diaFFTLength)/dia.size)[:diaFFTLength//2];
        # Calculate center frequency of each bin
        fDia=audioSamplerate/2*np.linspace(0,1,diaFFTLength/2);

        s1Mag = np.abs(s1FFT)
        sysMag = np.abs(sysFFT)
        s2Mag = np.abs(s2FFT)
        diaMag = np.abs(diaFFT)

        # Spectral Flatness
        perSegFeatures['s1Flat'][i] = (np.e**np.mean(np.log(s1Mag)))/np.mean(s1Mag)
        perSegFeatures['sysFlat'][i] = (np.e**np.mean(np.log(sysMag)))/np.mean(sysMag)
        perSegFeatures['s2Flat'][i] = (np.e**np.mean(np.log(s2Mag)))/np.mean(s2Mag)
        perSegFeatures['diaFlat'][i] = (np.e**np.mean(np.log(diaMag)))/np.mean(diaMag)

        # Spectral Centroid
        perSegFeatures['s1Cent'][i] = np.sum(s1Mag*fS1) / np.sum(s1Mag)
        perSegFeatures['sysCent'][i] = np.sum(sysMag*fSys) / np.sum(sysMag)
        perSegFeatures['s2Cent'][i] = np.sum(s2Mag*fS2) / np.sum(s2Mag)
        perSegFeatures['diaCent'][i] = np.sum(diaMag*fDia) / np.sum(diaMag)

        # Spectral Spread
        perSegFeatures['s1Spread'][i] = spectralspread(s1Mag, fS1, perSegFeatures['s1Cent'][i])
        perSegFeatures['sysSpread'][i] = spectralspread(sysMag, fSys, perSegFeatures['sysCent'][i])
        perSegFeatures['s2Spread'][i] = spectralspread(s2Mag, fS2, perSegFeatures['s2Cent'][i])
        perSegFeatures['diaSpread'][i] = spectralspread(diaMag, fDia, perSegFeatures['diaCent'][i])


        i += 1

    features['m_Ratio_SysRR']   = np.mean(perSegFeatures['R_SysRR']);  # mean value of the interval ratios between systole and RR in each heart beat
    features['sd_Ratio_SysRR']  = np.std(perSegFeatures['R_SysRR']);   # SD value of the interval ratios between systole and RR in each heart beat
    features['m_Ratio_DiaRR']   = np.mean(perSegFeatures['R_DiaRR']);  # mean value of the interval ratios between diastole and RR in each heart beat
    features['sd_Ratio_DiaRR']  = np.std(perSegFeatures['R_DiaRR']);   # SD value of the interval ratios between diastole and RR in each heart beat
    features['m_Ratio_SysDia']  = np.mean(perSegFeatures['R_SysDia']); # mean value of the interval ratios between systole and diastole in each heart beat
    features['sd_Ratio_SysDia'] = np.std(perSegFeatures['R_SysDia']);  # SD value of the interval ratios between systole and diastole in each heart beat

    # Average all per-segment features and store as output features
    for key in perSegFeatures.keys():
        features[key] = np.nanmean(perSegFeatures[key])

    return pd.Series(features)

'''
Scale a pandas DataFrame of features between 0 and 1, maintaining DataFrame
datatype
'''
def normaliseFeatures(features):
    minmax_scale = preprocessing.MinMaxScaler().fit(features)
    features = pd.DataFrame(
        minmax_scale.transform(features),
        index=features.index,
        columns=features.columns
    )
    return features


'''
Calculate the spectral spread of a magnitude spectrum, given it's spectral
ceontroid
'''
def spectralspread(mag, freqs, centroid):
    a = (freqs-centroid)**2
    mag_sqrd = mag**2
    # Calculate the weighted mean
    return np.sqrt(np.sum(a*mag_sqrd) / np.sum(mag_sqrd))


'''
Find 2^n that is equal to or greater than.
'''
def nextpow2(i):
    n = 1
    while n < i: n *= 2
    return n


'''
Helper function to allow for parallelization of feature extraction
'''
def calculateFeatures_helper(args):
    return calculateFeatures(*args)


'''
Processes filepath dictionary to generate a set of features for each file
Return a Pandas DataFrame containing features for all PCG recordings
'''
def generateFeatures(dataFilepaths, output_dir, filename=None, parallelize=True, reanalyse=False):
    features = pd.DataFrame()
    outputFile = None
    # If a file of features has been generated previously...
    if filename:
        pathops.dir_must_exist(output_dir)
        outputFile = os.path.join(output_dir, filename)
        if not reanalyse:
            try:
                logger.debug("Attempting to load previously generated features from file: {0}".format(os.path.relpath(outputFile)))
                features = pd.read_pickle(outputFile)
                logger.debug("Features loaded from file succesfully".format(outputFile))
                #return features
            except IOError:
                logger.debug("No previously generated features loaded, generating new features...")
                pass
        else:
            logger.debug("Reanalysis flag is set, generating new features...")


    results = []
    args = []
    # Find all files that are in the current dataset that have not been
    # processed previously
    for pcgData in dataFilepaths:
        if pcgData['name'] not in features.index:
            args.append((pcgData['name'],pcgData['audio'],pcgData['seg']))

    # Filter features not in the current dataset
    keys = [x['name'] for x in dataFilepaths if x['name'] in features.index]
    features = features.ix[keys]

    if parallelize:
        pool = Pool(cpu_count())
        results = pool.map(calculateFeatures_helper, args)
        newFeatures = pd.DataFrame(results, map(lambda x: x[0], args))
    else:
        for arg in args:
            results.append(calculateFeatures(*arg))
        newFeatures = pd.DataFrame(results, map(lambda x: x[0], args))

    features = pd.concat([features, newFeatures])
    if outputFile:
        features.to_pickle(outputFile)

    if features.isnull().values.any():
        raise ValueError("Some features contain Nan values.")

    return features
