from __future__ import division
import numpy as np
import pysndfile
import matplotlib.pyplot as plt
import pdb
import logging
from multiprocessing import Pool, cpu_count
from scipy.stats import skew, tvar, kurtosis
from scipy.signal import decimate
import collections


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
    audioSamplerate = audioFile.samplerate()
    # TODO: Apply downsampling here...
    resampleRatio = segmentation['originalSR'] / segmentation['downsampledSR']
    if resampleRatio % 1.0:
        raise ValueError("Resample ratio is not an integer for audio file {0}".format(audioPath))
    audioData = decimate(audioData, int(resampleRatio), zero_phase=True)
    audioSamplerate // resampleRatio
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

        # Shannon Energy (Directly on PCG signal)
        perSegFeatures['s1SEngy'][i] = (-1/s1.size) * np.sum((s1**2)*np.log(s1**2))
        perSegFeatures['sysSEngy'][i] = (-1/s1.size) * np.sum((sys**2)*np.log(sys**2))
        perSegFeatures['s2SEngy'][i] = (-1/s1.size) * np.sum((s2**2)*np.log(s2**2))
        perSegFeatures['diaSEngy'][i] = (-1/s1.size) * np.sum((dia**2)*np.log(dia**2))

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

        # Spectral Flatness
        perSegFeatures['s1Flat'][i] = np.mean(s1FFT)/(np.e**np.mean(np.log(s1FFT)))
        perSegFeatures['sysFlat'][i] = np.mean(sysFFT)/(np.e**np.mean(np.log(sysFFT)))
        perSegFeatures['s2Flat'][i] = np.mean(s2FFT)/(np.e**np.mean(np.log(s2FFT)))
        perSegFeatures['diaFlat'][i] = np.mean(diaFFT)/(np.e**np.mean(np.log(diaFFT)))


        # Spectral Spread
        # Spectral Flatness
        # Spectral Centroid
        i += 1

    avrS1ZeroX = np.nanmean(perSegFeatures['s1ZeroX'])
    avrSys1ZeroX = np.nanmean(perSegFeatures['sysZeroX'])
    avrS2ZeroX = np.nanmean(perSegFeatures['s2ZeroX'])
    avrDiaZeroX = np.nanmean(perSegFeatures['diaZeroX'])

    return 0

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
'''
def generateFeatures(dataFilepaths, output_dir, parallelize=False):
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


