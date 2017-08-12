from __future__ import division
import numpy as np
import pysndfile
import matplotlib.pyplot as plt
import logging
from pathos.multiprocessing import Pool, cpu_count
from scipy.stats import skew, tvar, kurtosis, moment
from scipy.signal import decimate
from sklearn import preprocessing
import collections
import os
import pandas as pd
import pathops
from scipy.stats import entropy
from pyeeg import samp_entropy
from librosa.feature import mfcc
import io
import functools
import sys

import pywt

logger = logging.getLogger(__name__)

from scipy.signal import butter, lfilter

genfromtxt_old = np.genfromtxt
@functools.wraps(genfromtxt_old)
def genfromtxt_py3_fixed(f, encoding="utf-8", *args, **kwargs):
  if isinstance(f, io.TextIOBase):
    if hasattr(f, "buffer") and hasattr(f.buffer, "raw") and \
    isinstance(f.buffer.raw, io.FileIO):
      # Best case: get underlying FileIO stream (binary!) and use that
      fb = f.buffer.raw
      # Reset cursor on the underlying object to match that on wrapper
      fb.seek(f.tell())
      result = genfromtxt_old(fb, *args, **kwargs)
      # Reset cursor on wrapper to match that of the underlying object
      f.seek(fb.tell())
    else:
      # Not very good but works: Put entire contents into BytesIO object,
      # otherwise same ideas as above
      old_cursor_pos = f.tell()
      fb = io.BytesIO(bytes(f.read(), encoding=encoding))
      result = genfromtxt_old(fb, *args, **kwargs)
      f.seek(old_cursor_pos + fb.tell())
  else:
    result = genfromtxt_old(f, *args, **kwargs)
  return result

if sys.version_info >= (3,):
  np.genfromtxt = genfromtxt_py3_fixed

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
    with io.open(segPath, 'r') as csvfile:
        originalSR, downsampledSR, heartRate = np.genfromtxt(csvfile, delimiter=",", max_rows=1, dtype=float, unpack=True)
        csvData = np.genfromtxt(csvfile, delimiter=",",  dtype=int)
        return {
            "originalSR": originalSR,
            "downsampledSR": downsampledSR,
            "data": csvData,
            "heartRate": heartRate
        }


def wrcoef(X, coef_type, coeffs, wavename, level):
    N = np.array(X).size
    a, ds = coeffs[0], list(reversed(coeffs[1:]))

    if coef_type =='a':
        return pywt.upcoef('a', a, wavename, level=level)[:N]
    elif coef_type == 'd':
        return pywt.upcoef('d', ds[level-1], wavename, level=level)[:N]
    else:
        raise ValueError("Invalid coefficient type: {}".format(coef_type))


def calculateFeatures(name, audioPath, segPath):
    logger.debug("Calculating features for: {0}".format(os.path.relpath(audioPath)))
    segmentation = parse_segmentation_file(segPath)
    # Get audio data from PCG file
    audioFile = pysndfile.PySndfile(audioPath, 'r')
    audioData = audioFile.read_frames()
    audioSamplerate = audioFile.samplerate()
    resampleRatio = segmentation['originalSR'] / segmentation['downsampledSR']
    if resampleRatio % 1.0:
        raise ValueError("Resample ratio is not an integer for audio file {0}".format(audioPath))
    #audioData = decimate(audioData, int(resampleRatio), zero_phase=True)
    #audioSamplerate = audioSamplerate // resampleRatio
    audioData = butter_bandpass_filter(audioData, 25, 600, audioSamplerate, order=4)
    segData = segmentation['data']

    # Organise segments into a 4*N array, where each column represents the S1,
    # Systole, S2 and Diastole segments for N rows of complete heart cycles
    firstS1 = np.where(segData[:, 1] == 1)[0][0]
    lastDiastole = np.where(segData[:, 1] == 4)[0][-1]
    segs = segData[firstS1:lastDiastole+1, 0].reshape(segData[firstS1:lastDiastole+1, 1].size // 4, 4)

    if False:
        plt.plot(audioData)
        plt.step(segData[:, 0]*int(segmentation['originalSR']/segmentation['downsampledSR']), segData[:, 1])
        plt.show()

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

        # Signal Max
        perSegFeatures['s1Max'][i] = np.max(s1)
        perSegFeatures['sysMax'][i] = np.max(sys)
        perSegFeatures['s2Max'][i] = np.max(s2)
        perSegFeatures['diaMax'][i] = np.max(dia)

        # Signal Mean
        perSegFeatures['s1Mean'][i] = np.mean(s1)
        perSegFeatures['sysMean'][i] = np.mean(sys)
        perSegFeatures['s2Mean'][i] = np.mean(s2)
        perSegFeatures['diaMean'][i] = np.mean(dia)

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

        # Shannon Entropy
        perSegFeatures['s1ShanEnt'][i] = entropy(s1**2)
        perSegFeatures['sysShanEnt'][i] = entropy(sys**2)
        perSegFeatures['s2ShanEnt'][i] = entropy(s2**2)
        perSegFeatures['diaShanEnt'][i] = entropy(dia**2)

        # Total power in time domain
        perSegFeatures['TPTs1'][i] = (np.linalg.norm(s1)**2)/s1.size;
        perSegFeatures['TPTsys'][i] = (np.linalg.norm(sys)**2)/sys.size;
        perSegFeatures['TPTs2'][i] = (np.linalg.norm(s2)**2)/s2.size;
        perSegFeatures['TPTdia'][i] = (np.linalg.norm(dia)**2)/dia.size;

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

        # Total power in frequency domain
        perSegFeatures['TPFs1'][i] = np.real(np.sum(s1FFT*np.conj(s1FFT))/(s1FFTLength**2))
        perSegFeatures['TPFsys'][i] = np.real(np.sum(sysFFT*np.conj(sysFFT))/(sysFFTLength**2))
        perSegFeatures['TPFs2'][i] = np.real(np.sum(s2FFT*np.conj(s2FFT))/(s2FFTLength**2))
        perSegFeatures['TPFdia'][i] = np.real(np.sum(diaFFT*np.conj(diaFFT))/(diaFFTLength**2))

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

        # =====================================================================
        # MFCC Features
        # =====================================================================



        n_mfcc = 13
        s1Mel = np.hstack(mfcc(y=s1, sr=audioSamplerate, n_mfcc=n_mfcc, hop_length=s1.size+1))
        for ind, m in enumerate(s1Mel):
            perSegFeatures['s1MFCC{}'.format(ind)] = m
        sysMel = np.hstack(mfcc(y=sys, sr=audioSamplerate, n_mfcc=n_mfcc, hop_length=sys.size+1))
        for ind, m in enumerate(sysMel):
            perSegFeatures['sysMFCC{}'.format(ind)] = m
        s2Mel = np.hstack(mfcc(y=s2, sr=audioSamplerate, n_mfcc=n_mfcc, hop_length=s2.size+1))
        for ind, m in enumerate(s2Mel):
            perSegFeatures['s2MFCC{}'.format(ind)] = m
        diaMel = np.hstack(mfcc(y=dia, sr=audioSamplerate, n_mfcc=n_mfcc, hop_length=dia.size+1))
        for ind, m in enumerate(diaMel):
            perSegFeatures['diaMFCC{}'.format(ind)] = m

        i += 1

    features['m_Ratio_SysRR']   = np.mean(perSegFeatures['R_SysRR']);  # mean value of the interval ratios between systole and RR in each heart beat
    features['sd_Ratio_SysRR']  = np.std(perSegFeatures['R_SysRR']);   # SD value of the interval ratios between systole and RR in each heart beat
    features['m_Ratio_DiaRR']   = np.mean(perSegFeatures['R_DiaRR']);  # mean value of the interval ratios between diastole and RR in each heart beat
    features['sd_Ratio_DiaRR']  = np.std(perSegFeatures['R_DiaRR']);   # SD value of the interval ratios between diastole and RR in each heart beat
    features['m_Ratio_SysDia']  = np.mean(perSegFeatures['R_SysDia']); # mean value of the interval ratios between systole and diastole in each heart beat
    features['sd_Ratio_SysDia'] = np.std(perSegFeatures['R_SysDia']);  # SD value of the interval ratios between systole and diastole in each heart beat

    # =====================================================================
    # Wavelet-based Features
    # =====================================================================

    level = 5
    wavelet_type = 'db4'
    #create wavelet coefficients: cAn, cDn, cD(n-1)... cD1
    coeffs = pywt.wavedec(audioData, wavelet_type,level=level)
    cA5, cD5, cD4, cD3, cD2, cD1 = coeffs

    # Shannon entropy of wavelet coefficients for entire signal
    features['D1Shan'] = entropy(cD1**2)
    features['D2Shan'] = entropy(cD2**2)
    features['D3Shan'] = entropy(cD3**2)
    features['D4Shan'] = entropy(cD4**2)
    features['D5Shan'] = entropy(cD5**2)
    features['A5Shan'] = entropy(cA5**2)

    # Reconstruct signal at each branch from individual wavelet coefficients
    A5 = wrcoef(audioData,'a', coeffs,  wavelet_type, 5)
    D5 = wrcoef(audioData,'d', coeffs,  wavelet_type, 5)
    D4 = wrcoef(audioData,'d', coeffs,  wavelet_type, 4)
    D3 = wrcoef(audioData,'d', coeffs,  wavelet_type, 3)
    D2 = wrcoef(audioData,'d', coeffs,  wavelet_type, 2)
    D1 = wrcoef(audioData,'d', coeffs,  wavelet_type, 1)

    if False:
        import matplotlib.pyplot as plt
        plt.plot(audioData)
        plt.plot(X)
        plt.show()


    s1Slices = []
    sysSlices = []
    s2Slices = []
    diaSlices = []

    i = 0
    while i < maxCycles:
        # Generate slices for all segments of each heart beat cycle
        s1 = slice(segs[i,0], segs[i,1])
        sys = slice(segs[i,1], segs[i,2])
        s2 = slice(segs[i,2], segs[i,3])
        dia = slice(segs[i,3], segs[i+1,0])
        s1Slices.append(s1)
        sysSlices.append(sys)
        s2Slices.append(s2)
        diaSlices.append(dia)

        # Shannon entropy of reconstructed signals from wavelet decomposition
        # per segment
        perSegFeatures['AvrA5s1Shan'][i] = entropy(A5[s1]**2)
        perSegFeatures['AvrD5s1Shan'][i] = entropy(D5[s1]**2)
        perSegFeatures['AvrD4s1Shan'][i] = entropy(D4[s1]**2)
        perSegFeatures['AvrD3s1Shan'][i] = entropy(D3[s1]**2)
        perSegFeatures['AvrD2s1Shan'][i] = entropy(D2[s1]**2)
        perSegFeatures['AvrD1s1Shan'][i] = entropy(D1[s1]**2)

        perSegFeatures['AvrA5sysShan'][i] = entropy(A5[sys]**2)
        perSegFeatures['AvrD5sysShan'][i] = entropy(D5[sys]**2)
        perSegFeatures['AvrD4sysShan'][i] = entropy(D4[sys]**2)
        perSegFeatures['AvrD3sysShan'][i] = entropy(D3[sys]**2)
        perSegFeatures['AvrD2sysShan'][i] = entropy(D2[sys]**2)
        perSegFeatures['AvrD1sysShan'][i] = entropy(D1[sys]**2)

        perSegFeatures['AvrA5s2Shan'][i] = entropy(A5[s2]**2)
        perSegFeatures['AvrD5s2Shan'][i] = entropy(D5[s2]**2)
        perSegFeatures['AvrD4s2Shan'][i] = entropy(D4[s2]**2)
        perSegFeatures['AvrD3s2Shan'][i] = entropy(D3[s2]**2)
        perSegFeatures['AvrD2s2Shan'][i] = entropy(D2[s2]**2)
        perSegFeatures['AvrD1s2Shan'][i] = entropy(D1[s2]**2)

        perSegFeatures['AvrA5diaShan'][i] = entropy(A5[dia]**2)
        perSegFeatures['AvrD5diaShan'][i] = entropy(D5[dia]**2)
        perSegFeatures['AvrD4diaShan'][i] = entropy(D4[dia]**2)
        perSegFeatures['AvrD3diaShan'][i] = entropy(D3[dia]**2)
        perSegFeatures['AvrD2diaShan'][i] = entropy(D2[dia]**2)
        perSegFeatures['AvrD1diaShan'][i] = entropy(D1[dia]**2)

        i += 1

    # Convert slices to numpy compatible indexes
    s1Slices = np.r_[tuple(s1Slices)]
    sysSlices = np.r_[tuple(sysSlices)]
    s2Slices = np.r_[tuple(s2Slices)]
    diaSlices = np.r_[tuple(diaSlices)]

    # Total shannon entropy accross all decomposed segments
    features['TotA5s1Shan']  = entropy(A5[s1Slices]**2)
    features['TotA5sysShan'] = entropy(A5[sysSlices]**2)
    features['TotA5s2Shan']  = entropy(A5[s2Slices]**2)
    features['TotA5diaShan'] = entropy(A5[diaSlices]**2)

    features['TotD5s1Shan']  = entropy(D5[s1Slices]**2)
    features['TotD5sysShan'] = entropy(D5[sysSlices]**2)
    features['TotD5s2Shan']  = entropy(D5[s2Slices]**2)
    features['TotD5diaShan'] = entropy(D5[diaSlices]**2)

    features['TotD4s1Shan']  = entropy(D4[s1Slices]**2)
    features['TotD4sysShan'] = entropy(D4[sysSlices]**2)
    features['TotD4s2Shan']  = entropy(D4[s2Slices]**2)
    features['TotD4diaShan'] = entropy(D4[diaSlices]**2)

    features['TotD3s1Shan']  = entropy(D3[s1Slices]**2)
    features['TotD3sysShan'] = entropy(D3[sysSlices]**2)
    features['TotD3s2Shan']  = entropy(D3[s2Slices]**2)
    features['TotD3diaShan'] = entropy(D3[diaSlices]**2)

    features['TotD2s1Shan']  = entropy(D2[s1Slices]**2)
    features['TotD2sysShan'] = entropy(D2[sysSlices]**2)
    features['TotD2s2Shan']  = entropy(D2[s2Slices]**2)
    features['TotD2diaShan'] = entropy(D2[diaSlices]**2)

    features['TotD1s1Shan']  = entropy(D1[s1Slices]**2)
    features['TotD1sysShan'] = entropy(D1[sysSlices]**2)
    features['TotD1s2Shan']  = entropy(D1[s2Slices]**2)
    features['TotD1diaShan'] = entropy(D1[diaSlices]**2)


    # Average all per-segment features and store as output features
    for key in perSegFeatures.keys():
        features[key] = np.nanmean(perSegFeatures[key])

    return pd.Series(features)


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
