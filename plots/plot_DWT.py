import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import decimate
sns.set()
import numpy as np
import pdb
import io
import pysndfile

################################################################################
# Generic imports
################################################################################
# Wavelet transform library
import pywt

fig = plt.figure() # create a plot figure

def wrcoef(X, coef_type, coeffs, wavename, level):
    N = np.array(X).size
    a, ds = coeffs[0], list(reversed(coeffs[1:]))

    if coef_type =='a':
        return pywt.upcoef('a', a, wavename, level=level)[:N]
    elif coef_type == 'd':
        return pywt.upcoef('d', ds[level-1], wavename, level=level)[:N]
    else:
        raise ValueError("Invalid coefficient type: {}".format(coef_type))

def parse_segmentation_file(segPath):
    with io.open(segPath, 'r') as csvfile:
        originalSR, downsampledSR, heartRate = np.genfromtxt(csvfile, delimiter=",", max_rows=1, dtype=float, unpack=True)
        csvData = np.genfromtxt(csvfile, delimiter=",",  dtype=int)
        return {
            "originalSR": originalSR,
            "downsampledSR": downsampledSR,
            "data": csvData,
            "heartRate": heartRate
        }

# Get segmentations from segmentation file generated in Matlab
segmentation = parse_segmentation_file('../out_optimized/seg/a0003_segs.csv')
segs = segmentation['data']
# Read audio data from PCG file
audioFile = pysndfile.PySndfile('../demo_dataset/training-a/a0003.wav', 'r')
audioData = audioFile.read_frames()
audioSamplerate = audioFile.samplerate()
audioSamplerate = audioFile.samplerate()
# Calculate the ratio between samplerate used for segmentation and tyhe
# file's original sample rate
resampleRatio = segmentation['originalSR'] / segmentation['downsampledSR']
if resampleRatio % 1.0:
    raise ValueError("Resample ratio is not an integer for audio file {0}".format(audioPath))
# Resample audio to segmentation samplerate
audioData = decimate(audioData, int(resampleRatio), zero_phase=True)
# Set variable for the resampled rate
audioSamplerate = audioSamplerate // resampleRatio



count = 0
num_segs = 2
found_segs = num_segs
index =10
for i in segs:
    if i[1] == 1:
        if found_segs == num_segs:
            start = i[0]
            found_segs = 0
        found_segs += 1
    if i[1] == 4:
        end = i[0]
        if count == index:
            break
        count += 1
length = end-start
start -= int(round(length * 0.1))
end += int(round(length * 0.1))
seg = segs[(segs[:, 0] >= start) & (segs[:, 0] <= end)]
audioData = audioData[start:end]

level = 5
wavelet_type = 'db4'
#create wavelet coefficients: cAn, cDn, cD(n-1)... cD1
coeffs = pywt.wavedec(audioData, wavelet_type,level=level)
cA5, cD5, cD4, cD3, cD2, cD1 = coeffs

# Reconstruct signal at each branch from individual wavelet coefficients
A5 = wrcoef(audioData,'a', coeffs,  wavelet_type, 5)
D5 = wrcoef(audioData,'d', coeffs,  wavelet_type, 5)
D4 = wrcoef(audioData,'d', coeffs,  wavelet_type, 4)
D3 = wrcoef(audioData,'d', coeffs,  wavelet_type, 3)
D2 = wrcoef(audioData,'d', coeffs,  wavelet_type, 2)
D1 = wrcoef(audioData,'d', coeffs,  wavelet_type, 1)



for ind, wave in enumerate(reversed([A5, D5, D4, D3, D2, D1])):
    pdb.set_trace()
    plt.subplot(6, 1, 1+ind) #
    plt.plot(wave)
    labels = ['S1', 'Sys', 'S2', 'Dia']
    for i, j in seg:
        plt.axvline(x=i-start, color='r', linestyle='--')
        #plt.text(i-start+50*0.25,-0.11, labels[j-1],rotation=45)
    plt.yticks([])
    plt.xlim([0, audioData.size])
    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='off') # labels along the bottom edge are off
    plt.tick_params(
        axis='y',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='off') # labels along the bottom edge are off

plt.tight_layout()
plt.show()

