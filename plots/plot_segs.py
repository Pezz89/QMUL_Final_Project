import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import decimate
sns.set()
import numpy as np
import pdb
import io
import pysndfile

fig = plt.figure() # create a plot figure


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
plt.plot(audioData)
labels = ['S1', 'Sys', 'S2', 'Dia']
for i, j in seg:
    plt.axvline(x=i-start, color='r', linestyle='--')
    plt.text(i-start+50*0.25,-0.11, labels[j-1],rotation=45)
plt.xlim([0, audioData.size])
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    labelbottom='off') # labels along the bottom edge are off
plt.show()

