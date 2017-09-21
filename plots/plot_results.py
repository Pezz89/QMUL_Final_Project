import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import numpy as np

fig = plt.figure() # create a plot figure

plt.subplot(1, 3, 1) #
x = np.arange(6)
y = np.array([0.5395, 0.4896, 0.5673, 0.5173, 0.5869, 0.5492,])
e = np.array([0.0104, 0.0129, 0.0298, 0.0223, 0.0300, 0.0140])

plt.errorbar(x, y, yerr=e, fmt='o', color='black', ecolor='darkgray', elinewidth=3, capsize=0)
plt.xticks(x)
x = ['A', 'B', 'C', 'D', 'E', 'F']
plt.xticks(np.arange(6), x)
plt.title('Leave-one-out Accuracy');

plt.subplot(1, 3, 2) #
x = np.arange(6)
y = np.array([0.7281, 0.8664, 0.6775, 0.7865, 0.5397, 0.7387,])
e = np.array([0.0164, 0.0240, 0.0208, 0.0218, 0.0459, 0.0493])

plt.errorbar(x, y, yerr=e, fmt='o', color='black', ecolor='darkgray', elinewidth=3, capsize=0)
plt.xticks(x)
x = ['A', 'B', 'C', 'D', 'E', 'F']
plt.xticks(np.arange(6), x)
plt.title('Leave-one-out Sensitivity');

plt.subplot(1, 3, 3) #
x = np.arange(6)
y = np.array([0.3509, 0.1127, 0.4571, 0.2481, 0.6340, 0.3596,])
e = np.array([0.0264, 0.012, 0.0571, 0.0416, 0.0387, 0.0464])

plt.errorbar(x, y, yerr=e, fmt='o', color='black', ecolor='darkgray', elinewidth=3, capsize=0)
plt.xticks(x)
x = ['A', 'B', 'C', 'D', 'E', 'F']
plt.xticks(np.arange(6), x)
plt.title('Leave-one-out Specificity');
plt.tight_layout()
plt.show()

########################################################################
fig = plt.figure() # create a plot figure

plt.subplot(3, 1, 1) #
x = np.arange(10)+1
y = np.array([0.7969, 0.8049, 0.8043, 0.8111, 0.8095, 0.7999, 0.8061, 0.8150, 0.8140, 0.7928])
e = np.array([0.0246, 0.0244, 0.0153, 0.0295, 0.0261, 0.0208, 0.0299, 0.0198, 0.0245, 0.0224])

plt.errorbar(x, y, yerr=e, fmt='o', color='black', ecolor='darkgray', elinewidth=3, capsize=0)
plt.xticks(x)
plt.xticks(np.arange(10)+1, x)
plt.title('10-fold Accuracy');

plt.subplot(3, 1, 2) #
x = np.arange(10)+1
y = np.array([0.8121, 0.8164, 0.8193, 0.8184, 0.8158, 0.8061, 0.8325, 0.8421, 0.8246, 0.7798])
e = np.array([0.0420, 0.0360, 0.0302, 0.0634, 0.0484, 0.0438, 0.0546, 0.0321, 0.0474, 0.0302])

plt.errorbar(x, y, yerr=e, fmt='o', color='black', ecolor='darkgray', elinewidth=3, capsize=0)
plt.xticks(x)
plt.xticks(np.arange(10)+1, x)
plt.title('10-fold Sensitivity');

plt.subplot(3, 1, 3) #
x = np.arange(10)+1
y = np.array([0.7818, 0.7935, 0.7894, 0.8037, 0.8033, 0.7937, 0.7798, 0.7878, 0.8035, 0.8059])
e = np.array([0.0293, 0.0267, 0.0208, 0.0280, 0.0226, 0.0214, 0.0229, 0.0206, 0.0219, 0.0228])

plt.errorbar(x, y, yerr=e, fmt='o', color='black', ecolor='darkgray', elinewidth=3, capsize=0)
plt.xticks(x)
plt.xticks(np.arange(10), x)
plt.tight_layout()

plt.title('10-fold Specificity');
fig.set_size_inches(9*1.3, 3*1.3*3)
