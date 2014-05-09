import numpy as np
import matplotlib.pyplot as plt

#data_file = '/Users/ejhumphrey/Dropbox/NYU/2014_01_Spring/msaf/exp2b_subset_annotator_annotator_metrics.npz'
data_file = '/Users/uri/Dropbox/NYU/Publications/ISMIR2014-NietoHumphreyFarboodBello/exp2b.npz'
x = np.load(data_file)
d1, d2 = np.triu_indices(6,1)
stats = np.array([x[n, d1, d2, 0] for n in range(50)])

c_idx = np.zeros(50, dtype=bool)
c_idx[np.array([8, 9, 10, 14, 37])] = True
h_idx = np.invert(c_idx)

num_bins = 10
b_h, e = np.histogram(stats[h_idx].flatten(), bins=num_bins, range=(0, 1))
b_c, e = np.histogram(stats[c_idx].flatten(), bins=num_bins, range=(0, 1))

fig = plt.figure(figsize=(6,3), dpi=120)
# ax = fig.gca()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
width = 1.0 / num_bins
ax1.bar(e[:-1], b_c / float(np.sum(b_c)), width=width, fc='b', alpha=0.5)
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 0.4)
ax1.set_ylabel("Easy Tracks")
ax1.set_yticks(np.arange(0, 4.1, 1)*0.1)
ax2.bar(e[:-1], b_h / float(np.sum(b_h)), width=width, fc='g', alpha=0.5)
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 0.4)
ax2.set_ylabel("Hard Tracks")
ax2.set_yticks(np.arange(0, 4.1, 1)*0.1)
plt.suptitle("Agreement of Reported Tags")
plt.xlabel("Mean Mutual Agreement")
plt.subplots_adjust(bottom=0.2)
plt.show()
