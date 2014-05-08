"""Call from MSAF top-level, like

~/msaf $ python paper/code/annotator_as_GT_boxplots.py

"""
import json
import numpy as np

from matplotlib import pyplot as plt

CONTROL_IDX = np.array([8, 9, 10, 14, 37])
FIGSIZE = (8, 6)
DPI = 120
INCLUDE_AVERAGED = False

scores_json_file = "experiment/results/subset_f3_scores.json"

results = json.load(open(scores_json_file))
scores = np.array(results['data'])
mu = scores.mean(axis=0)

colorset = ['b', 'g', 'r', 'y', 'm']
fig = plt.figure(dpi=DPI)
ax = fig.gca()
width = 0.1
for ann_idx in range(mu.shape[0] + INCLUDE_AVERAGED - 1):
    handles = []
    for alg_idx in range(mu.shape[1]):
        handles.append(ax.bar(ann_idx + alg_idx*width,
                              mu[ann_idx, alg_idx],
                              width=width,
                              fc=colorset[alg_idx]))

ax.set_xticks(np.arange(6) + (width * 5)/2.0)
ax.set_xticklabels(range(1, 7))
ax.set_xlabel("Annotator Index")
ax.set_ylabel("F3-Score")
ax.set_ylim(mu.min()*0.85, mu.max()*1.01)
legend = ax.legend(
    [h[0] for h in handles],
    ["olda", "siplca", "serra", "levy", "foote"],
    loc='lower right', shadow=False)

fig.tight_layout()
plt.show()
