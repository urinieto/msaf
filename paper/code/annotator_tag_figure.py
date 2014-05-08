"""Call from MSAF top-level, like

~/msaf $ python paper/code/annotator_tag_figure.py

"""
import json
import numpy as np

from matplotlib import pyplot as plt

CONTROL_IDX = np.array([8, 9, 10, 14, 37])
FIGSIZE = (6, 4)

tag_json_file = "experiment/results/merged_tags_ejh_resolved.json"
with open(tag_json_file) as fp:
    tags = json.load(fp)

tag_act_matrix = np.zeros([50, 5, 5])
tag_names = ['annotator', 'audio_quality', 'form', 'instrumentation', 'style']
tag_idx = dict([(k, n) for n, k in enumerate(tag_names)])
for track_idx, tag_set in enumerate(tags.values()):
    for a_idx, annot in enumerate(tag_set):
        for tag in annot.split(", "):
            if tag == '':
                continue
            tag_act_matrix[track_idx, a_idx, tag_idx[tag.split("-")[0]]] = 1

# fig = plt.figure(figsize=FIGSIZE)  #, dpi=150)
# ax = fig.gca()
# ax.imshow(
#     tag_act_matrix.mean(axis=1).T,
#     cmap=plt.get_cmap("binary"),
#     interpolation='nearest',
#     aspect='auto')
# ax.set_yticks(range(len(tag_names)))
# ax.set_yticklabels(tag_names)
# ax.set_xlabel("Tracks")
# plt.show()

tag_counts = dict()
for track_idx, tag_set in enumerate(tags.values()):
    for a_idx, annot in enumerate(tag_set):
        for tag in annot.split(", "):
            if tag == '':
                continue
            if "(" in tag:
                tag = tag.split("(")[0]
            if not tag in tag_counts:
                tag_counts[tag] = 0
            tag_counts[tag] += 1

colorset = ['b', 'g', 'r', 'y', 'm']
final_labels = ["Annotator", "Audio Quality", "Form", "Instrumentation",
                "Style"]
width = 1.0
tag_colors = dict([(n, c) for n, c in zip(tag_names, colorset)])
tag_keys = tag_counts.keys()
tag_keys.sort()

fig = plt.figure(dpi=120, figsize=FIGSIZE)
ax = fig.gca()
prev_tag = None
labels = []
for n, k in enumerate(tag_keys):
    ret = ax.bar((n - 0.5)*width,
           tag_counts[k],
           width=width,
           fc=tag_colors[k.split('-')[0]])
    if prev_tag != tag_colors[k.split('-')[0]]:
        labels.append(ret[0])
        prev_tag = tag_colors[k.split('-')[0]]
ax.set_xlim(-width, width*len(tag_keys))
ax.set_xticks(range(len(tag_keys)))
for i in xrange(len(tag_keys)):
    tag_keys[i] = tag_keys[i].split("-")[1]
ax.set_xticklabels(tag_keys, rotation=75, ha='right')
ax.set_ylabel("Count")
fig.tight_layout()
plt.gcf().subplots_adjust(top=0.9)
plt.xlim(-1, 22)
ax.legend(tuple(labels), tuple(final_labels), prop={'size':10})
plt.title("Reported Tags")

plt.show()
