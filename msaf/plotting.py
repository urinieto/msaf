"""This script contains methods to plot multiple aspects of the results
of MSAF.
"""

import jams
import logging
import mir_eval
import numpy as np
import os

# Local stuff
from msaf import io
from msaf import utils

translate_ids = {
    "2dfmc": "2D-FMC",
    "cnmf3": "C-NMF",
    "foote": "Ckboard",
    "levy": "CC",
    "cc": "CC",
    "olda": "OLDA",
    "serra": "SF",
    "sf": "SF",
    "siplca": "SI-PLCA"
}


def _plot_formatting(title, est_file, algo_ids, last_bound, N, output_file):
    """Formats the plot with the correct axis labels, title, ticks, and
    so on."""
    import matplotlib.pyplot as plt
    if title is None:
        title = os.path.basename(est_file).split(".")[0]
    plt.title(title)
    plt.yticks(np.arange(0, 1, 1 / float(N)) + 1 / (float(N) * 2))
    plt.gcf().subplots_adjust(bottom=0.22)
    plt.gca().set_yticklabels(algo_ids)
    plt.xlabel("Time (seconds)")
    plt.xlim((0, last_bound))
    plt.tight_layout()
    if output_file is not None:
        plt.savefig(output_file)
    plt.show()


def plot_boundaries(all_boundaries, est_file, algo_ids=None, title=None,
                    output_file=None):
    """Plots all the boundaries.

    Parameters
    ----------
    all_boundaries: list
        A list of np.arrays containing the times of the boundaries, one array
        for each algorithm.
    est_file: str
        Path to the estimated file (JSON file)
    algo_ids : list
        List of algorithm ids to to read boundaries from.
        If None, all algorithm ids are read.
    title : str
        Title of the plot. If None, the name of the file is printed instead.
    """
    import matplotlib.pyplot as plt
    N = len(all_boundaries)  # Number of lists of boundaries
    if algo_ids is None:
        algo_ids = io.get_algo_ids(est_file)

    # Translate ids
    for i, algo_id in enumerate(algo_ids):
        algo_ids[i] = translate_ids[algo_id]
    algo_ids = ["GT"] + algo_ids

    figsize = (6, 4)
    plt.figure(1, figsize=figsize, dpi=120, facecolor='w', edgecolor='k')
    for i, boundaries in enumerate(all_boundaries):
        color = "b"
        if i == 0:
            color = "g"
        for b in boundaries:
            plt.axvline(b, i / float(N), (i + 1) / float(N), color=color)
        plt.axhline(i / float(N), color="k", linewidth=1)

    # Format plot
    _plot_formatting(title, est_file, algo_ids, all_boundaries[0][-1], N,
                     output_file)


def plot_labels(all_labels, gt_times, est_file, algo_ids=None, title=None,
                output_file=None):
    """Plots all the labels.

    Parameters
    ----------
    all_labels: list
        A list of np.arrays containing the labels of the boundaries, one array
        for each algorithm.
    gt_times: np.array
        Array with the ground truth boundaries.
    est_file: str
        Path to the estimated file (JSON file)
    algo_ids : list
        List of algorithm ids to to read boundaries from.
        If None, all algorithm ids are read.
    title : str
        Title of the plot. If None, the name of the file is printed instead.
    """
    import matplotlib.pyplot as plt
    N = len(all_labels)  # Number of lists of labels
    if algo_ids is None:
        algo_ids = io.get_algo_ids(est_file)

    # Translate ids
    for i, algo_id in enumerate(algo_ids):
        algo_ids[i] = translate_ids[algo_id]
    algo_ids = ["GT"] + algo_ids

    # Index the labels to normalize them
    for i, labels in enumerate(all_labels):
        all_labels[i] = mir_eval.util.index_labels(labels)[0]

    # Get color map
    cm = plt.get_cmap('gist_rainbow')
    max_label = max(max(labels) for labels in all_labels)

    # To intervals
    gt_inters = utils.times_to_intervals(gt_times)

    # Plot labels
    figsize = (6, 4)
    plt.figure(1, figsize=figsize, dpi=120, facecolor='w', edgecolor='k')
    for i, labels in enumerate(all_labels):
        for label, inter in zip(labels, gt_inters):
            plt.axvspan(inter[0], inter[1], ymin=i / float(N),
                        ymax=(i + 1) / float(N), alpha=0.6,
                        color=cm(label / float(max_label)))
        plt.axhline(i / float(N), color="k", linewidth=1)

    # Draw the boundary lines
    for bound in gt_times:
        plt.axvline(bound, color="g")

    # Format plot
    _plot_formatting(title, est_file, algo_ids, gt_times[-1], N,
                     output_file)


def plot_one_track(file_struct, est_times, est_labels, boundaries_id, labels_id,
                   title=None):
    """Plots the results of one track, with ground truth if it exists."""
    import matplotlib.pyplot as plt
    # Set up the boundaries id
    bid_lid = boundaries_id
    if labels_id is not None:
        bid_lid += " + " + labels_id
    try:
        # Read file
        jam = jams.load(file_struct.ref_file)
        ann = jam.search(namespace='segment_.*')[0]
        ref_inters, ref_labels = ann.to_interval_values()

        # To times
        ref_times = utils.intervals_to_times(ref_inters)
        all_boundaries = [ref_times, est_times]
        all_labels = [ref_labels, est_labels]
        algo_ids = ["GT", bid_lid]
    except:
        logging.warning("No references found in %s. Not plotting groundtruth"
                        % file_struct.ref_file)
        all_boundaries = [est_times]
        all_labels = [est_labels]
        algo_ids = [bid_lid]

    N = len(all_boundaries)

    # Index the labels to normalize them
    for i, labels in enumerate(all_labels):
        all_labels[i] = mir_eval.util.index_labels(labels)[0]

    # Get color map
    cm = plt.get_cmap('gist_rainbow')
    max_label = max(max(labels) for labels in all_labels)

    figsize = (8, 4)
    plt.figure(1, figsize=figsize, dpi=120, facecolor='w', edgecolor='k')
    for i, boundaries in enumerate(all_boundaries):
        color = "b"
        if i == 0:
            color = "g"
        for b in boundaries:
            plt.axvline(b, i / float(N), (i + 1) / float(N), color=color)
        if labels_id is not None:
            labels = all_labels[i]
            inters = utils.times_to_intervals(boundaries)
            for label, inter in zip(labels, inters):
                plt.axvspan(inter[0], inter[1], ymin=i / float(N),
                            ymax=(i + 1) / float(N), alpha=0.6,
                            color=cm(label / float(max_label)))
        plt.axhline(i / float(N), color="k", linewidth=1)

    # Format plot
    _plot_formatting(title, os.path.basename(file_struct.audio_file), algo_ids,
                     all_boundaries[0][-1], N, None)


def plot_tree(T, res=None, title=None, cmap_id="Pastel2"):
    """Plots a given tree, containing hierarchical segmentation.

    Parameters
    ----------
    T: mir_eval.segment.tree
        A tree object containing the hierarchical segmentation.
    res: float
        Frame-rate resolution of the tree (None to use seconds).
    title: str
        Title for the plot. `None` for no title.
    cmap_id: str
        Color Map ID
    """
    import matplotlib.pyplot as plt
    def round_time(t, res=0.1):
        v = int(t / float(res)) * res
        return v

    # Get color map
    cmap = plt.get_cmap(cmap_id)

    # Get segments by level
    level_bounds = []
    for level in T.levels:
        if level == "root":
            continue
        segments = T.get_segments_in_level(level)
        level_bounds.append(segments)

    # Plot axvspans for each segment
    B = float(len(level_bounds))
    #plt.figure(figsize=figsize)
    for i, segments in enumerate(level_bounds):
        labels = utils.segment_labels_to_floats(segments)
        for segment, label in zip(segments, labels):
            #print i, label, cmap(label)
            if res is None:
                start = segment.start
                end = segment.end
                xlabel = "Time (seconds)"
            else:
                start = int(round_time(segment.start, res=res) / res)
                end = int(round_time(segment.end, res=res) / res)
                xlabel = "Time (frames)"
            plt.axvspan(start, end,
                        ymax=(len(level_bounds) - i) / B,
                        ymin=(len(level_bounds) - i - 1) / B,
                        facecolor=cmap(label))

    # Plot labels
    L = float(len(T.levels) - 1)
    plt.yticks(np.linspace(0, (L - 1) / L, num=L) + 1 / L / 2.,
               T.levels[1:][::-1])
    plt.xlabel(xlabel)
    if title is not None:
        plt.title(title)
    plt.gca().set_xlim([0, end])
