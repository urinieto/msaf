"""Plotting functions for MSAF results."""

from __future__ import annotations

import logging
import os

import jams
import matplotlib.pyplot as plt
import mir_eval
import numpy as np

from msaf import io, utils

translate_ids = {
    "2dfmc": "2D-FMC",
    "cnmf3": "C-NMF",
    "foote": "Ckboard",
    "levy": "CC",
    "cc": "CC",
    "olda": "OLDA",
    "serra": "SF",
    "sf": "SF",
    "siplca": "SI-PLCA",
}


def _plot_formatting(
    title: str | None,
    est_file: str,
    algo_ids: list[str],
    last_bound: float,
    N: int,
    output_file: str | None,
) -> None:
    """Format the plot with axis labels, title, ticks, etc."""
    if title is None:
        title = os.path.basename(est_file).split(".")[0]
    plt.title(title)
    plt.yticks(
        np.linspace(0, 1, N, endpoint=False) + 1 / (2 * N),
        algo_ids,
    )
    plt.xlabel("Time (seconds)")
    plt.xlim((0, last_bound))
    plt.tight_layout()
    if output_file is not None:
        plt.savefig(output_file)
    plt.show()


def plot_boundaries(
    all_boundaries: list[np.ndarray],
    est_file: str,
    algo_ids: list[str] | None = None,
    title: str | None = None,
    output_file: str | None = None,
) -> None:
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
    N = len(all_boundaries)
    if algo_ids is None:
        algo_ids = io.get_algo_ids(est_file)

    for i, algo_id in enumerate(algo_ids):
        algo_ids[i] = translate_ids[algo_id]
    algo_ids = ["GT"] + algo_ids

    plt.figure(figsize=(6, 4), dpi=120, facecolor="w", edgecolor="k")
    for i, boundaries in enumerate(all_boundaries):
        color = "g" if i == 0 else "b"
        for b in boundaries:
            plt.axvline(b, i / N, (i + 1) / N, color=color)
        plt.axhline(i / N, color="k", linewidth=1)

    _plot_formatting(title, est_file, algo_ids, all_boundaries[0][-1], N, output_file)


def plot_labels(
    all_labels: list[np.ndarray],
    gt_times: np.ndarray,
    est_file: str,
    algo_ids: list[str] | None = None,
    title: str | None = None,
    output_file: str | None = None,
) -> None:
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
    N = len(all_labels)
    if algo_ids is None:
        algo_ids = io.get_algo_ids(est_file)

    for i, algo_id in enumerate(algo_ids):
        algo_ids[i] = translate_ids[algo_id]
    algo_ids = ["GT"] + algo_ids

    # Index the labels to normalize them
    for i, labels in enumerate(all_labels):
        all_labels[i] = mir_eval.util.index_labels(labels)[0]

    cmap = plt.colormaps["tab10"]
    max_label = max(max(labels) for labels in all_labels)

    gt_inters = utils.times_to_intervals(gt_times)

    plt.figure(figsize=(6, 4), dpi=120, facecolor="w", edgecolor="k")
    for i, labels in enumerate(all_labels):
        for label, inter in zip(labels, gt_inters):
            plt.axvspan(
                inter[0],
                inter[1],
                ymin=i / N,
                ymax=(i + 1) / N,
                alpha=0.6,
                color=cmap(label / max(max_label, 1)),
            )
        plt.axhline(i / N, color="k", linewidth=1)

    for bound in gt_times:
        plt.axvline(bound, color="g")

    _plot_formatting(title, est_file, algo_ids, gt_times[-1], N, output_file)


def plot_one_track(
    file_struct,
    est_times: np.ndarray,
    est_labels: np.ndarray,
    boundaries_id: str,
    labels_id: str | None,
    title: str | None = None,
) -> None:
    """Plots the results of one track, with ground truth if it exists."""
    bid_lid = boundaries_id
    if labels_id is not None:
        bid_lid += " + " + labels_id
    try:
        jam = jams.load(file_struct.ref_file)
        ann = jam.search(namespace="segment_.*")[0]
        ref_inters, ref_labels = ann.to_interval_values()

        ref_times = utils.intervals_to_times(ref_inters)
        all_boundaries = [ref_times, est_times]
        all_labels = [ref_labels, est_labels]
        algo_ids = ["GT", bid_lid]
    except (FileNotFoundError, IndexError, jams.SchemaError) as e:
        logging.warning(
            "No references found in %s: %s. Not plotting ground truth.",
            file_struct.ref_file,
            e,
        )
        all_boundaries = [est_times]
        all_labels = [est_labels]
        algo_ids = [bid_lid]

    N = len(all_boundaries)

    for i, labels in enumerate(all_labels):
        all_labels[i] = mir_eval.util.index_labels(labels)[0]

    cmap = plt.colormaps["tab10"]
    max_label = max(max(labels) for labels in all_labels)

    plt.figure(figsize=(8, 4), dpi=120, facecolor="w", edgecolor="k")
    for i, boundaries in enumerate(all_boundaries):
        color = "g" if i == 0 else "b"
        for b in boundaries:
            plt.axvline(b, i / N, (i + 1) / N, color=color)
        if labels_id is not None:
            labels = all_labels[i]
            inters = utils.times_to_intervals(boundaries)
            for label, inter in zip(labels, inters):
                plt.axvspan(
                    inter[0],
                    inter[1],
                    ymin=i / N,
                    ymax=(i + 1) / N,
                    alpha=0.6,
                    color=cmap(label / max(max_label, 1)),
                )
        plt.axhline(i / N, color="k", linewidth=1)

    _plot_formatting(
        title,
        os.path.basename(file_struct.audio_file),
        algo_ids,
        all_boundaries[0][-1],
        N,
        None,
    )


def plot_tree(
    T,
    res: float | None = None,
    title: str | None = None,
    cmap_id: str = "Pastel2",
) -> None:
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

    def round_time(t: float, res: float = 0.1) -> float:
        return int(t / res) * res

    cmap = plt.colormaps[cmap_id]

    level_bounds = []
    for level in T.levels:
        if level == "root":
            continue
        segments = T.get_segments_in_level(level)
        level_bounds.append(segments)

    B = float(len(level_bounds))
    end = 0
    for i, segments in enumerate(level_bounds):
        # Convert segment labels to numeric values for colormap
        unique_labels = sorted(set(s.label for s in segments))
        label_map = {lbl: idx / max(len(unique_labels) - 1, 1) for idx, lbl in enumerate(unique_labels)}
        for segment in segments:
            label = label_map[segment.label]
            if res is None:
                start = segment.start
                end = segment.end
                xlabel = "Time (seconds)"
            else:
                start = int(round_time(segment.start, res=res) / res)
                end = int(round_time(segment.end, res=res) / res)
                xlabel = "Time (frames)"
            plt.axvspan(
                start,
                end,
                ymax=(len(level_bounds) - i) / B,
                ymin=(len(level_bounds) - i - 1) / B,
                facecolor=cmap(label),
            )

    L = float(len(T.levels) - 1)
    plt.yticks(np.linspace(0, (L - 1) / L, num=int(L)) + 1 / L / 2.0, T.levels[1:][::-1])
    plt.xlabel(xlabel)
    if title is not None:
        plt.title(title)
    plt.gca().set_xlim([0, end])
