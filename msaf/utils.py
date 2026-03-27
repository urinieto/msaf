"""Useful functions that are common in MSAF."""

from __future__ import annotations

import os

import librosa
import mir_eval
import numpy as np
import scipy.io.wavfile


def lognormalize(F: np.ndarray, floor: float = 0.1, min_db: float = -80) -> np.ndarray:
    """Log-normalizes features such that each vector is between min_db to 0."""
    assert min_db < 0
    F = min_max_normalize(F, floor=floor)
    F = np.abs(min_db) * np.log10(F)
    return F


def min_max_normalize(F: np.ndarray, floor: float = 0.001) -> np.ndarray:
    """Normalizes features such that each vector is between floor to 1."""
    F = F.copy()
    F += -F.min() + floor
    F = F / F.max(axis=0)
    return F


def normalize(
    X: np.ndarray,
    norm_type: str | float | None,
    floor: float = 0.0,
    min_db: float = -80,
) -> np.ndarray:
    """Normalizes the given matrix of features.

    Parameters
    ----------
    X: np.ndarray
        Each row represents a feature vector.
    norm_type: {"min_max", "log", np.inf, -np.inf, 0, float > 0, None}
        - ``"min_max"``: Min/max scaling is performed
        - ``"log"``: Logarithmic scaling is performed
        - ``np.inf``: Maximum absolute value
        - ``-np.inf``: Minimum absolute value
        - ``0``: Number of non-zeros
        - float: Corresponding l_p norm.
        - None : No normalization is performed

    Returns
    -------
    norm_X: np.ndarray
        Normalized ``X`` according the the input parameters.
    """
    if isinstance(norm_type, str):
        if norm_type == "min_max":
            return min_max_normalize(X, floor=floor)
        if norm_type == "log":
            return lognormalize(X, floor=floor, min_db=min_db)
    return librosa.util.normalize(X, norm=norm_type, axis=1)


def ensure_dir(directory: str) -> None:
    """Makes sure that the given directory exists."""
    if not os.path.exists(directory):
        os.makedirs(directory)


def times_to_intervals(times: np.ndarray) -> np.ndarray:
    """Given a set of times, convert them into intervals.

    Parameters
    ----------
    times: np.ndarray(N)
        A set of times.

    Returns
    -------
    inters: np.ndarray(N-1, 2)
        A set of intervals.
    """
    return np.asarray(list(zip(times[:-1], times[1:])))


def intervals_to_times(inters: np.ndarray) -> np.ndarray:
    """Given a set of intervals, convert them into times.

    Parameters
    ----------
    inters: np.ndarray(N-1, 2)
        A set of intervals.

    Returns
    -------
    times: np.ndarray(N)
        A set of times.
    """
    return np.concatenate((inters.flatten()[::2], [inters[-1, -1]]), axis=0)


def remove_empty_segments(
    times: np.ndarray, labels: list | np.ndarray
) -> tuple[np.ndarray, list]:
    """Removes empty segments if needed."""
    assert len(times) - 1 == len(labels)
    inters = times_to_intervals(times)
    new_inters = []
    new_labels = []
    for inter, label in zip(inters, labels):
        if inter[0] < inter[1]:
            new_inters.append(inter)
            new_labels.append(label)
    return intervals_to_times(np.asarray(new_inters)), new_labels


def sonify_clicks(
    audio: np.ndarray,
    clicks: np.ndarray,
    out_file: str,
    fs: int,
    offset: float = 0,
) -> None:
    """Sonifies the estimated times into the output file.

    Parameters
    ----------
    audio: np.ndarray
        Audio samples of the input track.
    clicks: np.ndarray
        Click positions in seconds.
    out_file: str
        Path to the output file.
    fs: int
        Sample rate.
    offset: float
        Offset of the clicks with respect to the audio.
    """
    times = clicks + offset
    length = int(times.max() * fs + fs * 0.1 + 1)
    audio_clicks = mir_eval.sonify.clicks(times, fs, length=length)

    out_audio = np.zeros(max(len(audio), len(audio_clicks)))
    out_audio[: len(audio)] = audio
    out_audio[: len(audio_clicks)] += audio_clicks

    # Peak normalize
    out_audio /= np.abs(out_audio).max()

    amplitude = np.iinfo(np.int16).max
    data = (amplitude * out_audio).astype(np.int16)
    scipy.io.wavfile.write(out_file, fs, data)


def synchronize_labels(
    new_bound_idxs: np.ndarray,
    old_bound_idxs: np.ndarray,
    old_labels: np.ndarray,
    N: int,
) -> np.ndarray:
    """Synchronizes the labels from the old_bound_idxs to the new_bound_idxs.

    Parameters
    ----------
    new_bound_idxs: np.ndarray
        New indices to synchronize with.
    old_bound_idxs: np.ndarray
        Old indices, same shape as labels + 1.
    old_labels: np.ndarray
        Labels associated to the old_bound_idxs.
    N: int
        Total number of frames.

    Returns
    -------
    new_labels: np.ndarray
        New labels, synchronized to the new boundary indices.
    """
    assert len(old_bound_idxs) - 1 == len(old_labels)

    # Compute new labels by finding the median old label in each new segment
    new_labels = np.zeros(len(new_bound_idxs) - 1)
    for i, new_start in enumerate(new_bound_idxs[:-1]):
        new_end = new_bound_idxs[i + 1]
        # Find which old segments overlap with this new segment
        labels_in_range = []
        for j, old_start in enumerate(old_bound_idxs[:-1]):
            old_end = old_bound_idxs[j + 1]
            # Check overlap
            if old_end > new_start and old_start < new_end:
                overlap = min(old_end, new_end) - max(old_start, new_start)
                labels_in_range.extend([old_labels[j]] * max(1, int(overlap)))
        if labels_in_range:
            new_labels[i] = np.median(labels_in_range)

    return new_labels


def process_segmentation_level(
    est_idxs: np.ndarray,
    est_labels: np.ndarray,
    N: int,
    frame_times: np.ndarray,
    dur: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Processes a level of segmentation, and converts it into times.

    Parameters
    ----------
    est_idxs: np.ndarray
        Estimated boundaries in frame indices.
    est_labels: np.ndarray
        Estimated labels.
    N: int
        Number of frames in the whole track.
    frame_times: np.ndarray
        Time stamp for each frame.
    dur: float
        Duration of the audio track.

    Returns
    -------
    est_times: np.ndarray
        Estimated segment boundaries in seconds.
    est_labels: np.ndarray
        Estimated labels for each segment.
    """
    assert est_idxs[0] == 0 and est_idxs[-1] == N - 1
    assert len(est_idxs) - 1 == len(est_labels)

    # Add silences, if needed
    est_times = np.concatenate(([0], frame_times[est_idxs], [dur]))
    silence_label = np.max(est_labels) + 1
    est_labels = np.concatenate(([silence_label], est_labels, [silence_label]))

    # Remove empty segments if needed
    est_times, est_labels = remove_empty_segments(est_times, est_labels)

    assert np.allclose([est_times[0]], [0]) and np.allclose([est_times[-1]], [dur])

    return est_times, est_labels


def align_end_hierarchies(
    hier1: list[np.ndarray],
    hier2: list[np.ndarray],
    thres: float = 0.5,
) -> None:
    """Align the end of the hierarchies such that they end at the same exact
    second as long they have the same duration within a certain threshold.

    Parameters
    ----------
    hier1: list
        List containing hierarchical segment boundaries.
    hier2: list
        List containing hierarchical segment boundaries.
    thres: float > 0
        Threshold to decide whether two values are the same.
    """
    dur_h1 = hier1[0][-1]
    for hier in hier1:
        assert hier[-1] == dur_h1, "hier1 is not correctly formatted %s %s" % (
            hier[-1],
            dur_h1,
        )
    dur_h2 = hier2[0][-1]
    for hier in hier2:
        assert hier[-1] == dur_h2, "hier2 is not correctly formatted"

    if abs(dur_h1 - dur_h2) > thres:
        return

    for hier in hier1:
        hier[-1] = dur_h2
