"""
Useful functions that are common in MSAF
"""

__author__ = "Oriol Nieto"
__copyright__ = "Copyright 2014, Music and Audio Research Lab (MARL)"
__license__ = "GPL"
__version__ = "1.0"
__email__ = "oriol@nyu.edu"

import copy
import mir_eval
import numpy as np
import os
import scipy.io.wavfile

import msaf

def lognormalize_chroma(C):
    """Log-normalizes chroma such that each vector is between -80 to 0."""
    C += np.abs(C.min()) + 0.1
    C = C / C.max(axis=0)
    C = 80 * np.log10(C)  # Normalize from -80 to 0
    return C


def normalize_chroma(C):
    """Normalizes chroma such that each vector is between 0 to 1."""
    C += np.abs(C.min())
    C = C/C.max(axis=0)
    return C


def normalize_matrix(X):
    """Nomalizes a matrix such that it's maximum value is 1 and minimum is 0."""
    X += np.abs(X.min())
    X /= X.max()
    return X


def ensure_dir(directory):
    """Makes sure that the given directory exists."""
    if not os.path.exists(directory):
        os.makedirs(directory)


def chroma_to_tonnetz(C):
    """Transforms chromagram to Tonnetz (Harte, Sandler, 2006)."""
    N = C.shape[0]
    T = np.zeros((N, 6))

    r1 = 1      # Fifths
    r2 = 1      # Minor
    r3 = 0.5    # Major

    # Generate Transformation matrix
    phi = np.zeros((6, 12))
    for i in range(6):
        for j in range(12):
            if i % 2 == 0:
                fun = np.sin
            else:
                fun = np.cos

            if i < 2:
                phi[i, j] = r1 * fun(j * 7 * np.pi / 6.)
            elif i >= 2 and i < 4:
                phi[i, j] = r2 * fun(j * 3 * np.pi / 2.)
            else:
                phi[i, j] = r3 * fun(j * 2 * np.pi / 3.)

    # Do the transform to tonnetz
    for i in range(N):
        for d in range(6):
            denom = float(C[i, :].sum())
            if denom == 0:
                T[i, d] = 0
            else:
                T[i, d] = 1 / denom * (phi[d, :] * C[i, :]).sum()

    return T


def times_to_intervals(times):
    """Given a set of times, convert them into intervals.

    Parameters
    ----------
    times: np.array(N)
        A set of times.

    Returns
    -------
    inters: np.array(N-1, 2)
        A set of intervals.
    """
    return np.asarray(zip(times[:-1], times[1:]))


def intervals_to_times(inters):
    """Given a set of intervals, convert them into times.

    Parameters
    ----------
    inters: np.array(N-1, 2)
        A set of intervals.

    Returns
    -------
    times: np.array(N)
        A set of times.
    """
    return np.concatenate((inters.flatten()[::2], [inters[-1, -1]]), axis=0)


def get_num_frames(dur, anal):
    """Given the duration of a track and a dictionary containing analysis
    info, return the number of frames."""
    total_samples = dur * anal["sample_rate"]
    return int(total_samples / anal["hop_size"])


def get_time_frames(dur, anal):
    """Gets the time frames and puts them in a numpy array."""
    n_frames = get_num_frames(dur, anal)
    return np.linspace(0, dur, num=n_frames)


def remove_empty_segments(times, labels):
    """Removes empty segments if needed."""
    inters = times_to_intervals(times)
    new_inters = []
    new_labels = []
    for inter, label in zip(inters, labels):
        if inter[0] < inter[1]:
            new_inters.append(inter)
            new_labels.append(label)
    return intervals_to_times(np.asarray(new_inters)), new_labels


def sonify_clicks(audio, clicks, out_file, fs, offset=0):
    """Sonifies the estimated times into the output file.

    Parameters
    ----------
    audio: np.array
        Audio samples of the input track.
    clicks: np.array
        Click positions in seconds.
    out_file: str
        Path to the output file.
    fs: int
        Sample rate.
    offset: float
        Offset of the clicks with respect to the audio.
    """
    # Generate clicks
    audio_clicks = mir_eval.sonify.clicks(clicks + offset, fs)

    # Create array to store the audio plus the clicks
    out_audio = np.zeros(max(len(audio), len(audio_clicks)))

    # Assign the audio and the clicks
    out_audio[:len(audio)] = audio
    out_audio[:len(audio_clicks)] += audio_clicks

    # Write to file
    scipy.io.wavfile.write(out_file, fs, out_audio)


def synchronize_labels(new_bound_idxs, old_bound_idxs, old_labels, N):
    """Synchronizes the labels from the old_bound_idxs to the new_bound_idxs.

    Parameters
    ----------
    new_bound_idxs: np.array
        New indeces to synchronize with.
    old_bound_idxs: np.array
        Old indeces, same shape as labels + 1.
    old_labels: np.array
        Labels associated to the old_bound_idxs.
    N: int
        Total number of frames.

    Returns
    -------
    new_labels: np.array
        New labels, synchronized to the new boundary indeces.
    """
    assert len(old_bound_idxs) - 1 == len(old_labels)

    # Construct unfolded labels array
    unfold_labels = np.zeros(N)
    for i, (bound_idx, label) in \
            enumerate(zip(old_bound_idxs[:-1], old_labels)):
        unfold_labels[bound_idx:old_bound_idxs[i+1]] = label

    # Constuct new labels
    new_labels = np.zeros(len(new_bound_idxs) - 1)
    for i, bound_idx in enumerate(new_bound_idxs[:-1]):
        new_labels[i] = np.median(unfold_labels[bound_idx:new_bound_idxs[i+1]])

    return new_labels
