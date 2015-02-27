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


def resample_mx(X, incolpos, outcolpos):
    """
    Y = resample_mx(X, incolpos, outcolpos)
    X is taken as a set of columns, each starting at 'time'
    colpos, and continuing until the start of the next column.
    Y is a similar matrix, with time boundaries defined by
    outcolpos.  Each column of Y is a duration-weighted average of
    the overlapping columns of X.
    2010-04-14 Dan Ellis dpwe@ee.columbia.edu  based on samplemx/beatavg
    -> python: TBM, 2011-11-05, TESTED
    """
    noutcols = len(outcolpos)
    Y = np.zeros((X.shape[0], noutcols))
    # assign 'end times' to final columns
    if outcolpos.max() > incolpos.max():
        incolpos = np.concatenate([incolpos, [outcolpos.max()]])
        X = np.concatenate([X, X[:, -1].reshape(X.shape[0], 1)], axis=1)
    outcolpos = np.concatenate([outcolpos, [outcolpos[-1]]])
    # durations (default weights) of input columns)
    incoldurs = np.concatenate([np.diff(incolpos), [1]])

    for c in range(noutcols):
        firstincol = np.where(incolpos <= outcolpos[c])[0][-1]
        firstincolnext = np.where(incolpos < outcolpos[c+1])[0][-1]
        lastincol = max(firstincol, firstincolnext)
        # default weights
        wts = copy.deepcopy(incoldurs[firstincol:lastincol+1])
        # now fix up by partial overlap at ends
        if len(wts) > 1:
            wts[0] = wts[0] - (outcolpos[c] - incolpos[firstincol])
            wts[-1] = wts[-1] - (incolpos[lastincol+1] - outcolpos[c+1])
        wts = wts * 1. / sum(wts)
        Y[:, c] = np.dot(X[:, firstincol:lastincol+1], wts)
    # done
    return Y


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


def write_audio_boundaries(audio, est_times, out_file, fs, offset=0):
    """Writes the estimated boundary times into the output file."""
    audio_bounds = mir_eval.sonify.clicks(est_times + offset, fs)
    audio_bounds = audio_bounds[:min(len(audio_bounds), len(audio))]
    audio = audio[:min(len(audio_bounds), len(audio))]
    audio[:len(audio_bounds)] += audio_bounds
    scipy.io.wavfile.write(out_file, fs, audio)
