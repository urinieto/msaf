#!/usr/bin/env python
# coding: utf-8
"""
This script identifies the boundaries of a given track using the Serrà
method:

Serrà, J., Müller, M., Grosche, P., & Arcos, J. L. (2012). Unsupervised
Detection of Music Boundaries by Time Series Structure Features.
In Proc. of the 26th AAAI Conference on Artificial Intelligence
(pp. 1613–1619). Toronto, Canada.
"""
import cPickle as pickle
import logging
import os
import pylab as plt
import numpy as np
import librosa
from scipy.spatial import distance
from scipy import signal
from scipy.ndimage import filters

import msaf
from msaf.algorithms.interface import SegmenterInterface


def median_filter(X, M=8):
    """Median filter along the first axis of the feature matrix X."""
    for i in xrange(X.shape[1]):
        X[:, i] = filters.median_filter(X[:, i], size=M)
    return X


def gaussian_filter(X, M=8, axis=0):
    """Gaussian filter along the first axis of the feature matrix X."""
    for i in xrange(X.shape[axis]):
        if axis == 1:
            X[:, i] = filters.gaussian_filter(X[:, i], sigma=M / 2.)
        elif axis == 0:
            X[i, :] = filters.gaussian_filter(X[i, :], sigma=M / 2.)
    return X


def compute_gaussian_krnl(M):
    """Creates a gaussian kernel following Serra's paper."""
    g = signal.gaussian(M, M / 3., sym=True)
    G = np.dot(g.reshape(-1, 1), g.reshape(1, -1))
    G[M / 2:, :M / 2] = -G[M / 2:, :M / 2]
    G[:M / 2, M / 1:] = -G[:M / 2, M / 1:]
    return G


def compute_ssm(X, metric="seuclidean"):
    """Computes the self-similarity matrix of X."""
    D = distance.pdist(X, metric=metric)
    D = distance.squareform(D)
    D /= D.max()
    return 1 - D


def compute_nc(X):
    """Computes the novelty curve from the structural features."""
    N = X.shape[0]
    # nc = np.sum(np.diff(X, axis=0), axis=1) # Difference between SF's

    nc = np.zeros(N)
    for i in xrange(N - 1):
        nc[i] = distance.euclidean(X[i, :], X[i + 1, :])

    # Normalize
    nc += np.abs(nc.min())
    nc /= nc.max()
    return nc


def pick_peaks(nc, L=16, offset_denom=0.1):
    """Obtain peaks from a novelty curve using an adaptive threshold."""
    offset = nc.mean() * float(offset_denom)
    th = filters.median_filter(nc, size=L) + offset
    #th = filters.gaussian_filter(nc, sigma=L/2., mode="nearest") + offset
    #import pylab as plt
    #plt.plot(nc)
    #plt.plot(th)
    #plt.show()
    # th = np.ones(nc.shape[0]) * nc.mean() - 0.08
    peaks = []
    for i in xrange(1, nc.shape[0] - 1):
        # is it a peak?
        if nc[i - 1] < nc[i] and nc[i] > nc[i + 1]:
            # is it above the threshold?
            if nc[i] > th[i]:
                peaks.append(i)
    return peaks


def circular_shift(X):
    """Shifts circularly the X squre matrix in order to get a
        time-lag matrix."""
    N = X.shape[0]
    L = np.zeros(X.shape)
    for i in xrange(N):
        L[i, :] = np.asarray([X[(i + j) % N, j] for j in xrange(N)])
    return L


def embedded_space(X, m, tau=1):
    """Time-delay embedding with m dimensions and tau delays."""
    N = X.shape[0] - int(np.ceil(m))
    Y = np.zeros((N, int(np.ceil(X.shape[1] * m))))
    for i in xrange(N):
        # print X[i:i+m,:].flatten().shape, w, X.shape
        # print Y[i,:].shape
        rem = int((m % 1) * X.shape[1])  # Reminder for float m
        Y[i, :] = np.concatenate((X[i:i + int(m), :].flatten(),
                                 X[i + int(m), :rem]))
    return Y


def read_cqt_features(audio_file, features_dir):
    """Reads the pre-computed cqt (from the dtw project).

    Paramters
    ---------
    audio_file : str
        Path to the original audio file.
    features_dir : str
        Path to the precomputed features folder.

    Returns
    -------
    F : np.array
        Pre-computed CQT log spectrogram.
    indeces : np.array
        Indeces for the frames for each F position (sub-beats).
    """
    features_file = os.path.join(features_dir, os.path.basename(audio_file) +
                                 ".pk")
    with open(features_file, "r") as f:
        file_data = pickle.load(f)

    F = file_data["cqgram"].T
    indeces = file_data["subseg"]
    return F, indeces

def map_indeces(in_idxs, subbeats_idxs, frame_times):
    """Maps the in_index that index the subbeats_idxs into frame_times.

    Paramters
    ---------
    in_idxs: np.array
        The indeces to be mapped.
    subbeats_idxs: np.array
        The subbeats indeces to where in_idxs map to.
    frame_times: np.array
        Times for each of the final frame times that in_idxs will be mapped to.

    Returns
    -------
    out_idxs: np.array
        The new output indeces in the frame_times space.
    """
    in_frames_idxs = subbeats_idxs[in_idxs]
    in_times = np.array([idx * msaf.Anal.hop_size / float(msaf.Anal.sample_rate)
                 for idx in in_frames_idxs])
    out_idxs = np.abs(np.subtract.outer(in_times, frame_times)).argmin(axis=1)
    return np.unique(out_idxs)


def symstack(X, n_steps=5, delay=1, **kwargs):
    '''Symmetric history stacking.

    like librosa.feature.stack_memory, but IN THE FUTURE!!!
    '''
    rpad = n_steps * delay
    Xpad = np.pad(X,
                  [(0, 0), (0, rpad)],
                  **kwargs)

    Xstack = librosa.feature.stack_memory(Xpad,
                                          n_steps=2 * n_steps + 1,
                                          delay=delay,
                                          **kwargs)

    return Xstack[:, rpad:]


def compute_recurrence_plot(F, model):
    """Computes the recurrence plot for the given features using the
    similarity model previously trained.

    Parameters
    ----------
    F : np.array
        Set of features: must be CQT features.
    model : object
        Scikits classifier.

    Returns
    -------
    R : np.array
        The recurrence plot.
    """
    C = F.T
    X = symstack(C, n_steps=5, mode='edge')
    N = C.shape[1]
    R = np.zeros((N, N))
    for i in xrange(N):
        for j in xrange(N):
            Xt = np.abs(X[:, i] - X[:, j])[np.newaxis, :]
            R[i, j] = model.predict(Xt)
    return R


class Segmenter(SegmenterInterface):
    def processFlat(self):
        """Main process.
        Returns
        -------
        est_idxs : np.array(N)
            Estimated times for the segment boundaries in frame indeces.
        est_labels : np.array(N-1)
            Estimated labels for the segments.
        """
        # Structural Features params
        Mp = self.config["Mp_adaptive"]   # Size of the adaptive threshold for
                                          # peak picking
        od = self.config["offset_thres"]  # Offset coefficient for adaptive
                                          # thresholding

        M = self.config["M_gaussian"]     # Size of gaussian kernel in beats
        m = self.config["m_embedded"]     # Number of embedded dimensions
        k = self.config["k_nearest"]      # k*N-nearest neighbors for the
                                          # recurrence plot

        # Preprocess to obtain features, times, and input boundary indeces
        F = self._preprocess()
        if self.config["model"] is not None:
            F_dtw, subbeats_idxs = read_cqt_features(self.audio_file,
                                                    self.config["features_dir"])
            F = F_dtw
            with open(self.config["model"]) as f:
                model = pickle.load(f)["model"]

            R = compute_recurrence_plot(F, model)
            #plt.imshow(R, interpolation="nearest", aspect="auto"); plt.show()

        else:
            # Emedding the feature space (i.e. shingle)
            E = embedded_space(F, m)
            #plt.imshow(E.T, interpolation="nearest", aspect="auto"); plt.show()

            # Recurrence matrix
            R = librosa.segment.recurrence_matrix(
                E.T,
                k=k * int(F.shape[0]),
                width=1,  # zeros from the diagonal
                metric="seuclidean",
                sym=True).astype(np.float32)
            #plt.imshow(R, interpolation="nearest", aspect="auto"); plt.show()

        # Check size in case the track is too short
        if F.shape[0] > 20:

            # Circular shift
            L = circular_shift(R)
            #plt.imshow(L, interpolation="nearest", cmap=plt.get_cmap("binary"))
            #plt.show()

            # Obtain structural features by filtering the lag matrix
            SF = gaussian_filter(L.T, M=M, axis=1)
            SF = gaussian_filter(L.T, M=1, axis=0)
            # plt.imshow(SF.T, interpolation="nearest", aspect="auto")
            #plt.show()

            # Compute the novelty curve
            nc = compute_nc(SF)

            # Find peaks in the novelty curve
            est_bounds = pick_peaks(nc, L=Mp, offset_denom=od)

            # Re-align embedded space
            est_bounds = np.asarray(est_bounds) + int(np.ceil(m / 2.))

            if self.framesync:
                est_bounds /= red
                F = F_copy
        else:
            est_bounds = []

        # Add first and last frames
        est_idxs = np.concatenate(([0], est_bounds, [F.shape[0] - 1]))
        est_idxs = np.unique(est_idxs)

        assert est_idxs[0] == 0 and est_idxs[-1] == F.shape[0] - 1

        # Map times from CQT to current indeces
        if self.config["model"] is not None:
            est_idxs = map_indeces(est_idxs, subbeats_idxs, self.frame_times)

        # Empty labels
        est_labels = np.ones(len(est_idxs) - 1) * - 1

        # Post process estimations
        est_idxs, est_labels = self._postprocess(est_idxs, est_labels)

        # plt.figure(1)
        # plt.plot(nc);
        # [plt.axvline(p, color="m", ymin=.6) for p in est_bounds]
        # [plt.axvline(b, color="b", ymax=.6, ymin=.3) for b in brian_bounds]
        # [plt.axvline(b, color="g", ymax=.3) for b in ann_bounds]
        # plt.show()

        return est_idxs, est_labels
