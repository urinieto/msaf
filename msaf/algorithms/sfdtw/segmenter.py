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
import matplotlib.pyplot as plt
import numpy as np
import librosa
from scipy.spatial import distance
from scipy import signal
from scipy.ndimage import filters
import scipy.misc

import seaborn as sns

sns.set_style("dark")

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


def pick_peaks_new(nc, L=16, ref_bounds=[]):
    """Obtain peaks from a novelty curve using an adaptive threshold."""
    offset = -nc.mean() / 20.

    #nc = filters.gaussian_filter1d(nc, sigma=1)  # Smooth out nc

    #th = filters.median_filter(nc, size=L) + offset
    #th = filters.gaussian_filter(nc, sigma=L/2., mode="nearest") + offset
    th = np.zeros(nc.shape)

    peaks = []
    for i in xrange(1, nc.shape[0] - 1):
        # is it a peak?
        if nc[i - 1] < nc[i] and nc[i] > nc[i + 1]:
            # is it above the threshold?
            if nc[i] > th[i]:
                peaks.append(i)
    #plt.plot(nc)
    #plt.plot(th)
    ##for peak in peaks:
        ##plt.axvline(peak, color="blue")
    #for bound in ref_bounds:
        #plt.axvline(bound, color="green", alpha=0.5, linewidth=3)
    #plt.show()

    return peaks

#def pick_peaks(nc, L=16):
    #"""Obtain peaks from a novelty curve using an adaptive threshold."""
    #offset = -nc.mean() / 20.

    ##nc = filters.gaussian_filter1d(nc, sigma=1)  # Smooth out nc

    #th = filters.median_filter(nc, size=L) + offset
    ##th = filters.gaussian_filter(nc, sigma=L/2., mode="nearest") + offset

    #peaks = []
    #for i in xrange(1, nc.shape[0] - 1):
        ## is it a peak?
        #if nc[i - 1] < nc[i] and nc[i] > nc[i + 1]:
            ## is it above the threshold?
            #if nc[i] > th[i]:
                #peaks.append(i)
    ##plt.plot(nc)
    ##plt.plot(th)
    ##for peak in peaks:
        ##plt.axvline(peak)
    ##plt.show()

    #return peaks

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


def k_nearest(X, k):
    rec = np.zeros(X.shape, dtype=bool)

    # get the k nearest neighbors for each point
    for i in range(X.shape[0]):
        for j in np.argsort(X[i])[-k:]:
            rec[i, j] = True

    return rec


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

        if self.config["beats"]:
            features_dir = self.config["features_dir_beats"]
            recplots_dir = self.config["recplots_dir_beats"]
            model_sufix = "_beat"
        else:
            features_dir = self.config["features_dir_subbeats"]
            recplots_dir = self.config["recplots_dir_subbeats"]
            model_sufix = ""

        if self.config["bias"]:
            bias_sufix = "_bias"
        else:
            bias_sufix = ""


        # Preprocess to obtain features, times, and input boundary indeces
        F = self._preprocess()
        if self.config["model"] is not None:
            if self.config["model_type"] == "iso":
                recplots_dir += "_" + self.config["model_type"]
                self.config["model"] = os.path.join(self.config["model"],
                                                    "similarity_model_isophonics" +
                                                    model_sufix + ".pickle")
            elif self.config["model_type"] == "salami":
                recplots_dir += "_" + self.config["model_type"]
                self.config["model"] = os.path.join(self.config["model"],
                                                    "similarity_model_salami" +
                                                    model_sufix + ".pickle")
            else:
                raise RuntimeError("Wrong model type in config")
            msaf.utils.ensure_dir(recplots_dir)

            F_dtw, subbeats_idxs = msaf.utils.read_cqt_features(
                self.audio_file, features_dir)
            ref_times, ref_labels = msaf.io.read_references(self.audio_file)
            ref_bounds = msaf.utils.times_to_bounds(ref_times, subbeats_idxs)

            recplot_file = msaf.utils.get_recplot_file(recplots_dir,
                                                       self.audio_file,
                                                       bias=bias_sufix)
            R = msaf.utils.get_recurrence_plot(F_dtw, recplot_file, self.config)
            R = k_nearest(R, int(R.shape[0] * k))

        else:
            ref_bounds = msaf.io.read_ref_bound_frames(self.audio_file,
                                                       self.frame_times)

            #F, subbeats_idxs = msaf.utils.read_cqt_features(
                #self.audio_file, features_dir)

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

        # Check size in case the track is too short
        if F.shape[0] > 20:

            from scipy.ndimage import median_filter
            diagonal_median = librosa.segment.timelag_filter(median_filter)
            #R = diagonal_median(R, size=(1, self.config["diag_filter"]), mode='mirror')
            #plt.imshow(R, interpolation="nearest", aspect="auto"); plt.show()

            # Circular shift
            L = circular_shift(R)
            #plt.figure(figsize=(7, 15))
            #SSM = compute_ssm(F)
            #np.fill_diagonal(R, 0.5)
            #plt.subplot(2,1,1)
            #plt.imshow(R, interpolation="nearest")
            #plt.subplot(2,1,2)
            #plt.imshow(SSM, interpolation="nearest")
            #plt.show()
            #for bound in ref_bounds:
                #plt.axvline(bound)
                #plt.axhline(bound)
            #plt.imshow(L, interpolation="nearest", cmap=plt.get_cmap("binary"))
            #plt.show()

            # Obtain structural features by filtering the lag matrix
            SF = gaussian_filter(L.T, M=M, axis=1)
            SF = gaussian_filter(L.T, M=1, axis=0)
            #plt.imshow(SF.T, interpolation="nearest", aspect="auto")
            #plt.show()

            # Compute the novelty curve
            nc = compute_nc(SF)

            # Find peaks in the novelty curve
            est_bounds = pick_peaks(nc, L=Mp, offset_denom=od)
            #est_bounds = pick_peaks(nc, L=Mp)
            #est_bounds = pick_peaks_new(nc, L=Mp, ref_bounds=ref_bounds)

            # Re-align embedded space
            est_bounds = np.asarray(est_bounds) + int(np.ceil(m / 2.))

        else:
            est_bounds = []

        # Different audio files? Hack...
        if len(est_bounds) > 0 and est_bounds[-1] >= F.shape[0]:
            est_bounds = est_bounds[:-1]

        # Add first and last frames
        est_idxs = np.concatenate(([0], est_bounds, [len(self.frame_times)]))
        est_idxs = np.unique(est_idxs)

        assert est_idxs[0] == 0 and est_idxs[-1] == len(self.frame_times)

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
