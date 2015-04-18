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
import numba
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


@numba.jit
def compute_recurrence_plot(F, model, w=5):
    """Computes the recurrence plot for the given features using the
    similarity model previously trained.

    Parameters
    ----------
    F : np.array
        Set of features: must be CQT features.
    model : object
        Scikits classifier.
    w : int
        Number of frames in front / back.

    Returns
    -------
    R : list
        R_predict : np.array
            The recurrence plot using binary prediction.
        R_proba : np.array
            The recurrence plot using predicted probabilities.
        R_mask : np.array
            The recurrence plot using the binary predictions as a mask on the
            predicted probabilities.
    """
    C = F.T
    X = symstack(C, n_steps=w, mode='edge')
    N = X.shape[1]
    R_predict = np.eye(N)
    R_proba = np.eye(N)
    for i in range(N):
        for j in range(i + 1, N):
            Xt = np.abs(X[:, i] - X[:, j])[np.newaxis, :]
            R_predict[i, j] = model.predict(Xt)
            R_predict[j, i] = R_predict[i, j]
            R_proba[i, j] = model.predict_proba(Xt)[0][1]
            R_proba[j, i] = R_proba[i, j]

    # Recurrence plot, mask type
    R_mask = R_proba * R_predict

    return [R_predict, R_proba, R_mask]


def get_recplot_file(recplots_dir, audio_file):
    """Gets the recurrence plot file.

    Parameters
    ----------
    recplots_dir : str
        Directory where to store the recurrence plots.
    audio_file : str
        Path to the audio file.

    Returns
    -------
    recplot_path : str
        Path to the recplot pk file.
    """
    return os.path.join(recplots_dir, os.path.basename(audio_file) + ".pk")


def times_to_bounds(in_times, subbeats_idxs):
    """Converts the times to bounds of the given subbeats.

    Paramters
    ---------
    in_times: np.array
        The times to be converted.
    subbeats_idxs: np.array
        The subbeats indeces to where in_times map to.

    Returns
    -------
    out_idxs: np.array
        The new output indeces in the subbeats_idxs space.
    """
    frame_times = np.array(
        [idx * msaf.Anal.hop_size / float(msaf.Anal.sample_rate)
         for idx in subbeats_idxs])
    out_idxs = np.abs(np.subtract.outer(in_times, frame_times)).argmin(axis=1)
    return np.unique(out_idxs)


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
        else:
            features_dir = self.config["features_dir_subbeats"]
            recplots_dir = self.config["recplots_dir_subbeats"]
        # Preprocess to obtain features, times, and input boundary indeces
        F = self._preprocess()
        if self.config["model"] is not None:
            F_dtw, subbeats_idxs = read_cqt_features(
                self.audio_file, features_dir)
            ref_times, ref_labels = msaf.io.read_references(self.audio_file)
            ref_bounds = times_to_bounds(ref_times, subbeats_idxs)

            recplot_file = get_recplot_file(recplots_dir,
                                            self.audio_file)
            if os.path.isfile(recplot_file):
                with open(recplot_file) as f:
                    R = pickle.load(f)[self.config["recplot_type"]]
            else:
                with open(self.config["model"]) as f:
                    model = pickle.load(f)["model"]

                R = compute_recurrence_plot(F_dtw, model, self.config["w"])

                R_dict = {}
                R_dict["predict"] = R[0]
                R_dict["proba"] = R[1]
                R_dict["mask"] = R[2]
                with open(recplot_file, "w") as f:
                    pickle.dump(R_dict, f)
                R = R_dict[self.config["recplot_type"]]
                #R = scipy.misc.imresize(R, (len(self.frame_times),
                                            #len(self.frame_times)))
        else:
            ref_bounds = msaf.io.read_ref_bound_frames(self.audio_file,
                                                       self.frame_times)
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

            # Circular shift
            L = circular_shift(R)
            SSM = compute_ssm(F)
            np.fill_diagonal(R, 0.5)
            plt.figure(figsize=(7, 15))
            plt.subplot(2,1,1)
            plt.imshow(R, interpolation="nearest")
            plt.subplot(2,1,2)
            plt.imshow(SSM, interpolation="nearest")
            plt.show()
            #for bound in ref_bounds:
                #plt.axvline(bound)
                #plt.axhline(bound)
            #plt.imshow(L, interpolation="nearest", cmap=plt.get_cmap("binary"))
            #plt.show()

            # Obtain structural features by filtering the lag matrix
            SF = gaussian_filter(L.T, M=M, axis=1)
            SF = gaussian_filter(L.T, M=2, axis=0)
            #plt.imshow(SF.T, interpolation="nearest", aspect="auto")
            #plt.show()

            # Compute the novelty curve
            nc = compute_nc(SF)

            # Find peaks in the novelty curve
            #est_bounds = pick_peaks(nc, L=Mp, offset_denom=od)
            #est_bounds = pick_peaks(nc, L=Mp)
            est_bounds = pick_peaks_new(nc, L=Mp, ref_bounds=ref_bounds)

            # Re-align embedded space
            est_bounds = np.asarray(est_bounds) + int(np.ceil(m / 2.))

        else:
            est_bounds = []

        # Different audio files? Hack...
        if est_bounds[-1] >= F.shape[0]:
            est_bounds = est_bounds[:-1]

        # Add first and last frames
        est_idxs = np.concatenate(([0], est_bounds, [F.shape[0] - 1]))
        est_idxs = np.unique(est_idxs)

        assert est_idxs[0] == 0 and est_idxs[-1] == F.shape[0] - 1

        # Map times from CQT to current indeces
        #if self.config["model"] is not None:
            #est_idxs = map_indeces(est_idxs, subbeats_idxs, self.frame_times)

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
