#!/usr/bin/env python
# coding: utf-8
import logging
import numpy as np
import scipy.cluster.vq as vq
from sklearn import mixture
from sklearn.cluster import KMeans

# Local stuff
from . import utils_2dfmc as utils2d
from .xmeans import XMeans

import msaf.utils as U
from msaf.algorithms.interface import SegmenterInterface


def get_feat_segments(F, bound_idxs):
    """Returns a set of segments defined by the bound_idxs.

    Parameters
    ----------
    F: np.ndarray
        Matrix containing the features, one feature vector per row.
    bound_idxs: np.ndarray
        Array with boundary indeces.

    Returns
    -------
    feat_segments: list
        List of segments, one for each boundary interval.
    """
    # Make sure bound_idxs are not empty
    assert len(bound_idxs) > 0, "Boundaries can't be empty"

    # Make sure that boundaries are sorted
    bound_idxs = np.sort(bound_idxs)

    # Make sure we're not out of bounds
    assert bound_idxs[0] >= 0 and bound_idxs[-1] < F.shape[0], \
        "Boundaries are not correct for the given feature dimensions."

    # Obtain the segments
    feat_segments = []
    for i in range(len(bound_idxs) - 1):
        feat_segments.append(F[bound_idxs[i]:bound_idxs[i + 1], :])
    return feat_segments


def feat_segments_to_2dfmc_max(feat_segments, offset=4):
    """From a list of feature segments, return a list of 2D-Fourier Magnitude
    Coefs using the maximum segment size as main size and zero pad the rest.

    Parameters
    ----------
    feat_segments: list
        List of segments, one for each boundary interval.
    offset: int >= 0
        Number of frames to ignore from beginning and end of each segment.

    Returns
    -------
    fmcs: np.ndarray
        Tensor containing the 2D-FMC matrices, one matrix per segment.
    """
    if len(feat_segments) == 0:
        return []

    # Get maximum segment size
    max_len = max([feat_segment.shape[0] for feat_segment in feat_segments])

    fmcs = []
    for feat_segment in feat_segments:
        # Zero pad if needed
        X = np.zeros((max_len, feat_segment.shape[1]))

        # Remove a set of frames in the beginning an end of the segment
        if feat_segment.shape[0] <= offset or offset == 0:
            X[:feat_segment.shape[0], :] = feat_segment
        else:
            X[:feat_segment.shape[0] - offset, :] = \
                feat_segment[offset // 2:-offset // 2, :]

        # Compute the 2D-FMC
        try:
            fmcs.append(utils2d.compute_ffmc2d(X))
        except:
            logging.warning("Couldn't compute the 2D Fourier Transform")
            fmcs.append(np.zeros((X.shape[0] * X.shape[1]) // 2 + 1))

        # Normalize
        # fmcs[-1] = fmcs[-1] / float(fmcs[-1].max())

    return np.asarray(fmcs)


def compute_labels_kmeans(fmcs, k):
    # Removing the higher frequencies seem to yield better results
    fmcs = fmcs[:, fmcs.shape[1] // 2:]

    # Pre-process
    fmcs = np.log1p(fmcs)
    wfmcs = vq.whiten(fmcs)

    # Make sure we are not using more clusters than existing segments
    if k > fmcs.shape[0]:
        k = fmcs.shape[0]

    # K-means
    kmeans = KMeans(n_clusters=k, n_init=100)
    kmeans.fit(wfmcs)

    return kmeans.labels_


def compute_similarity(F, bound_idxs, dirichlet=False, xmeans=False, k=5,
                       offset=4):
    """Main function to compute the segment similarity of file file_struct.

    Parameters
    ----------
    F: np.ndarray
        Matrix containing one feature vector per row.
    bound_idxs: np.ndarray
        Array with the indeces of the segment boundaries.
    dirichlet: boolean
        Whether to use the dirichlet estimator of the number of unique labels.
    xmeans: boolean
        Whether to use the xmeans estimator of the number of unique labels.
    k: int > 0
        If the other two predictors are `False`, use fixed number of labels.
    offset: int >= 0
        Number of frames to ignore from beginning and end of each segment.

    Returns
    -------
    labels_est: np.ndarray
        Estimated labels, containing integer identifiers.
    """
    # Get the feature segments
    feat_segments = get_feat_segments(F, bound_idxs)

    # Get the 2D-FMCs segments
    fmcs = feat_segments_to_2dfmc_max(feat_segments, offset)
    if len(fmcs) == 0:
        return np.arange(len(bound_idxs) - 1)

    # Compute the labels using kmeans
    if dirichlet:
        k_init = np.min([fmcs.shape[0], k])
        # Only compute the dirichlet method if the fmc shape is small enough
        if fmcs.shape[1] > 500:
            labels_est = compute_labels_kmeans(fmcs, k=k)
        else:
            dpgmm = mixture.DPGMM(n_components=k_init, covariance_type='full')
            # dpgmm = mixture.VBGMM(n_components=k_init, covariance_type='full')
            dpgmm.fit(fmcs)
            k = len(dpgmm.means_)
            labels_est = dpgmm.predict(fmcs)
            # print("Estimated with Dirichlet Process:", k)
    if xmeans:
        xm = XMeans(fmcs, plot=False)
        k = xm.estimate_K_knee(th=0.01, maxK=8)
        labels_est = compute_labels_kmeans(fmcs, k=k)
        # print("Estimated with Xmeans:", k)
    else:
        labels_est = compute_labels_kmeans(fmcs, k=k)

    return labels_est


class Segmenter(SegmenterInterface):
    """
    This method labels segments using the 2D-FMC method described here:

    Nieto, O., Bello, J.P., Music Segment Similarity Using 2D-Fourier Magnitude
    Coefficients. Proc. of the 39th IEEE International Conference on Acoustics,
    Speech, and Signal Processing (ICASSP). Florence, Italy, 2014 (`PDF`_).

    .. _PDF: http://marl.smusic.nyu.edu/nieto/publications/NietoBello-ICASSP14.pdf
    """
    def processFlat(self):
        """Main process.
        Returns
        -------
        est_idx : np.array(N)
            Estimated indeces for the segment boundaries in frames.
        est_labels : np.array(N-1)
            Estimated labels for the segments.
        """
        # Preprocess to obtain features, times, and input boundary indeces
        F = self._preprocess()

        # Normalize
        F = U.normalize(F, norm_type=self.config["label_norm_feats"],
                        floor=self.config["label_norm_floor"],
                        min_db=self.config["label_norm_min_db"])

        # Find the labels using 2D-FMCs
        est_labels = compute_similarity(F, self.in_bound_idxs,
                                        dirichlet=self.config["dirichlet"],
                                        xmeans=self.config["xmeans"],
                                        k=self.config["k"],
                                        offset=self.config["2dfmc_offset"])

        # Post process estimations
        self.in_bound_idxs, est_labels = self._postprocess(self.in_bound_idxs,
                                                           est_labels)

        return self.in_bound_idxs, est_labels
