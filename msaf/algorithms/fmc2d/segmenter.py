#!/usr/bin/env python
# coding: utf-8
import logging
import numpy as np
import scipy.cluster.vq as vq
from sklearn import mixture

# Local stuff
from . import utils_2dfmc as utils2d
from .xmeans import XMeans

from msaf.algorithms.interface import SegmenterInterface

MIN_LEN = 4


def get_pcp_segments(PCP, bound_idxs):
    """Returns a set of segments defined by the bound_idxs."""
    pcp_segments = []
    for i in range(len(bound_idxs) - 1):
        pcp_segments.append(PCP[bound_idxs[i]:bound_idxs[i + 1], :])
    return pcp_segments


def pcp_segments_to_2dfmc_max(pcp_segments):
    """From a list of PCP segments, return a list of 2D-Fourier Magnitude
        Coefs using the maximumg segment size and zero pad the rest."""
    if len(pcp_segments) == 0:
        return []

    # Get maximum segment size
    max_len = max([pcp_segment.shape[0] for pcp_segment in pcp_segments])

    OFFSET = 4
    fmcs = []
    for pcp_segment in pcp_segments:
        # Zero pad if needed
        X = np.zeros((max_len, pcp_segment.shape[1]))
        #X[:pcp_segment.shape[0],:] = pcp_segment
        if pcp_segment.shape[0] <= OFFSET:
            X[:pcp_segment.shape[0], :] = pcp_segment
        else:
            X[:pcp_segment.shape[0]-OFFSET, :] = \
                pcp_segment[OFFSET // 2:-OFFSET // 2, :]

        # 2D-FMC
        try:
            fmcs.append(utils2d.compute_ffmc2d(X))
        except:
            logging.warning("Couldn't compute the 2D Fourier Transform")
            fmcs.append(np.zeros((X.shape[0] * X.shape[1]) // 2 + 1))

        # Normalize
        #fmcs[-1] = fmcs[-1] / float(fmcs[-1].max())

    return np.asarray(fmcs)


def compute_labels_kmeans(fmcs, k=6):
    # Removing the higher frequencies seem to yield better results
    fmcs = fmcs[:, fmcs.shape[1] // 2:]

    fmcs = np.log1p(fmcs)
    wfmcs = vq.whiten(fmcs)

    dic, dist = vq.kmeans(wfmcs, k, iter=100)
    labels, dist = vq.vq(wfmcs, dic)

    return labels


def compute_similarity(PCP, bound_idxs, dirichlet=False, xmeans=False, k=5):
    """Main function to compute the segment similarity of file file_struct."""

    # Get PCP segments
    pcp_segments = get_pcp_segments(PCP, bound_idxs)

    # Get the 2d-FMCs segments
    fmcs = pcp_segments_to_2dfmc_max(pcp_segments)
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
            #dpgmm = mixture.VBGMM(n_components=k_init, covariance_type='full')
            dpgmm.fit(fmcs)
            k = len(dpgmm.means_)
            labels_est = dpgmm.predict(fmcs)
            #print "Estimated with Dirichlet Process:", k
    if xmeans:
        xm = XMeans(fmcs, plot=False)
        k = xm.estimate_K_knee(th=0.01, maxK=8)
        labels_est = compute_labels_kmeans(fmcs, k=k)
        #print "Estimated with Xmeans:", k
    else:
        labels_est = compute_labels_kmeans(fmcs, k=k)

    # Plot results
    #plot_pcp_wgt(PCP, bound_idxs)

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
        #F = self._preprocess(valid_features=["pcp", "cqt"])
        F = self._preprocess()

        # Find the labels using 2D-FMCs
        est_labels = compute_similarity(F, self.in_bound_idxs,
                                        dirichlet=self.config["dirichlet"],
                                        xmeans=self.config["xmeans"],
                                        k=self.config["k"])

        # Post process estimations
        self.in_bound_idxs, est_labels = self._postprocess(self.in_bound_idxs,
                                                           est_labels)

        return self.in_bound_idxs, est_labels
