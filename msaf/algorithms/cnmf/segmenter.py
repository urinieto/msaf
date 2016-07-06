# coding: utf-8
import logging
import numpy as np
import pylab as plt
from scipy.ndimage import filters

from msaf.algorithms.interface import SegmenterInterface
from msaf import pymf


def median_filter(X, M=8):
    """Median filter along the first axis of the feature matrix X."""
    for i in range(X.shape[1]):
        X[:, i] = filters.median_filter(X[:, i], size=M)
    return X


def cnmf(S, rank, niter=500, hull=False):
    """(Convex) Non-Negative Matrix Factorization.

    Parameters
    ----------
    S: np.array(p, N)
        Features matrix. p row features and N column observations.
    rank: int
        Rank of decomposition
    niter: int
        Number of iterations to be used

    Returns
    -------
    F: np.array
        Cluster matrix (decomposed matrix)
    G: np.array
        Activation matrix (decomposed matrix)
        (s.t. S ~= F * G)
    """
    if hull:
        nmf_mdl = pymf.CHNMF(S, num_bases=rank)
    else:
        nmf_mdl = pymf.CNMF(S, num_bases=rank)
    nmf_mdl.factorize(niter=niter)
    F = np.asarray(nmf_mdl.W)
    G = np.asarray(nmf_mdl.H)
    return F, G


def most_frequent(x):
    """Returns the most frequent value in x."""
    return np.argmax(np.bincount(x))


def compute_labels(X, rank, R, bound_idxs, niter=300):
    """Computes the labels using the bounds."""

    try:
        F, G = cnmf(X, rank, niter=niter, hull=False)
    except:
        return [1]

    label_frames = filter_activation_matrix(G.T, R)
    label_frames = np.asarray(label_frames, dtype=int)

    #labels = [label_frames[0]]
    labels = []
    bound_inters = zip(bound_idxs[:-1], bound_idxs[1:])
    for bound_inter in bound_inters:
        if bound_inter[1] - bound_inter[0] <= 0:
            labels.append(np.max(label_frames) + 1)
        else:
            labels.append(most_frequent(
                label_frames[bound_inter[0]: bound_inter[1]]))
        #print bound_inter, labels[-1]
    #labels.append(label_frames[-1])

    return labels


def filter_activation_matrix(G, R):
    """Filters the activation matrix G, and returns a flattened copy."""

    #import pylab as plt
    #plt.imshow(G, interpolation="nearest", aspect="auto")
    #plt.show()

    idx = np.argmax(G, axis=1)
    max_idx = np.arange(G.shape[0])
    max_idx = (max_idx, idx.flatten())
    G[:, :] = 0
    G[max_idx] = idx + 1

    # TODO: Order matters?
    G = np.sum(G, axis=1)
    G = median_filter(G[:, np.newaxis], R)

    return G.flatten()


def get_segmentation(X, rank, R, rank_labels, R_labels, niter=300,
                     bound_idxs=None, in_labels=None):
    """
    Gets the segmentation (boundaries and labels) from the factorization
    matrices.

    Parameters
    ----------
    X: np.array()
        Features matrix (e.g. chromagram)
    rank: int
        Rank of decomposition
    R: int
        Size of the median filter for activation matrix
    niter: int
        Number of iterations for k-means
    bound_idxs : list
        Use previously found boundaries (None to detect them)
    in_labels : np.array()
        List of input labels (None to compute them)

    Returns
    -------
    bounds_idx: np.array
        Bound indeces found
    labels: np.array
        Indeces of the labels representing the similarity between segments.
    """

    #import pylab as plt
    #plt.imshow(X, interpolation="nearest", aspect="auto")
    #plt.show()

    # Find non filtered boundaries
    compute_bounds = True if bound_idxs is None else False
    while True:
        if bound_idxs is None:
            try:
                F, G = cnmf(X, rank, niter=niter, hull=False)
            except:
                return np.empty(0), [1]

            # Filter G
            G = filter_activation_matrix(G.T, R)
            if bound_idxs is None:
                bound_idxs = np.where(np.diff(G) != 0)[0] + 1

        # Increase rank if we found too few boundaries
        if compute_bounds and len(np.unique(bound_idxs)) <= 2:
            rank += 1
            bound_idxs = None
        else:
            break

    # Add first and last boundary
    bound_idxs = np.concatenate(([0], bound_idxs, [X.shape[1] - 1]))
    bound_idxs = np.asarray(bound_idxs, dtype=int)
    if in_labels is None:
        labels = compute_labels(X, rank_labels, R_labels, bound_idxs,
                                niter=niter)
    else:
        labels = np.ones(len(bound_idxs) - 1)

    #plt.imshow(G[:, np.newaxis], interpolation="nearest", aspect="auto")
    #for b in bound_idxs:
        #plt.axvline(b, linewidth=2.0, color="k")
    #plt.show()

    return bound_idxs, labels


class Segmenter(SegmenterInterface):
    """
    This script identifies the structure of a given track using a modified version
    of the C-NMF method described here:

    Nieto, O., Jehan, T., Convex Non-negative Matrix Factorization For Automatic
    Music Structure Identification. Proc. of the 38th IEEE International
    Conference on Acoustics, Speech, and Signal Processing (ICASSP).
    Vancouver, Canada, 2013 (`PDF`_).

    The modified version can be found in Oriol Nieto's `PhD Thesis`_.

    .. _PDF: http://marl.smusic.nyu.edu/nieto/publications/Nieto-ICASSP13.pdf
    .. _PhD Thesis: http://marl.smusic.nyu.edu/nieto/publications/Nieto-Dissertation.pdf
    """
    def processFlat(self):
        """Main process.
        Returns
        -------
        est_idxs : np.array(N)
            Estimated indeces for the segment boundaries in frames.
        est_labels : np.array(N-1)
            Estimated labels for the segments.
        """
        # C-NMF params
        niter = 500     # Iterations for the matrix factorization and clustering

        # Preprocess to obtain features, times, and input boundary indeces
        F = self._preprocess()

        if F.shape[0] >= self.config["h"]:
            # Median filter
            F = median_filter(F, M=self.config["h"])
            #plt.imshow(F.T, interpolation="nearest", aspect="auto"); plt.show()

            # Find the boundary indices and labels using matrix factorization
            est_idxs, est_labels = get_segmentation(
                F.T, self.config["rank"], self.config["R"],
                self.config["rank_labels"], self.config["R_labels"],
                niter=niter, bound_idxs=self.in_bound_idxs, in_labels=None)

            est_idxs = np.unique(np.asarray(est_idxs, dtype=int))
        else:
            # The track is too short. We will only output the first and last
            # time stamps
            if self.in_bound_idxs is None:
                est_idxs = np.array([0, F.shape[0]-1])
                est_labels = [1]
            else:
                est_idxs = self.in_bound_idxs
                est_labels = [1] * (len(est_idxs) + 1)

        # Make sure that the first and last boundaries are included
        assert est_idxs[0] == 0  and est_idxs[-1] == F.shape[0] - 1

        # Post process estimations
        est_idxs, est_labels = self._postprocess(est_idxs, est_labels)

        return est_idxs, est_labels
