#!/usr/bin/env python
# coding: utf-8
"""
This script identifies the boundaries of a given track using a novel C-NMF
method.
"""

__author__ = "Oriol Nieto"
__copyright__ = "Copyright 2014, Music and Audio Research Lab (MARL)"
__license__ = "GPL"
__version__ = "1.0"
__email__ = "oriol@nyu.edu"

import argparse
import logging
import numpy as np
import time
from scipy.spatial import distance
from scipy.ndimage import filters
import sys
import pylab as plt

sys.path.append("../../")
import msaf_io as MSAF
#import eval as EV
import utils as U
try:
    import pymf
except:
    logging.error("PyMF module not found, C-NMF won't work")
    sys.exit()


def median_filter(X, M=8):
    """Median filter along the first axis of the feature matrix X."""
    for i in xrange(X.shape[1]):
        X[:, i] = filters.median_filter(X[:, i], size=M)
    return X


def mean_filter(x, M=9):
    """Average filter."""
    #window = np.ones(M) / float(M)
    window = np.hanning(M)
    return np.convolve(x, window, 'same')


def compute_ssm(X, metric="seuclidean"):
    """Computes the self-similarity matrix of X."""
    D = distance.pdist(X, metric=metric)
    D = distance.squareform(D)
    D /= D.max()
    #D = D ** 2
    D /= D.max()
    return 1 - D


def most_frequent(x):
    """Returns the most frequent value in x."""
    return np.argmax(np.bincount(x))


def compute_labels(label_frames, bound_idxs):
    """Computes the labels using the bounds."""

    labels = [label_frames[0]]
    bound_inters = zip(bound_idxs[:-1], bound_idxs[1:])
    for bound_inter in bound_inters:
        if bound_inter[1] - bound_inter[0] <= 0:
            labels.append(np.max(label_frames) + 1)
        else:
            labels.append(most_frequent(
                label_frames[bound_inter[0]: bound_inter[1]]))
        #print bound_inter, labels[-1]
    labels.append(label_frames[-1])

    return labels


def cnmf(S, rank=2, niter=500):
    """(Convex) Non-Negative Matrix Factorization.

    Parameters
    ----------
    S: np.array
        Self-similarity matrix (must be symmetric)
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
    W: np.array
        Convex weights matrix
        (s.t. S ~= S * W * G = F * G)
    """
    nmf_mdl = pymf.CNMF(S, num_bases=rank)
    nmf_mdl.factorize(niter=niter)
    F = np.asarray(nmf_mdl.W)
    G = np.asarray(nmf_mdl.H)
    W = np.asarray(nmf_mdl.G)
    return F, G, W


def get_boundaries(S, rank, R=9, niter=500, bound_idxs=None):
    """
    Gets the boundaries from the factorization matrices.

    Parameters
    ----------
    S: np.array()
        Self-similarity matrix.
    rank: int
        Rank of decomposition.
    R : int
        Size of the median filter for activation matrix
    niter: int
        Number of iterations for k-means.
    bound_idxs : list
        Use previously found boundaries (None to detect them).

    Returns
    -------
    bounds_idx: np.array
        Bound indeces found.
    labels: list
        Estimated labels.
    """
    # Find non filtered boundaries
    while True:
        try:
            F, G, W = cnmf(S, rank=rank, niter=niter)
        except:
            return bound_idxs, [0]

        W = G.T

        # Filter W
        idx = np.argmax(W, axis=1)
        max_idx = np.arange(W.shape[0])
        max_idx = (max_idx, idx.flatten())
        W[:, :] = 0
        W[max_idx] = idx + 1

        #plt.imshow(W, interpolation="nearest", aspect="auto")
        #plt.show()

        # TODO: Order matters?
        W = np.sum(W, axis=1)
        W = median_filter(W[:, np.newaxis], R)

        #plt.imshow(W, interpolation="nearest", aspect="auto")
        #plt.show()

        if bound_idxs is None:
            bound_idxs = np.where(np.diff(W.flatten()) != 0)[0] + 1

        # Increase rank if we found too few boundaries
        if len(np.unique(bound_idxs)) <= 2:
            rank += 1
            bound_idxs = None
        else:
            break

    labels = compute_labels(W.flatten().astype(int), bound_idxs)

    #print labels
    #plt.imshow(G, interpolation="nearest", aspect="auto", cmap="hot")
    #for b in bound_idxs:
        #plt.axvline(b, linewidth=2.0, color="b")
    #plt.show()

    return bound_idxs, labels


def process(in_path, feature="hpcp", annot_beats=False, annot_bounds=False,
            h=16, R=9, rank=4):
    """Main process. 
    
    Parameters
    ----------
    in_path : str
        Path to the dataset folder.
    feature : str
        Type of feature (hpcp or mfcc).
    annot_beats : bool
        Whether to use annotated beats or not.
    annot_bounds : bool
        Whether to use annotated bounds or not.
    h : int
        Size of the median filter for the SSM.
    R : int
        Size of the median filter for the decomposed C-NMF matrix.
    rank : int
        Rank of decomposition
    """

    # C-NMF params
    niter = 300     # Iterations for the matrix factorization and clustering
    dist = "correlation"

    # Read features
    chroma, mfcc, beats, dur = MSAF.get_features(in_path,
                                                 annot_beats=annot_beats)

    # Read annotated bounds if necessary
    bound_idxs = None
    if annot_bounds:
        try:
            bound_idxs = MSAF.read_annot_bound_frames(in_path, beats)[1:-1]
        except:
            logging.warning("No annotated boundaries in file %s" % in_path)

    # Use specific feature
    if feature == "hpcp":
        F = U.lognormalize_chroma(chroma)  # Normalize chromas
    elif "mfcc":
        F = mfcc
    else:
        logging.error("Feature type not recognized: %s" % feature)

    if F.shape[0] >= h:
        # Median filter
        F = median_filter(F, M=h)
        #plt.imshow(F.T, interpolation="nearest", aspect="auto"); plt.show()

        # Self similarity matrix
        S = compute_ssm(F, metric=dist)
        #plt.imshow(S, interpolation="nearest", aspect="auto"); plt.show()

        # Find the boundary indices using matrix factorization
        bound_idxs, est_labels = get_boundaries(S, rank, niter=niter, R=R,
                                                bound_idxs=bound_idxs)
    else:
        # The track is too short. We will only output the first and last
        # time stamps
        bound_idxs = np.empty(0)
        est_labels = [0]

    # Concatenate first boundary
    bound_idxs = np.concatenate(([0], bound_idxs)).astype(int)

    # Read annotated bounds for comparison purposes
    #ann_bounds = MSAF.read_annot_bound_frames(in_path, beats)
    #logging.info("Annotated bounds: %s" % ann_bounds)
    #logging.info("Estimated bounds: %s" % bound_idxs)

    # Get times
    est_times = beats[bound_idxs]

    # Concatenate last boundary
    est_times = np.concatenate((est_times, [dur]))

    # Concatenate last boundary
    logging.info("Estimated times: %s" % est_times)
    logging.info("Estimated labels: %s" % est_labels)

    assert len(est_times) - 1 == len(est_labels)

    return est_times, est_labels


def main():
    """Main function to parse the arguments and call the main process."""
    parser = argparse.ArgumentParser(description=
        "Segments the given audio file using the new version of the C-NMF "
                                     "method.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("in_path",
                        action="store",
                        help="Input path to the audio file")
    parser.add_argument("-f",
                        action="store",
                        dest="feature",
                        help="Feature to use (mfcc or hpcp)",
                        default="hpcp")
    parser.add_argument("-b",
                        action="store_true",
                        dest="annot_beats",
                        help="Use annotated beats",
                        default=False)
    parser.add_argument("-bo",
                        action="store_true",
                        dest="annot_bounds",
                        help="Use annotated bounds",
                        default=False)
    args = parser.parse_args()
    start_time = time.time()

    # Setup the logger
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s',
        level=logging.INFO)

    # Run the algorithm
    process(args.in_path, feature=args.feature, annot_beats=args.annot_beats,
            annot_bounds=args.annot_bounds)

    # Done!
    logging.info("Done! Took %.2f seconds." % (time.time() - start_time))

if __name__ == '__main__':
    main()
