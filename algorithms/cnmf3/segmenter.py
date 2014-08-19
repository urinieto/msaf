#!/usr/bin/env python
# coding: utf-8
"""
This script identifies the boundaries of a given track using a novel C-NMF
method (v3).
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
from scipy.ndimage import filters
import sys

from msaf import io
from msaf import utils as U

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


def filter_activation_matrix(G, R):
    """Filters the activation matrix G, and returns a flattened copy."""

    #originalG = np.copy(G)
    idx = np.argmax(G, axis=1)
    max_idx = np.arange(G.shape[0])
    max_idx = (max_idx, idx.flatten())
    G[:, :] = 0
    G[max_idx] = idx + 1

    # TODO: Order matters?
    #oG = np.copy(G)
    G = np.sum(G, axis=1)
    G = median_filter(G[:, np.newaxis], R)
    #plt.subplot(1, 2, 1)
    #plt.imshow(originalG, interpolation="nearest", aspect="auto")
    #plt.subplot(1, 2, 2)
    #plt.imshow(F, interpolation="nearest", aspect="auto"); plt.show()

    return G.flatten()


def get_segmentation(X, rank, R, rank_labels, R_labels, niter=300,
                     bound_idxs=None):
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

    Returns
    -------
    bounds_idx: np.array
        Bound indeces found
    labels: np.array
        Indeces of the labels representing the similarity between segments.
    """

    # Find non filtered boundaries
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
        if len(np.unique(bound_idxs)) <= 2:
            rank += 1
            bound_idxs = None
        else:
            break

    # Add first label
    bound_idxs = np.concatenate(([0], bound_idxs, [X.shape[1]-1]))
    bound_idxs = np.asarray(bound_idxs, dtype=int)
    labels = compute_labels(X, rank_labels, R_labels, bound_idxs)

    #plt.imshow(G.T, interpolation="nearest", aspect="auto")
    #for b in bound_idxs:
        #plt.axvline(b, linewidth=2.0, color="k")
    #plt.show()

    return bound_idxs, labels


def process(in_path, feature="hpcp", annot_beats=False, boundaries_id=None,
            framesync=False, h=10, R=9, rank=3, R_labels=6, rank_labels=5):
    """Main process.

    Parameters
    ----------
    in_path : str
        Path to audio file
    feature : str
        Identifier of the features to use
    annot_beats : boolean
        Whether to use annotated beats or not
    boundaries_id : str
        Algorithm id for the boundaries algorithm to use
        (None for C-NMF, and "gt" for ground truth)
    framesync : bool
        Whether to use framesync features
    h : int
        Size of median filter
    R : int
        Size of the median filter for activation matrix
    rank : int
        Rank of decomposition
    R_labels : int
        Size of the median filter for activation matrix for the labels
    rank_labels : int
        Rank of decomposition for the labels
    """
    # C-NMF params
    niter = 300     # Iterations for the matrix factorization and clustering

    # Read features
    hpcp, mfcc, tonnetz, beats, dur, anal = io.get_features(
        in_path, annot_beats=annot_beats, framesync=framesync)

    # Read annotated bounds if necessary
    bound_idxs = None
    if boundaries_id == "gt":
        try:
            bound_idxs = io.read_ref_bound_frames(in_path, beats)[1:-1]
        except:
            logging.warning("No annotated boundaries in file %s" % in_path)

    # Use specific feature
    if feature == "hpcp":
        F = U.lognormalize_chroma(hpcp)  # Normalize chromas
    elif "mfcc":
        F = mfcc
    elif "tonnetz":
        F = U.lognormalize_chroma(tonnetz)  # Normalize tonnetz
        F = U.chroma_to_tonnetz(F)
    else:
        logging.error("Feature type not recognized: %s" % feature)

    if F.shape[0] >= h:
        # Median filter
        F = median_filter(F, M=h)
        #plt.imshow(F.T, interpolation="nearest", aspect="auto"); plt.show()

        # Find the boundary indices and labels using matrix factorization
        bound_idxs, est_labels = get_segmentation(
            F.T, rank, R, rank_labels, R_labels, niter=niter,
            bound_idxs=bound_idxs)
    else:
        # The track is too short. We will only output the first and last
        # time stamps
        bound_idxs = np.empty(0)
        est_labels = [1]

    # Use correct frames to find times
    frames_to_times = beats
    if framesync:
        frames_to_times = U.get_time_frames(dur, anal)

    # Add first and last boundaries
    est_times = np.concatenate(([0], frames_to_times[bound_idxs], [dur]))

    # Remove empty segments if needed
    est_times, est_labels = U.remove_empty_segments(est_times, est_labels)

    logging.info("Estimated times: %s" % est_times)
    logging.info("Estimated labels: %s" % est_labels)

    assert len(est_times) - 1 == len(est_labels), "Number of boundaries (%d) " \
        "and number of labels(%d) don't match" % (len(est_times),
                                                  len(est_labels))

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
    parser.add_argument("-fs",
                        action="store_true",
                        dest="framesync",
                        help="Use frame-synchronous features",
                        default=False)
    parser.add_argument("-b",
                        action="store_true",
                        dest="annot_beats",
                        help="Use annotated beats",
                        default=False)
    parser.add_argument("-bid",
                        action="store",
                        dest="boundaries_id",
                        help="Algorithm id for the boundaries to use "
                        "(None for C-NMF, and \"gt\" for ground truth)",
                        default=None)
    args = parser.parse_args()
    start_time = time.time()

    # Setup the logger
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s',
        level=logging.INFO)

    # Run the algorithm
    process(args.in_path, feature=args.feature, annot_beats=args.annot_beats,
            boundaries_id=args.boundaries_id, framesync=args.framesync)

    # Done!
    logging.info("Done! Took %.2f seconds." % (time.time() - start_time))

if __name__ == '__main__':
    main()
