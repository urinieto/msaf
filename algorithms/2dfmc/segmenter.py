#!/usr/bin/env python
# coding: utf-8
"""
This method labels segments using the 2D-FMC method described here:

Nieto, O., Bello, J.P., Music Segment Similarity Using 2D-Fourier Magnitude
    Coefficients. Proc. of the 39th IEEE International Conference on Acoustics,
    Speech, and Signal Processing (ICASSP). Florence, Italy, 2014.
"""

__author__ = "Oriol Nieto"
__copyright__ = "Copyright 2014, Music and Audio Research Lab (MARL)"
__license__ = "GPL"
__version__ = "1.0"
__email__ = "oriol@nyu.edu"

import argparse
import sys
import time
import logging

import numpy as np
import pylab as plt

import scipy.cluster.vq as vq

# Local stuff
import utils_2dfmc as utils2d
from xmeans import XMeans

sys.path.append("../../")
import msaf_io as MSAF
#import eval as EV
import utils as U

MIN_LEN = 4


def get_pcp_segments(PCP, bound_idxs):
    """Returns a set of segments defined by the bound_idxs."""
    pcp_segments = []
    for i in xrange(len(bound_idxs)-1):
        pcp_segments.append(PCP[bound_idxs[i]:bound_idxs[i+1], :])
    return pcp_segments


def pcp_segments_to_2dfmc_max(pcp_segments):
    """From a list of PCP segments, return a list of 2D-Fourier Magnitude
        Coefs using the maximumg segment size and zero pad the rest."""
    # Get maximum segment size
    max_len = max([pcp_segment.shape[0] for pcp_segment in pcp_segments])

    OFFSET = 4
    fmcs = []
    for pcp_segment in pcp_segments:
        # Zero pad if needed
        X = np.zeros((max_len, 12))
        #X[:pcp_segment.shape[0],:] = pcp_segment
        if pcp_segment.shape[0] <= OFFSET:
            X[:pcp_segment.shape[0], :] = pcp_segment
        else:
            X[:pcp_segment.shape[0]-OFFSET, :] = \
                pcp_segment[OFFSET/2:-OFFSET/2, :]

        # 2D-FMC
        try:
            fmcs.append(utils2d.compute_ffmc2d(X))
        except:
            logging.warning("Couldn't compute the 2D Fourier Transform")
            fmcs.append(np.zeros((X.shape[0] * X.shape[1]) / 2 + 1))

        # Normalize
        #fmcs[-1] = fmcs[-1] / fmcs[-1].max()

    return np.asarray(fmcs)


def compute_labels_kmeans(fmcs, k=6):
    # Removing the higher frequencies seem to yield better results
    fmcs = fmcs[:, fmcs.shape[1]/2:]

    fmcs = np.log1p(fmcs)
    wfmcs = vq.whiten(fmcs)

    dic, dist = vq.kmeans(wfmcs, k, iter=100)
    labels, dist = vq.vq(wfmcs, dic)

    return labels


def compute_similarity(PCP, bound_idxs, xmeans=False, k=5):
    """Main function to compute the segment similarity of file file_struct."""

    # Get PCP segments
    pcp_segments = get_pcp_segments(PCP, bound_idxs)

    # Get the 2d-FMCs segments
    fmcs = pcp_segments_to_2dfmc_max(pcp_segments)

    # Compute the labels using kmeans
    if xmeans:
        xm = XMeans(fmcs, plot=False)
        k = xm.estimate_K_knee(th=0.01, maxK=8)
    labels_est = compute_labels_kmeans(fmcs, k=k)

    # Plot results
    #plot_pcp_wgt(PCP, bound_idxs)

    return labels_est


def process(in_path, annot_beats=False, xmeans=False, k=5):
    """Main process.

    Parameters
    ----------
    in_path : str
        Path to audio file
    annot_beats : boolean
        Whether to use annotated beats or not
    feature : str
        Identifier of the features to use
    """
    # Read features
    chroma, mfcc, beats, dur = MSAF.get_features(in_path,
                                                 annot_beats=annot_beats)

    # Read annotated bounds
    try:
        bound_idxs = MSAF.read_annot_bound_frames(in_path, beats)[1:-1]
    except:
        logging.warning("No annotated boundaries in file %s" % in_path)

    # Use specific feature
    F = U.lognormalize_chroma(chroma)  # Normalize chromas

    # Find the labels using 2D-FMCs
    est_labels = compute_similarity(F, bound_idxs, xmeans=xmeans, k=k)

    # Get times
    est_times = beats[bound_idxs]  # Index to times

    # Add first and last boundary
    bound_idxs = np.concatenate(([0], bound_idxs)).astype(int)
    est_times = beats[bound_idxs]  # Index to times
    est_times = np.concatenate((est_times, [dur]))  # Last bound
    est_labels = np.concatenate(([-1], est_labels, [-1]))

    # Print out
    logging.info("Estimated times: %s" % est_times)
    logging.info("Estimated labels: %s" % est_labels)

    # Make sure labels and times match
    assert len(est_times) - 1 == len(est_labels), "Labels and times do not " \
        "match: len(est_times) = %d, len(est_labels) = %d." % \
        (len(est_times), len(est_labels))

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
    parser.add_argument("-k",
                        action="store",
                        dest="k",
                        help="Number of unique segments",
                        default=6)
    parser.add_argument("-x",
                        action="store_true",
                        dest="xmeans",
                        help="Estimate K using the BIC method",
                        default=False)
    parser.add_argument("-b",
                        action="store_true",
                        dest="annot_beats",
                        help="Use annotated beats",
                        default=False)
    args = parser.parse_args()
    start_time = time.time()

    # Setup the logger
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s',
        level=logging.INFO)

    # Run the algorithm
    process(args.in_path, annot_beats=args.annot_beats, xmeans=args.xmeans,
            k=args.k)

    # Done!
    logging.info("Done! Took %.2f seconds." % (time.time() - start_time))

if __name__ == '__main__':
    main()

