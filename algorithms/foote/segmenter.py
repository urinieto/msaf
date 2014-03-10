#!/usr/bin/env python
# coding: utf-8
"""
This script identifies the boundaries of a given track using the Foote
method:

Foote, J. (2000). Automatic Audio Segmentation Using a Measure Of Audio Novelty. 
In Proc. of the IEEE International Conference of Multimedia and Expo
(pp. 452–455). New York City, NY, USA.
"""

__author__ = "Oriol Nieto"
__copyright__ = "Copyright 2014, Music and Audio Research Lab (MARL)"
__license__ = "GPL"
__version__ = "1.0"
__email__ = "oriol@nyu.edu"

import argparse
import glob
import jams
import logging
import os
import pylab as plt
import numpy as np
import time
import librosa
from scipy.spatial import distance
from scipy import signal
from scipy.ndimage import filters

import sys
sys.path.append( "../../" )
import msaf_io as MSAF
import utils as U

def median_filter(X, M=8):
    """Median filter along the first axis of the feature matrix X."""
    for i in xrange(X.shape[1]):
        X[:,i] = filters.median_filter(X[:,i], size=M)
    return X

def compute_gaussian_krnl(M):
    """Creates a gaussian kernel following Foote's paper."""
    g = signal.gaussian(M, M/3., sym=True)
    G = np.dot(g.reshape(-1,1), g.reshape(1,-1))
    G[M/2:,:M/2] = -G[M/2:,:M/2]
    G[:M/2,M/2:] = -G[:M/2,M/2:]
    return G


def compute_ssm(X, metric="seuclidean"):
    """Computes the self-similarity matrix of X."""
    D = distance.pdist(X, metric=metric)
    D = distance.squareform(D)
    D /= D.max()
    return 1-D

def compute_nc(X, G):
    """Computes the novelty curve from the self-similarity matrix X and
        the gaussian kernel G."""
    N = X.shape[0]
    M = G.shape[0]
    nc = np.zeros(N)

    for i in xrange(M/2, N-M/2+1):
        nc[i] = np.sum(X[i-M/2:i+M/2, i-M/2:i+M/2] * G)

    return nc

def process(in_path, feature="hpcp", annot_beats=False):
    """Main process."""

    # Foote's params
    M = 16  # Size of gaussian kernel
    m = 4   # Size of median filter

    # Read features
    chroma, mfcc, beats, dur = MSAF.get_features(in_path, annot_beats=annot_beats)

    # Use specific feature
    if feature == "hpcp":
        F = U.lognormalize_chroma(chroma) #Normalize chromas
    elif "mfcc":
        F = mfcc
    else:
        logging.error("Feature type not recognized: %s" % feature)

    # Median filter
    F = median_filter(F, M=m)
    plt.imshow(F.T, interpolation="nearest", aspect="auto"); plt.show()

    # Self similarity matrix
    S = compute_ssm(F)

    # Compute gaussian kernel
    G = compute_gaussian_krnl(M)

    # Compute the novelty curve
    nc = compute_nc(S, G)

    plt.figure(1)
    plt.plot(nc); 
    plt.figure(2)
    plt.imshow(S, interpolation="nearest", aspect="auto"); plt.show()


def main():
    """Main function to parse the arguments and call the main process."""
    parser = argparse.ArgumentParser(description=
        "Segments the given audio file using the Foote's method.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("in_path",
                        action="store",
                        help="Input path to the audio file")
    parser.add_argument("-f", 
                        action="store_true", 
                        dest="feature",
                        help="Feature to use (mfcc or hpcp)",
                        default="hpcp")
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
    process(args.in_path, feature=args.feature, annot_beats=args.annot_beats)

    # Done!
    logging.info("Done! Took %.2f seconds." % (time.time() - start_time))

if __name__ == '__main__':
    main()