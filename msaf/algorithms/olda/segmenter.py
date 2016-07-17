#!/usr/bin/env python
# CREATED:2013-08-22 12:20:01 by Brian McFee <brm2132@columbia.edu>
'''Music segmentation using timbre, pitch, repetition and time.
'''
import argparse
import logging
import sys

import numpy as np
import scipy.signal
import scipy.linalg

import librosa
import msaf

from msaf.algorithms.interface import SegmenterInterface
from msaf.base import Features

REP_WIDTH = 3
REP_FILTER = 7
N_MFCC = 32
N_CHROMA = 12
N_REP = 32

# mfcc, chroma, repetitions for each, and 4 time features
__DIMENSION = N_MFCC + N_CHROMA + 2 * N_REP + 4


def features(file_struct, annot_beats=False, framesync=False):
    '''Feature-extraction for audio segmentation
    Arguments:
        file_struct -- msaf.io.FileStruct
        paths to the input files in the Segmentation dataset

    Returns:
        - X -- ndarray

            beat-synchronous feature matrix:
            MFCC (mean-aggregated)
            Chroma (median-aggregated)
            Latent timbre repetition
            Latent chroma repetition
            Time index
            Beat index

        - dur -- float
            duration of the track in seconds

    '''
    def compress_data(X, k):
        Xtemp = X.dot(X.T)
        if len(Xtemp) == 0:
            return None
        e_vals, e_vecs = np.linalg.eig(Xtemp)

        e_vals = np.maximum(0.0, np.real(e_vals))
        e_vecs = np.real(e_vecs)

        idx = np.argsort(e_vals)[::-1]

        e_vals = e_vals[idx]
        e_vecs = e_vecs[:, idx]

        # Truncate to k dimensions
        if k < len(e_vals):
            e_vals = e_vals[:k]
            e_vecs = e_vecs[:, :k]

        # Normalize by the leading singular value of X
        Z = np.sqrt(e_vals.max())

        if Z > 0:
            e_vecs = e_vecs / Z

        return e_vecs.T.dot(X)

    # Latent factor repetition features
    def repetition(X, metric='euclidean'):
        R = librosa.segment.recurrence_matrix(
            X, k=2 * int(np.ceil(np.sqrt(X.shape[1]))),
            width=REP_WIDTH, metric=metric, sym=False).astype(np.float32)

        P = scipy.signal.medfilt2d(librosa.segment.structure_feature(R),
                                   [1, REP_FILTER])

        # Discard empty rows.
        # This should give an equivalent SVD, but resolves some numerical
        # instabilities.
        P = P[P.any(axis=1)]

        return compress_data(P, N_REP)

    #########
    # '\tloading annotations and features of ', audio_path
    pcp_obj = Features.select_features("pcp", file_struct, annot_beats,
                                       framesync)
    mfcc_obj = Features.select_features("mfcc", file_struct, annot_beats,
                                        framesync)
    chroma = pcp_obj.features
    mfcc = mfcc_obj.features
    beats = pcp_obj.frame_times
    dur = pcp_obj.dur

    # Sampling Rate
    sr = msaf.config.sample_rate

    ##########
    #print '\treading beats'
    B = beats[:chroma.shape[0]]
    #beat_frames = librosa.time_to_frames(B, sr=sr,
                                         #hop_length=msaf.config.hop_size)
    #print beat_frames, len(beat_frames), uidx

    #########
    M = mfcc.T
    #plt.imshow(M, interpolation="nearest", aspect="auto"); plt.show()

    #########
    # Get the beat-sync chroma
    C = chroma.T
    C += C.min() + 0.1
    C = C / C.max(axis=0)
    C = 80 * np.log10(C)  # Normalize from -80 to 0
    #plt.imshow(C, interpolation="nearest", aspect="auto"); plt.show()

    # Time-stamp features
    N = np.arange(float(chroma.shape[0]))

    #########
    #print '\tgenerating structure features'

    # TODO:  This might fail if audio file (or number of beats) is too small
    R_timbre = repetition(librosa.feature.stack_memory(M))
    R_chroma = repetition(librosa.feature.stack_memory(C))
    if R_timbre is None or R_chroma is None:
        return None, dur

    R_timbre += R_timbre.min()
    R_timbre /= R_timbre.max()
    R_chroma += R_chroma.min()
    R_chroma /= R_chroma.max()
    #plt.imshow(R_chroma, interpolation="nearest", aspect="auto"); plt.show()

    # Stack it all up
    #print M.shape, C.shape, R_timbre.shape, R_chroma.shape, len(B), len(N)
    X = np.vstack([M, C, R_timbre, R_chroma, B, B / dur, N,
                   N / float(chroma.shape[0])])

    #plt.imshow(X, interpolation="nearest", aspect="auto"); plt.show()

    return X, dur


def gaussian_cost(X):
    '''Return the average log-likelihood of data under a standard normal
    '''

    d, n = X.shape

    if n < 2:
        return 0

    sigma = np.var(X, axis=1, ddof=1)

    cost = -0.5 * d * n * np.log(2. * np.pi) - 0.5 * (n - 1.) * np.sum(sigma)
    return cost


def clustering_cost(X, boundaries):

    # Boundaries include beginning and end frames, so k is one less
    k = len(boundaries) - 1

    d, n = map(float, X.shape)

    # Compute the average log-likelihood of each cluster
    cost = [gaussian_cost(X[:, start:end]) for
            (start, end) in zip(boundaries[:-1], boundaries[1:])]

    cost = - 2 * np.sum(cost) / n + 2 * (d * k)

    return cost


def get_k_segments(X, k):

    # Step 1: run ward
    boundaries = librosa.segment.agglomerative(X, k)

    # Add first and last boundary indeces
    boundaries = np.unique(np.concatenate(([0], boundaries, [X.shape[1]-1])))

    # Step 2: compute cost
    cost = clustering_cost(X, boundaries)

    return boundaries, cost


def get_segments(X, kmin=8, kmax=32):

    cost_min = np.inf
    S_best = []
    best_k = -1
    for k in range(kmax, kmin, -1):
        S, cost = get_k_segments(X, k)
        if cost < cost_min:
            cost_min = cost
            S_best = S
            best_k = k
        else:
            break

    return S_best


def process_arguments():
    parser = argparse.ArgumentParser(description='Music segmentation')

    parser.add_argument('-t',
                        '--transform',
                        dest='transform',
                        required=False,
                        type=str,
                        help='npy file containing the linear projection',
                        default=None)

    parser.add_argument('input_song',
                        action='store',
                        help='path to input audio data')

    parser.add_argument('output_file',
                        action='store',
                        help='path to output segment file')

    return vars(parser.parse_args(sys.argv[1:]))


def load_transform(transform_file):

    if transform_file is None:
        W = np.eye(__DIMENSION)
    else:
        W = np.load(transform_file)

    return W


def get_num_segs(duration, MIN_SEG=10.0, MAX_SEG=45.0):
    kmin = max(1, np.floor(duration / MAX_SEG).astype(int))
    kmax = max(2, np.ceil(duration / MIN_SEG).astype(int))

    return kmin, kmax


class Segmenter(SegmenterInterface):
    """
    This class implements the algorithm described here:

    McFee, B. and Ellis, D.P.W., Learning to segment songs with ordinal linear
    discriminant analysis. International conference on acoustics, speech and
    signal processing (ICASSP). 2014 (`PDF`_).

    .. _PDF: https://bmcfee.github.io/papers/icassp2014_segments.pdf
    """
    def processFlat(self):
        """Main process for flat segmentation.
        Returns
        -------
        est_idxs : np.array(N)
            Estimated times for the segment boundaries in frame indeces.
        est_labels : np.array(N-1)
            Estimated labels for the segments.
        """
        # Preprocess to obtain features and duration
        F, dur = features(self.file_struct, self.annot_beats, self.framesync)

        try:
            # Load and apply transform
            W = load_transform(self.config["transform"])
            F = W.dot(F)

            # Get Segments
            kmin, kmax = get_num_segs(dur)
            est_idxs = get_segments(F, kmin=kmin, kmax=kmax)
        except:
            # The audio file is too short, only beginning and end
            logging.warning("Audio file too short! "
                            "Only start and end boundaries.")
            est_idxs = [0, F.shape[1] - 1]

        # Make sure that the first and last boundaries are included
        assert est_idxs[0] == 0 and est_idxs[-1] == F.shape[1] - 1

        # Empty labels
        est_labels = np.ones(len(est_idxs) - 1) * -1

        # Post process estimations
        est_idxs, est_labels = self._postprocess(est_idxs, est_labels)

        return est_idxs, est_labels

    def processHierarchical(self):
        """Main process for hierarchical segmentation.
        Returns
        -------
        est_idxs : list
            List containing estimated times for each layer in the hierarchy
            as np.arrays
        est_labels : list
            List containing estimated labels for each layer in the hierarchy
            as np.arrays
        """
        # Preprocess to obtain features, times, and input boundary indeces
        F, dur = features(self.file_struct, self.annot_beats, self.framesync)

        try:
            # Load and apply transform
            W = load_transform(self.config["transform"])
            F = W.dot(F)

            # Get Segments
            kmin, kmax = get_num_segs(dur)

            # Run algorithm layer by layer
            est_idxs = []
            est_labels = []
            for k in range(kmax, kmin, -1):
                S, cost = get_k_segments(F, k)
                est_idxs.append(S)
                est_labels.append(np.ones(len(S) - 1) * -1)

                # Make sure that the first and last boundaries are included
                assert est_idxs[-1][0] == 0 and \
                    est_idxs[-1][-1] == F.shape[1] - 1, "Layer %d does not " \
                    "start or end in the right frame(s)." % k

                # Post process layer
                est_idxs[-1], est_labels[-1] = \
                        self._postprocess(est_idxs[-1], est_labels[-1])
        except:
            # The audio file is too short, only beginning and end
            logging.warning("Audio file too short! "
                            "Only start and end boundaries.")
            est_idxs = [np.array([0, F.shape[1] - 1])]
            est_labels = [np.ones(1) * -1]

        return est_idxs, est_labels
