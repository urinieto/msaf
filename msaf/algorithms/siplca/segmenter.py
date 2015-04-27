#!/usr/bin/env python

# Copyright (C) 2009-2010 Ron J. Weiss (ronw@nyu.edu)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

"""Music structure segmentation using SI-PLCA

This module contains an implementation of the algorithm for music
structure segmentation described in [1].  It is based on
Shift-invariant Probabilistic Latent Component Analysis, a variant of
convolutive non-negative matrix factorization (NMF).  See plca.py for
more details.

Examples
--------
>>> import segmenter
>>> wavfile = '/path/to/come_together.wav'
>>> rank = 4  # rank corresponds to the number of segments
>>> win = 60  # win controls the length of each chroma pattern
>>> niter = 200  # number of iterations to perform
>>> np.random.seed(123)  # Make this reproduceable
>>> labels = segmenter.segment_wavfile(wavfile, win=win, rank=rank,
...                                    niter=niter, plotiter=10)
INFO:plca:Iteration 0: divergence = 10.065992
INFO:plca:Iteration 50: divergence = 9.468196
INFO:plca:Iteration 100: divergence = 9.421632
INFO:plca:Iteration 150: divergence = 9.409279
INFO:root:Iteration 199: final divergence = 9.404961
INFO:segmenter:Removing 2 segments shorter than 32 frames

.. image::come_together-segmentation.png

>>> print labels
0.0000 21.7480 segment0
21.7480 37.7640 segment1
37.7640 55.1000 segment0
55.1000 76.1440 segment1
76.1440 95.1640 segment0
95.1640 121.2360 segment1
121.2360 158.5360 segment2
158.5360 180.8520 segment1
180.8520 196.5840 segment0
196.5840 255.8160 segment3

See Also
--------
segmenter.extract_features : Beat-synchronous chroma feature extraction
segmenter.segment_song : Performs segmentation
segmenter.evaluate_segmentation : Evaluate frame-wise segmentation
segmenter.convert_labels_to_segments : Generate HTK formatted list of segments
                                       from frame-wise labels
plca.SIPLCA : Implementation of Shift-invariant PLCA

References
----------
 [1] R. J. Weiss and J. P. Bello. "Identifying Repeated Patterns in
     Music Using Sparse Convolutive Non-Negative Matrix
     Factorization". In Proc. International Conference on Music
     Information Retrieval (ISMIR), 2010.

Copyright (C) 2009-2010 Ron J. Weiss <ronw@nyu.edu>

LICENSE: This module is licensed under the GNU GPL. See COPYING for details.
"""

import logging
import numpy as np

import msaf
import msaf.input_output as io
from msaf.algorithms.interface import SegmenterInterface

# Local stuff
import plca

logging.basicConfig(level=logging.INFO,
                    format='%(levelname)s %(name)s %(asctime)s '
                    '%(filename)s:%(lineno)d  %(message)s')
logger = logging.getLogger('segmenter')


def segment_song(seq, rank=4, win=32, seed=None,
                 nrep=1, minsegments=3, maxlowen=10, maxretries=5,
                 uninformativeWinit=False, uninformativeHinit=True,
                 viterbi_segmenter=False, align_downbeats=False, **kwargs):
    """Segment the given feature sequence using SI-PLCA

    Parameters
    ----------
    seq : array, shape (F, T)
        Feature sequence to segment.
    rank : int
        Number of patterns (unique segments) to search for.
    win : int
        Length of patterns in frames.
    seed : int
        Random number generator seed.  Defaults to None.
    nrep : int
        Number of times to repeat the analysis.  The repetition with
        the lowest reconstrucion error is returned.  Defaults to 1.
    minsegments : int
        Minimum number of segments in the output.  The analysis is
        repeated until the output contains at least `minsegments`
        segments is or `maxretries` is reached.  Defaults to 3.
    maxlowen : int
        Maximum number of low energy frames in the SIPLCA
        reconstruction.  The analysis is repeated if it contains too
        many gaps.  Defaults to 10.
    maxretries : int
        Maximum number of retries to perform if `minsegments` or
       `maxlowen` are not satisfied.  Defaults to 5.
    uninformativeWinit : boolean
        If True, `W` is initialized to have a flat distribution.
        Defaults to False.
    uninformativeHinit : boolean
        If True, `H` is initialized to have a flat distribution.
        Defaults to True.
    viterbi_segmenter : boolean
        If True uses uses the Viterbi algorithm to convert SIPLCA
        decomposition into segmentation, otherwises uses the process
        described in [1].  Defaults to False.
    align_downbeats : boolean
        If True, postprocess the SIPLCA analysis to find the optimal
        alignments of the components of W with V.  I.e. try to align
        the first column of W to the downbeats in the song.  Defaults
        to False.
    kwargs : dict
        Keyword arguments passed to plca.SIPLCA.analyze.  See
        plca.SIPLCA for more details.

    Returns
    -------
    labels : array, length `T`
        Segment label for each frame of `seq`.
    W : array, shape (`F`, `rank`, `win`)
        Set of `F` x `win` shift-invariant basis functions found in `seq`.
    Z : array, length `rank`
        Set of mixing weights for each basis.
    H : array, shape (`rank`, `T`)
        Activations of each basis in time.
    segfun : array, shape (`rank`, `T`)
        Raw segmentation function used to generate segment labels from
        SI-PLCA decomposition.  Corresponds to $\ell_k(t)$ in [1].
    norm : float
        Normalization constant to make `seq` sum to 1.

    Notes
    -----
    The experimental results reported in [1] were found using the
    default values for all keyword arguments while varying kwargs.

    """
    seq = seq.copy()

    #logger.debug('Using random seed %s.', seed)
    np.random.seed(seed)

    if 'alphaWcutoff' in kwargs and 'alphaWslope' in kwargs:
        kwargs['alphaW'] = create_sparse_W_prior((seq.shape[0], win),
                                                 kwargs['alphaWcutoff'],
                                                 kwargs['alphaWslope'])
        del kwargs['alphaWcutoff']
        del kwargs['alphaWslope']

    F, T = seq.shape
    if uninformativeWinit:
        kwargs['initW'] = np.ones((F, rank, win)) / (F*win)
    if uninformativeHinit:
        kwargs['initH'] = np.ones((rank, T)) / T

    outputs = []
    for n in xrange(nrep):
        outputs.append(plca.SIPLCA.analyze(seq, rank=rank, win=win, **kwargs))
    div = [x[-1] for x in outputs]
    W, Z, H, norm, recon, div = outputs[np.argmin(div)]

    # Need to rerun segmentation if there are too few segments or
    # if there are too many gaps in recon (i.e. H)
    lowen = seq.shape[0] * np.finfo(float).eps
    nlowen_seq = np.sum(seq.sum(0) <= lowen)
    if nlowen_seq > maxlowen:
        maxlowen = nlowen_seq
    nlowen_recon = np.sum(recon.sum(0) <= lowen)
    nretries = maxretries
    while (len(Z) < minsegments or nlowen_recon > maxlowen) and nretries > 0:
        nretries -= 1
        #logger.debug('Redoing SIPLCA analysis (len(Z) = %d, number of '
                    #'low energy frames = %d).', len(Z), nlowen_recon)
        outputs = []
        for n in xrange(nrep):
            outputs.append(plca.SIPLCA.analyze(seq, rank=rank, win=win,
                                               **kwargs))
        div = [x[-1] for x in outputs]
        W, Z, H, norm, recon, div = outputs[np.argmin(div)]
        nlowen_recon = np.sum(recon.sum(0) <= lowen)

    if viterbi_segmenter:
        segmentation_function = nmf_analysis_to_segmentation_using_viterbi_path
    else:
        segmentation_function = nmf_analysis_to_segmentation

    labels, segfun = segmentation_function(seq, win, W, Z, H, **kwargs)

    return labels, W, Z, H, segfun, norm


def create_sparse_W_prior(shape, cutoff, slope):
    """Constructs sparsity parameters for W (alphaW) to learn pattern length

    Follows equation (6) in the ISMIR paper referenced in this
    module's docstring.
    """

    # W.shape is (ndim, nseg, nwin)
    prior = np.zeros(shape[-1])
    prior[cutoff:] = prior[0] + slope * np.arange(shape[-1] - cutoff)

    alphaW = np.zeros((shape[0], 1, shape[-1]))
    alphaW[:,:] = prior
    return alphaW

def nmf_analysis_to_segmentation(seq, win, W, Z, H, min_segment_length=32,
                                 use_Z_for_segmentation=True, **ignored_kwargs):
    if not use_Z_for_segmentation:
        Z = np.ones(Z.shape)

    segfun = []
    for n, (w,z,h) in enumerate(zip(np.transpose(W, (1, 0, 2)), Z, H)):
        reconz = plca.SIPLCA.reconstruct(w, z, h)
        score = np.sum(reconz, 0)

        # Smooth it out
        score = np.convolve(score, np.ones(min_segment_length), 'same')
        # kernel_size = min_segment_length
        # if kernel_size % 2 == 0:
        #     kernel_size += 1
        # score = sp.signal.medfilt(score, kernel_size)
        segfun.append(score)

    segfun = np.array(segfun)
    segfun /= segfun.max()

    labels = np.argmax(np.asarray(segfun), 0)
    remove_short_segments(labels, min_segment_length)

    return labels, segfun

def nmf_analysis_to_segmentation_using_viterbi_path(seq, win, W, Z, H,
                                                    selfloopprob=0.9,
                                                    use_Z_for_segmentation=True,
                                                    min_segment_length=32,
                                                    **ignored_kwargs):
    if not use_Z_for_segmentation:
        Z = np.ones(Z.shape)

    rank = len(Z)
    T = H.shape[1]
    likelihood = np.empty((rank, T))
    for z in xrange(rank):
        likelihood[z] = plca.SIPLCA.reconstruct(W[:,z], Z[z], H[z]).sum(0)

    transmat = np.zeros((rank, rank))
    for z in xrange(rank):
        transmat[z,:] = (1 - selfloopprob) / (rank - 1 + np.finfo(float).eps)
        transmat[z,z] = selfloopprob

    # Find Viterbi path.
    loglikelihood = np.log(likelihood)
    logtransmat = np.log(transmat)
    lattice = np.zeros(loglikelihood.shape)
    traceback = np.zeros(loglikelihood.shape, dtype=np.int)
    lattice[0] = loglikelihood[0]
    for n in xrange(1, T):
        pr = logtransmat.T + lattice[:,n-1]
        lattice[:,n] = np.max(pr, axis=1) + loglikelihood[:,n]
        traceback[:,n] = np.argmax(pr, axis=1)

    # Do traceback to find most likely path.
    reverse_state_sequence = []
    s = lattice[:,-1].argmax()
    for frame in reversed(traceback.T):
        reverse_state_sequence.append(s)
        s = frame[s]
    labels = np.array(list(reversed(reverse_state_sequence)))

    remove_short_segments(labels, min_segment_length)

    return labels, likelihood

def remove_short_segments(labels, min_segment_length):
    """Remove segments shorter than min_segment_length."""
    segment_borders = np.nonzero(np.diff(labels))[0]
    short_segments_idx = np.nonzero(np.diff(segment_borders)
                                    < min_segment_length)[0]
    #logger.debug('Removing %d segments shorter than %d frames',
                #len(short_segments_idx), min_segment_length)
    # Remove all adjacent short_segments.
    segment_borders[short_segments_idx]

    for idx in short_segments_idx:
        start = segment_borders[idx]
        try:
            end = segment_borders[idx + 1] + 1
        except IndexError:
            end = len(labels)

        try:
            label = labels[start - 1]
        except IndexError:
            label = labels[end]

        labels[start:end] = label


def compute_effective_pattern_length(w):
    wsum = w.sum(0)
    # Find all taus in w that contain significant probability mass.
    nonzero_idx, = np.nonzero(wsum > wsum.min())
    winlen = nonzero_idx[-1] - nonzero_idx[0] + 1
    return winlen

def convert_labels_to_segments(labels, num_frames):
    """Covert frame-wise segmentation labels to a list of segments in HTK
    format"""

    # Nonzero points in diff(labels) correspond to the final frame of
    # a segment (so just index into labels to find the segment label)
    boundaryidx = np.concatenate(([0], np.nonzero(np.diff(labels))[0],
                                  [num_frames-1]))
    seglabels = labels[boundaryidx[1:]]

    return boundaryidx, seglabels

def _compute_summary_correlation(A, B):
    return sum(np.correlate(A[x], B[x], 'full') for x in xrange(A.shape[0]))

def shift_key_to_zero(W, Z, H):
    newW = np.zeros(W.shape)
    newH = np.zeros(H.shape)
    for k in xrange(len(Z)):
        key_profile = H[k].sum(1)
        main_key = np.argmax(key_profile)
        newW[:,k] = plca.shift(W[:,k], main_key, axis=0, circular=True)
        newH[k] = plca.shift(H[k], -main_key, axis=0, circular=True)
    return newW, Z, newH


def segment_wavfile(features, **kwargs):
    """Convenience function to compute segmentation.

    Keyword arguments are passed into segment_song.

    Returns a string containing list of segments in HTK label format.
    """
    labels, W, Z, H, segfun, norm = segment_song(features, **kwargs)
    est_idxs, est_labels = convert_labels_to_segments(labels,
                                                      features.shape[1])
    return est_idxs, est_labels


def use_in_bounds(audio_file, in_bound_idxs, feats, config):
    """We update the initial matrices using the annotated bounds."""
    n_segments = len(in_bound_idxs) - 1
    max_beats_segment = np.max(np.diff(in_bound_idxs))

    # Inititalize the W and H matrices using the previously found bounds
    initW = np.zeros((feats.shape[1], n_segments, max_beats_segment))
    initH = np.zeros((n_segments, feats.shape[0]))
    for i in xrange(n_segments):
        dur = in_bound_idxs[i+1] - in_bound_idxs[i]
        initW[:, i, :dur] = feats[in_bound_idxs[i]:in_bound_idxs[i+1]].T
        initH[i, in_bound_idxs[i]] = 1

    # Update parameters
    config["win"] = max_beats_segment
    config["rank"] = n_segments
    config["initW"] = initW
    config["initH"] = initH

    return config, in_bound_idxs


class Segmenter(SegmenterInterface):
    def processFlat(self):
        """Main process.

        Returns
        -------
        est_idxs : np.array(N)
            Estimated times for the segment boundaries in frame times.
        est_labels : np.array(N-1)
            Estimated labels for the segments.
        """
        # Preprocess to obtain features
        #F = self._preprocess(valid_features=["hpcp", "tonnetz"])
        F = self._preprocess()

        # Read frame times
        self.hpcp, self.mfcc, self.tonnetz, self.cqt, beats, dur, self.anal = \
            msaf.io.get_features(self.audio_file, annot_beats=self.annot_beats,
                                 framesync=self.framesync,
                                 pre_features=self.features)
        frame_times = beats
        if self.framesync:
            frame_times = msaf.utils.get_time_frames(dur, self.anal)

        # Additional SI-PLCA params
        self.config["plotiter"] = None

        # Update parameters if using additional boundaries
        if self.in_bound_idxs is not None:
            self.config, bound_idxs = use_in_bounds(self.audio_file,
                                                    self.in_bound_idxs,
                                                    F,
                                                    self.config)

        # Make segmentation
        est_idxs, est_labels = segment_wavfile(F.T, **self.config)

        assert est_idxs[0] == 0 and est_idxs[-1] == F.shape[0] - 1

        # Align with annotated boundaries if needed
        if self.in_bound_idxs is not None:
            est_labels = msaf.utils.synchronize_labels(self.in_bound_idxs,
                                                       est_idxs,
                                                       est_labels,
                                                       F.shape[0])
            est_idxs = self.in_bound_idxs

        # Remove paramaters that we don't want to store
        self.config.pop("initW", None)
        self.config.pop("initH", None)
        self.config.pop("plotiter", None)
        self.config.pop("win", None)
        self.config.pop("rank", None)

        # Postprocess the estimations
        est_idxs, est_labels = self._postprocess(est_idxs, est_labels)

        return est_idxs, est_labels
