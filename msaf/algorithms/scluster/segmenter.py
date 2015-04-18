#!/usr/bin/env python
# coding: utf-8
"""
This script identifies the boundaries of a given track using the Spectral
Clustering method published here:

    Mcfee, B., & Ellis, D. P. W. (2014). Analyzing Song Structure with Spectral
    Clustering. In Proc. of the 15th International Society for Music Information
    Retrieval Conference (pp. 405–410). Taipei, Taiwan.

Original code by Brian McFee from:
    https://github.com/bmcfee/laplacian_segmentation
"""

__author__ = "Oriol Nieto"
__copyright__ = "Copyright 2014, Music and Audio Research Lab (MARL)"
__license__ = "GPL"
__version__ = "1.0"
__email__ = "oriol@nyu.edu"

import logging
import main
import numpy as np

import msaf
from msaf.algorithms.interface import SegmenterInterface


class Segmenter(SegmenterInterface):
    def process(self):
        """Main process.
        Returns
        -------
        est_idxs : np.array(N) or list
            Estimated times for the segment boundaries in frame indeces.
            List if hierarchical segmentation.
        est_labels : np.array(N-1) or list
            Estimated labels for the segments.
            List if hierarchical segmentation.
        """
        # Preprocess to obtain features, times, and input boundary indeces
        F = self._preprocess()

        # Read frame_times
        self.hpcp, self.mfcc, self.tonnetz, beats, dur, self.anal = \
            msaf.io.get_features(self.audio_file, annot_beats=self.annot_beats,
                                 framesync=self.framesync,
                                 pre_features=self.features)
        frame_times = beats
        if self.framesync:
            frame_times = msaf.utils.get_time_frames(dur, self.anal)

        # Brian wants HPCP and MFCC
        # (transosed, because he's that kind of person)
        F = (self.hpcp.T, self.mfcc.T)

        # Do actual segmentation
        est_idxs, est_labels = main.do_segmentation(F, frame_times,
                                                    self.config,
                                                    self.in_bound_idxs,
                                                    self.audio_file)

        if 'numpy' in str(type(est_idxs)):
            # Flat output
            assert est_idxs[0] == 0 and est_idxs[-1] == F[0].shape[1] - 1
            est_idxs, est_labels = self._postprocess(est_idxs, est_labels)
        else:
            # Hierarchical output
            for layer in range(len(est_idxs)):
                assert est_idxs[layer][0] == 0 and \
                    est_idxs[layer][-1] == F[0].shape[1] - 1
                est_idxs[layer], est_labels[layer] = \
                    self._postprocess(est_idxs[layer], est_labels[layer])

        return est_idxs, est_labels

    def processFlat(self):
        """Main process.for flat segmentation.
        Returns
        -------
        est_idxs : np.array(N)
            Estimated times for the segment boundaries in frame indeces.
        est_labels : np.array(N-1)
            Estimated labels for the segments.
        """
        return self.process()

    def processHierarchical(self):
        """Main process.for hierarchial segmentation.
        Returns
        -------
        est_idxs : list
            List with np.arrays for each layer of segmentation containing
            the estimated indeces for the segment boundaries.
        est_labels : list
            List with np.arrays containing the labels for each layer of the
            hierarchical segmentation.
        """
        return self.process()
