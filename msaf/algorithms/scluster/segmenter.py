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

from msaf.algorithms.interface import SegmenterInterface


class Segmenter(SegmenterInterface):
    def process(self):
        """Main process.
        Returns
        -------
        est_times : np.array(N)
            Estimated times for the segment boundaries in seconds.
        est_labels : np.array(N-1)
            Estimated labels for the segments.
        """
        # Preprocess to obtain features, times, and input boundary indeces
        F, frame_times, dur, bound_idxs = self._preprocess()

        # Brian wants HPCP and MFCC
        # (transosed, because he's that kind of person)
        F = (self.hpcp.T, self.mfcc.T)

        # Brian also wants the last duration
        frame_times = np.concatenate((frame_times, [dur]))

        # Do actual segmentation
        bound_idxs, est_labels = main.do_segmentation(F, frame_times,
                                                      self.config,
                                                      bound_idxs)

        # Add first and last boundaries (silence)
        bound_idxs = np.asarray(bound_idxs, dtype=int)
        est_times = np.concatenate(([0], frame_times[bound_idxs]))

        # Post process estimations
        est_times, est_labels = self._postprocess(est_times, est_labels)
        est_times = np.asarray(est_times)
        est_labels = np.asarray(est_labels, dtype=np.float)

        logging.info("Estimated times: %s" % est_times)
        logging.info("Estimated labels: %s" % est_labels)

        return est_times, est_labels
