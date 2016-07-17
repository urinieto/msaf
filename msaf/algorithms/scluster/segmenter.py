#!/usr/bin/env python
# coding: utf-8
import numpy as np

import msaf
from msaf.algorithms.interface import SegmenterInterface
from msaf.base import Features
from . import main


class Segmenter(SegmenterInterface):
    """
    This script identifies the boundaries of a given track using the Spectral
    Clustering method published here:

    Mcfee, B., & Ellis, D. P. W. (2014). Analyzing Song Structure with Spectral
    Clustering. In Proc. of the 15th International Society for Music
    Information Retrieval Conference (pp. 405–410). Taipei, Taiwan.

    Original code by Brian McFee from:
        https://github.com/bmcfee/laplacian_segmentation
    """
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
        # This algorithm only accepts one specific kind of features:
        # Combination of PCP + MFCC. Let's get them:
        pcp_obj = Features.select_features(
            "pcp", self.file_struct, self.annot_beats, self.framesync)
        mfcc_obj = Features.select_features(
            "mfcc", self.file_struct, self.annot_beats, self.framesync)

        # Get frame times and make sure they're the same in both features
        frame_times = pcp_obj.frame_times
        assert np.array_equal(frame_times, mfcc_obj.frame_times)

        # Brian wants PCP and MFCC
        # (tranpsosed, because he's that kind of person)
        F = (pcp_obj.features.T, mfcc_obj.features.T)

        # Do actual segmentation
        est_idxs, est_labels = main.do_segmentation(
            F, frame_times, self.config, self.in_bound_idxs)

        return est_idxs, est_labels, F

    def processFlat(self):
        """Main process.for flat segmentation.
        Returns
        -------
        est_idxs : np.array(N)
            Estimated times for the segment boundaries in frame indeces.
        est_labels : np.array(N-1)
            Estimated labels for the segments.
        """
        est_idxs, est_labels, F = self.process()
        assert est_idxs[0] == 0 and est_idxs[-1] == F[0].shape[1] - 1
        return self._postprocess(est_idxs, est_labels)

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
        est_idxs, est_labels, F = self.process()
        for layer in range(len(est_idxs)):
            assert est_idxs[layer][0] == 0 and \
                est_idxs[layer][-1] == F[0].shape[1] - 1
            est_idxs[layer], est_labels[layer] = \
                self._postprocess(est_idxs[layer], est_labels[layer])
        return est_idxs, est_labels
