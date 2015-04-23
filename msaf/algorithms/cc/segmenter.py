#!/usr/bin/env python
# coding: utf-8
"""
This script segments a given track using the Constrained Clustering method
described here:

Levy, M., & Sandler, M. (2008). Structural Segmentation of Musical Audio by
Constrained Clustering. IEEE Transactions on Audio, Speech, and Language
Processing, 16(2), 318–326. doi:10.1109/TASL.2007.910781
"""

import logging
import numpy as np

# Local stuff
from msaf.algorithms.interface import SegmenterInterface
try:
    from msaf.algorithms.cc import cc_segmenter
except:
    pass


class Segmenter(SegmenterInterface):
    def processFlat(self):
        """Main process.
        Returns
        -------
        est_idxs : np.array(N)
            Estimated indeces the segment boundaries in frames.
        est_labels : np.array(N-1)
            Estimated labels for the segments.
        """
        min_frames = 15
        # Preprocess to obtain features, times, and input boundary indeces
        F = self._preprocess(normalize=True)
        #import pylab as plt
        #plt.imshow(F.T, interpolation="nearest", aspect="auto")
        #plt.show()

        # Check if the cc module is compiled
        try:
            cc_segmenter
        except:
            logging.warning("CC not compiled, returning empty segmentation")
            if self.in_bound_idxs is None:
                return np.array([0, F.shape[0] - 1]), [0]
            else:
                return self.in_bound_idxs, [0] * (len(self.in_bound_idxs) - 1)

        if F.shape[0] > min_frames:
            if self.feature_str == "hpcp" or self.feature_str == "tonnetz" or \
                    self.feature_str == "cqt":
                is_harmonic = True
            elif self.feature_str == "mfcc":
                is_harmonic = False
            else:
                raise RuntimeError("Feature type %s is not valid" %
                                   self.feature_str)

            in_bound_idxs = self.in_bound_idxs
            if self.in_bound_idxs is None:
                in_bound_idxs = []

            if F.shape[0] > 2 and \
                    (len(in_bound_idxs) > 2 or len(in_bound_idxs) == 0):
                est_idxs, est_labels = cc_segmenter.segment(
                    is_harmonic, self.config["nHMMStates"],
                    self.config["nclusters"], self.config["neighbourhoodLimit"],
                    self.anal["sample_rate"], F, in_bound_idxs)
            else:
                est_idxs = in_bound_idxs
                est_labels = [0]

        else:
            # The track is too short. We will only output the first and last
            # time stamps
            est_idxs = np.array([0, F.shape[0] - 1])
            est_labels = [1]

        # Make sure that the first and last boundaries are included
        assert est_idxs[0] == 0  and est_idxs[-1] == F.shape[0] - 1

        # Post process estimations
        est_idxs, est_labels = self._postprocess(est_idxs, est_labels)

        return est_idxs, est_labels
