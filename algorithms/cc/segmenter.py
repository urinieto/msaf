#!/usr/bin/env python
# coding: utf-8
"""
This script segments a given track using the Constrained Clustering method
described here:

Levy, M., & Sandler, M. (2008). Structural Segmentation of Musical Audio by
Constrained Clustering. IEEE Transactions on Audio, Speech, and Language
Processing, 16(2), 318–326. doi:10.1109/TASL.2007.910781
"""

__author__ = "Oriol Nieto"
__copyright__ = "Copyright 2014, Music and Audio Research Lab (MARL)"
__license__ = "GPL"
__version__ = "1.0"
__email__ = "oriol@nyu.edu"

import logging
import numpy as np

# Local stuff
from msaf.algorithms.interface import SegmenterInterface
try:
    import cc_segmenter
except:
    logging.warning("You must compile the Constrained Clustering method (cc) "
                    "before you can use it. To do so, go to msaf/algorithms/cc "
                    "and type:\n\tpython setup.py build_ext --inplace")


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
        # CC params
        min_dur = 15     # Minimum duration of the track in seconds

        # Preprocess to obtain features, times, and input boundary indeces
        F, frame_times, dur, in_bound_idxs = self._preprocess(normalize=True)
        #import pylab as plt
        #plt.imshow(F.T, interpolation="nearest", aspect="auto")
        #plt.show()

        if dur >= min_dur:
            if self.feature_str == "hpcp" or self.feature_str == "tonnetz":
                is_harmonic = True
            elif self.feature_str == "mfcc":
                is_harmonic = False
            else:
                raise RuntimeError("Feature type %s is not valid" %
                                   self.feature_str)

            if in_bound_idxs is None:
                in_bound_idxs = []

            bound_idxs, est_labels = cc_segmenter.segment(
                is_harmonic, self.config["nHMMStates"],
                self.config["nclusters"], self.config["neighbourhoodLimit"],
                self.anal["sample_rate"], F, in_bound_idxs)

            # Add first and last boundaries
            est_times = np.concatenate(([0], frame_times[bound_idxs], [dur]))
            silencelabel = np.max(est_labels) + 1
            est_labels = np.concatenate(([silencelabel], est_labels,
                                         [silencelabel]))
        else:
            # The track is too short. We will only output the first and last
            # time stamps
            est_times = np.array([0, dur])
            est_labels = [1]

        # Post process estimations
        est_times, est_labels = self._postprocess(est_times, est_labels)

        logging.info("Estimated times: %s" % est_times)
        logging.info("Estimated labels: %s" % est_labels)

        return est_times, est_labels
