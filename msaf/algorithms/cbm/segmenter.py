"""
Example of algorithm for MSAF
"""
from msaf.algorithms.interface import SegmenterInterface
import numpy as np

# General module for manipulating data: conversion between time, bars, frame indexes, loading of data, ...
# import as_seg.data_manipulation as dm # Used functions may be replaced by utils apparently
from msaf.utils import intervals_to_times

# Module to process the compute the autosimilarity
# import as_seg.autosimilarity_computation as as_computation
import msaf.algorithms.cbm.autosimilarity_computation as as_comp  # Importing the file instead of the toolbox

# Module containing the CBM algorithm
# import as_seg.CBM_algorithm as cbm
import msaf.algorithms.cbm.CBM_algorithm as CBM  # Importing the file instead of the toolbox


class Segmenter(SegmenterInterface):
    def processFlat(self):
        """Main process.
        Returns
        -------
        est_idxs : np.array(N)
            Estimated indeces the segment boundaries in frame indeces.
        est_labels : np.array(N-1)
            Estimated labels for the segments.
        """
        # Preprocess to obtain features (array(n_frames, n_features))
        F = self._preprocess()

        # Compute the self-similarity matrix
        ssm = as_comp.switch_autosimilarity(
            F, self.config["ssm_type"]
        )  # To test, but should probably be transposed

        # Compute the CBM algorithm
        segments = CBM.compute_cbm(
            ssm,
            max_size=self.config["max_size"],
            penalty_weight=self.config["penalty_weight"],
            penalty_func=self.config["penalty_func"],
            bands_number=self.config["bands_number"],
        )[0]

        # Recast segments into frontiers
        my_bounds = intervals_to_times(np.array(segments))

        # Label the segments (use -1 to have empty segments)
        my_labels = np.ones(len(my_bounds) - 1) * -1

        # Post process estimations
        est_idxs, est_labels = self._postprocess(my_bounds, my_labels)

        # We're done!
        return est_idxs, est_labels
