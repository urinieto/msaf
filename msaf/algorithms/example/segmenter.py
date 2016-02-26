"""
Example of algorithm for MSAF
"""
from msaf.algorithms.interface import SegmenterInterface
import numpy as np


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

        # Do something with the default parameters
        # (these are defined in the in the config.py file).
        assert self.config["my_param1"] == 1.0

        # Identify boundaries in frame indeces with the new algorithm
        my_bounds = np.array([0, F.shape[0] - 1])

        # Label the segments (use -1 to have empty segments)
        my_labels = np.ones(len(my_bounds) - 1) * -1

        # Post process estimations
        est_idxs, est_labels = self._postprocess(my_bounds, my_labels)

        # We're done!
        return est_idxs, est_labels
