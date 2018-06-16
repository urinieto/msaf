"""
Variable Markov Oracle algorithm
"""
from msaf.algorithms.interface import SegmenterInterface
import librosa
from . import main


class Segmenter(SegmenterInterface):
    """
    This script identifies the structure of a given track using the Variable
    Markov Oracle technique described here:

    Wang, C., Mysore, J. G., Structural Segmentation With The Variable Markov
    Oracle And Boundary Adjustment. Proc. of the 41st IEEE International
    Conference on Acoustics, Speech, and Signal Processing (ICASSP).
    Shanghai, China, 2016 (`PDF`_).

    .. _PDF: https://ccrma.stanford.edu/~gautham/Site/Publications_files/segmentation-icassp_2016.pdf
    """
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
        F = librosa.util.normalize(F, axis=0)
        F = librosa.feature.stack_memory(F.T).T

        self.config["hier"] = False
        my_bounds, my_labels, _ = main.scluster_segment(F, self.config, self.in_bound_idxs)

        # Post process estimations
        est_idxs, est_labels = self._postprocess(my_bounds, my_labels)

        assert est_idxs[0] == 0 and est_idxs[-1] == F.shape[0] - 1
        # We're done!
        return est_idxs, est_labels

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
        F = self._preprocess()
        F = librosa.util.normalize(F, axis=0)
        F = librosa.feature.stack_memory(F.T).T

        self.config["hier"] = True
        est_idxs, est_labels, F = main.scluster_segment(F, self.config, self.in_bound_idxs)
        for layer in range(len(est_idxs)):
            assert est_idxs[layer][0] == 0 and \
                est_idxs[layer][-1] == F.shape[1] - 1
            est_idxs[layer], est_labels[layer] = \
                self._postprocess(est_idxs[layer], est_labels[layer])
        return est_idxs, est_labels