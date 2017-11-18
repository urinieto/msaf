"""
Variable Markov Oracle algorithm
"""
from msaf.algorithms.interface import SegmenterInterface
import librosa
import main


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

        my_bounds, my_labels = main.scluster_segment(F, self.config, self.in_bound_idxs)

        # my_bounds, my_labels = segmentation(oracle,
        #                                     method=self.config['method'],
        #                                     connectivity=self.config['connectivity'])
        # Post process estimations
        est_idxs, est_labels = self._postprocess(my_bounds, my_labels)

        assert est_idxs[0] == 0 and est_idxs[-1] == F.shape[0] - 1
        # We're done!
        return est_idxs, est_labels
