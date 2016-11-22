"""
Example of algorithm for MSAF
"""
from msaf.algorithms.interface import SegmenterInterface
from vmo.analysis.segmentation import segmentation
import vmo
import librosa


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

        F = librosa.feature.stack_memory(F.T).T
        ideal_t = vmo.find_threshold(F, dim=F.shape[1])
        oracle = vmo.build_oracle(F,flag='a', threshold=ideal_t[0][1], dim=F.shape[1])

        my_bounds, my_labels = segmentation(oracle, method=self.config['method'],
                                            connectivity=self.config['connectivity'])
        # Post process estimations
        est_idxs, est_labels = self._postprocess(my_bounds, my_labels)
        # We're done!
        return est_idxs, est_labels
