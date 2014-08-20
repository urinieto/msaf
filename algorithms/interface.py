"""Interface for all the algorithms in MSAF."""
import numpy as np
import msaf.input_output as io
import msaf.utils as U


class SegmenterInterface:
    def process(self, in_path, in_bound_times=None, in_labels=None,
                feature="hpcp", annot_beats=False, framesync=False, **config):
        raise NotImplementedError("This method must be implemented")

    def preprocess(self, in_path, in_bound_times, feature, annot_beats,
                   framesync, valid_features=["hpcp", "tonnetz", "mfcc"]):
        """This method obtains the actual features, their frame times,
        and the boundary indeces in these features if needed."""
        # Read features
        hpcp, mfcc, tonnetz, beats, dur, anal = io.get_features(
            in_path, annot_beats=annot_beats, framesync=framesync)

        # Use correct frames to find times
        frame_times = beats
        if framesync:
            frame_times = U.get_time_frames(dur, anal)

        # Read input bounds if necessary
        bound_idxs = None
        if in_bound_times is not None:
            bound_idxs = io.align_times(in_bound_times, frame_times)

        # Use specific feature
        if feature not in valid_features:
            raise RuntimeError("Feature %s in not valid for algorithm: %s" %
                               (feature, __name__))
        else:
            if feature == "hpcp":
                F = U.lognormalize_chroma(hpcp)  # Normalize chromas
            elif "mfcc":
                F = mfcc
            elif "tonnetz":
                F = U.lognormalize_chroma(tonnetz)  # Normalize tonnetz
            else:
                raise RuntimeError("Feature %s in not valid for algorithm: %s" %
                                   (feature, __name__))

        return F, frame_times, dur, bound_idxs

    def postprocess(self, in_labels, est_times, est_labels):
        """Post processes the estimations from the algorithm, removing empty
        segments and making sure the lenghts of the boundaries and labels
        match."""
        if in_labels is not None:
            est_labels = np.ones(len(est_times) - 1) * -1

        # Remove empty segments if needed
        est_times, est_labels = U.remove_empty_segments(est_times, est_labels)

        assert len(est_times) - 1 == len(est_labels), "Number of boundaries " \
            "(%d) and number of labels(%d) don't match" % (len(est_times),
                                                           len(est_labels))
        return est_times, est_labels
