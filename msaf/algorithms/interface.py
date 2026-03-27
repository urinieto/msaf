"""Interface for all the algorithms in MSAF."""
from __future__ import annotations

from typing import Any

import numpy as np

import msaf.utils as U


class SegmenterInterface:
    """This class is an interface for all the segmenter algorithms included in
    MSAF.

    These segmenters must inherit from it and implement one of the following
    methods:
        - processFlat()
        - processHierarchical()

    Additionally, two private helper functions are provided:
        - preprocess
        - postprocess

    These are meant to do common tasks for all the segmenters and they
    should be called inside the process method if needed.

    All segmenters must return estimates times for the boundaries (est_times),
    and estimated labels (est_labels), **even if they can't compute them**.

    The following three types of algorithms with their behaviors are:
        - Computes boundaries and labels:
            If in_bound_times is None:
                Compute the est_times
            Else:
                Do not compute est_times, simply use in_bound_times instead
            If in_labels is None:
                Compute the est_labels
            Else:
                Do not compute est_labels, simply use in_labels instead

        - Computes boundaries only:
            Compute boundaries and return est_labels as None.

        - Computes labels only:
            Use in_bound_times in order to compute the labels.
            Return est_times as in_bound_times and the computed labels.

    In these cases, est_times or est_labels will be empty (None).
    """

    def __init__(
        self,
        file_struct: Any,
        in_bound_idxs: np.ndarray | None = None,
        feature: str = "pcp",
        annot_beats: bool = False,
        framesync: bool = False,
        features: Any | None = None,
        **config: Any
    ) -> None:
        """Inits the Segmenter.

        Parameters
        ----------
        file_struct: `msaf.io.FileStruct`
            Object with the file paths.
        in_bound_idxs: np.array
            Array containing the frame indices of the previously find
            boundaries. `None` for computing them.
        feature: str
            Identifier of the features (e.g., pcp, mfcc)
        annot_beats: boolean
            Whether to use annotated beats or estimated ones.
        framesync: boolean
            Whether to use frame-synchronous or beat-synchronous features.
        features: dict
            Previously computed features. `None` for reading them.
        config: dict
            Configuration for the given algorithm (see module's __config.py__).
        """
        self.file_struct = file_struct
        self.audio_file = file_struct.audio_file
        self.in_bound_idxs = in_bound_idxs
        self.feature_str = feature
        self.annot_beats = annot_beats
        self.framesync = framesync
        self.config = config
        self.features = features

    def processFlat(self) -> tuple[np.ndarray, np.ndarray]:
        """Main process to obtain the flat segmentation of a given track."""
        raise NotImplementedError("This method does not return flat " "segmentations.")

    def processHierarchical(self) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Main process to obtain the hierarchical segmentation of a given
        track."""
        raise NotImplementedError(
            "This method does not return hierarchical " "segmentations."
        )

    def _preprocess(self) -> np.ndarray:
        """This method obtains the actual features."""
        try:
            F = self.features.features
        except KeyError:
            raise RuntimeError(
                "Feature %s in not supported by MSAF" % (self.feature_str)
            )

        return F

    def _postprocess(
        self, est_idxs: np.ndarray, est_labels: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Post processes the estimations from the algorithm, removing empty
        segments and making sure the lengths of the boundaries and labels
        match."""
        # Make sure we are using the previously input bounds, if any
        if self.in_bound_idxs is not None:
            F = self._preprocess()
            est_labels = U.synchronize_labels(
                self.in_bound_idxs, est_idxs, est_labels, F.shape[0]
            )
            est_idxs = self.in_bound_idxs

        # Remove empty segments if needed
        est_idxs, est_labels = U.remove_empty_segments(est_idxs, est_labels)

        assert len(est_idxs) - 1 == len(
            est_labels
        ), "Number of boundaries " "(%d) and number of labels(%d) don't match" % (
            len(est_idxs),
            len(est_labels),
        )

        # Make sure the indices are integers
        est_idxs = np.asarray(est_idxs, dtype=int)

        return est_idxs, est_labels
