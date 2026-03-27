"""Base module containing parent classes for the Features."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import jams
import librosa
import numpy as np

from msaf.exceptions import (
    FeatureTypeNotFound,
    NoAudioFileError,
)

# Three types of features:
#   - framesync: Frame-wise synchronous.
#   - est_beatsync: Beat-synchronous using estimated beats with librosa
#   - ann_beatsync: Beat-synchronous using annotated beats from ground-truth
FeatureTypes = Enum("FeatureTypes", "framesync est_beatsync ann_beatsync")

# All available features
features_registry: dict[str, type[Features]] = {}


@dataclass
class ProcessingContext:
    """Holds shared state for a single processing run.

    Separates the computed features object from the scalar algorithm
    parameters so that features are explicitly shared across boundary
    and label algorithms without being buried in a generic dict.
    """

    features: Features | None = None
    hier: bool = False
    algorithm_config: dict[str, Any] = field(default_factory=dict)

    # Convenience proxies so call-sites that used config["features"].X still work
    @property
    def dur(self) -> float | None:
        return self.features.dur if self.features is not None else None


class MetaFeatures(type):
    """Meta-class to register the available features."""

    def __new__(meta, name, bases, class_dict):
        cls = type.__new__(meta, name, bases, class_dict)
        if "Features" in [base.__name__ for base in bases]:
            features_registry[cls.get_id()] = cls
        return cls


class Features(metaclass=MetaFeatures):
    """This is the base class for all the features in MSAF.

    It contains functions to automatically estimate beats, read
    annotated beats, compute beat-synchronous features, and compute
    features on the fly from audio.

    Features are computed **lazily**: only the requested feature type
    (framesync, est_beatsync, or ann_beatsync) is computed.  Beat
    estimation is skipped entirely when only framesync features are
    needed.

    The ``features`` property does the main job, and it returns a matrix
    ``(N, F)``, where ``N`` is the number of frames and ``F`` is the
    number of features per frame.
    """

    def __init__(
        self,
        file_struct: Any,
        sr: int,
        hop_length: int,
        feat_type: FeatureTypes,
    ) -> None:
        self.file_struct = file_struct
        self.sr = sr
        self.hop_length = hop_length
        self.feat_type = feat_type

        self.dur: float | None = None
        self._features: np.ndarray | None = None
        self._framesync_features: np.ndarray | None = None
        self._est_beatsync_features: np.ndarray | None = None
        self._ann_beatsync_features: np.ndarray | None = None
        self._audio: np.ndarray | None = None
        self._audio_harmonic: np.ndarray | None = None
        self._audio_percussive: np.ndarray | None = None
        self._framesync_times: np.ndarray | None = None
        self._est_beatsync_times: np.ndarray | None = None
        self._est_beats_times: np.ndarray | None = None
        self._est_beats_frames: np.ndarray | None = None
        self._ann_beatsync_times: np.ndarray | None = None
        self._ann_beats_times: np.ndarray | None = None
        self._ann_beats_frames: np.ndarray | None = None

    # -- Audio & HPSS ---------------------------------------------------------

    def _load_audio(self) -> None:
        """Load audio from disk if not already loaded."""
        if self._audio is None:
            logging.info("Loading audio: %s", self.file_struct.audio_file)
            self._audio, _ = librosa.load(self.file_struct.audio_file, sr=self.sr)
            self.dur = len(self._audio) / float(self.sr)

    def compute_HPSS(self) -> tuple[np.ndarray, np.ndarray]:
        """Computes harmonic-percussive source separation."""
        return librosa.effects.hpss(self._audio)

    def release_audio(self) -> None:
        """Free audio arrays to reclaim memory after feature computation."""
        self._audio = None
        self._audio_harmonic = None
        self._audio_percussive = None

    # -- Beat estimation ------------------------------------------------------

    def estimate_beats(self) -> tuple[np.ndarray, np.ndarray]:
        """Estimates the beats using librosa.

        Returns
        -------
        times: np.ndarray
            Times of estimated beats in seconds.
        frames: np.ndarray
            Frame indices of estimated beats.
        """
        if self._audio_percussive is None:
            self._audio_harmonic, self._audio_percussive = self.compute_HPSS()

        tempo, frames = librosa.beat.beat_track(
            y=self._audio_percussive, sr=self.sr, hop_length=self.hop_length
        )
        times = librosa.frames_to_time(frames, sr=self.sr, hop_length=self.hop_length)

        if len(times) > 0 and times[0] == 0:
            times = times[1:]
            frames = frames[1:]

        return times, frames

    def read_ann_beats(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Reads the annotated beats if available."""
        times, frames = (None, None)

        if os.path.isfile(self.file_struct.ref_file):
            try:
                jam = jams.load(self.file_struct.ref_file)
            except TypeError:
                logging.warning(
                    "Can't read JAMS file %s. Maybe it's not "
                    "compatible with current JAMS version?",
                    self.file_struct.ref_file,
                )
                return times, frames
            beat_annot = jam.search(namespace="beat.*")

            if len(beat_annot) > 0:
                beats_inters, _ = beat_annot[0].to_interval_values()
                times = beats_inters[:, 0]
                frames = librosa.time_to_frames(
                    times, sr=self.sr, hop_length=self.hop_length
                )
        return times, frames

    # -- Beat-sync helpers ----------------------------------------------------

    def compute_beat_sync_features(
        self,
        beat_frames: np.ndarray | None,
        beat_times: np.ndarray | None,
        pad: bool,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Make the features beat-synchronous."""
        if beat_frames is None:
            return None, None

        beatsync_feats = librosa.util.utils.sync(
            self._framesync_features.T, beat_frames, pad=pad
        ).T

        beatsync_times = np.copy(beat_times)
        if beatsync_times.shape[0] != beatsync_feats.shape[0]:
            beatsync_times = np.concatenate(
                (beatsync_times, [self._framesync_times[-1]])
            )
        return beatsync_feats, beatsync_times

    # -- Core computation (lazy) ----------------------------------------------

    def _compute_framesync_times(self) -> None:
        """Computes the framesync times based on the framesync features."""
        self._framesync_times = librosa.core.frames_to_time(
            np.arange(self._framesync_features.shape[0]),
            sr=self.sr,
            hop_length=self.hop_length,
        )

    def _ensure_framesync(self) -> None:
        """Compute framesync features if not already done."""
        if self._framesync_features is not None:
            return
        self._load_audio()
        logging.info("Computing %s features...", self.get_id())
        # Temporarily set feat_type to framesync for compute_features()
        orig_feat_type = self.feat_type
        self.feat_type = FeatureTypes.framesync
        self._framesync_features = self.compute_features()
        self.feat_type = orig_feat_type
        self._compute_framesync_times()

    def _ensure_est_beatsync(self) -> None:
        """Compute estimated-beat-synchronous features if not already done."""
        if self._est_beatsync_features is not None:
            return
        self._ensure_framesync()
        logging.info("Estimating beats...")
        self._est_beats_times, self._est_beats_frames = self.estimate_beats()
        pad = True
        (
            self._est_beatsync_features,
            self._est_beatsync_times,
        ) = self.compute_beat_sync_features(
            self._est_beats_frames, self._est_beats_times, pad
        )

    def _ensure_ann_beatsync(self) -> None:
        """Compute annotated-beat-synchronous features if not already done."""
        if self._ann_beatsync_features is not None:
            return
        self._ensure_framesync()
        self._ann_beats_times, self._ann_beats_frames = self.read_ann_beats()
        pad = True
        (
            self._ann_beatsync_features,
            self._ann_beatsync_times,
        ) = self.compute_beat_sync_features(
            self._ann_beats_frames, self._ann_beats_times, pad
        )

    # -- Public properties ----------------------------------------------------

    @property
    def frame_times(self) -> np.ndarray | None:
        """Returns the frame times for the corresponding type of features."""
        # Trigger feature computation
        self.features
        if self.feat_type is FeatureTypes.framesync:
            self._compute_framesync_times()
            return self._framesync_times
        elif self.feat_type is FeatureTypes.est_beatsync:
            return self._est_beatsync_times
        elif self.feat_type is FeatureTypes.ann_beatsync:
            return self._ann_beatsync_times
        return None

    @property
    def features(self) -> np.ndarray:
        """Lazily compute and return the requested feature type.

        Returns
        -------
        features: np.ndarray
            The actual features. Each row corresponds to a feature vector.
        """
        if self._features is None:
            try:
                if self.feat_type is FeatureTypes.framesync:
                    self._ensure_framesync()
                elif self.feat_type is FeatureTypes.est_beatsync:
                    self._ensure_est_beatsync()
                elif self.feat_type is FeatureTypes.ann_beatsync:
                    self._ensure_ann_beatsync()
                else:
                    raise FeatureTypeNotFound(
                        "Feature type %s is not valid." % self.feat_type
                    )
            except OSError:
                raise NoAudioFileError(
                    "Couldn't find audio file in %s" % self.file_struct.audio_file
                )

        # Select the right array
        if self.feat_type is FeatureTypes.framesync:
            self._features = self._framesync_features
        elif self.feat_type is FeatureTypes.est_beatsync:
            self._features = self._est_beatsync_features
        elif self.feat_type is FeatureTypes.ann_beatsync:
            if self._ann_beatsync_features is None:
                raise FeatureTypeNotFound(
                    "Feature type %s is not valid because no annotated beats "
                    "were found" % self.feat_type
                )
            self._features = self._ann_beatsync_features
        else:
            raise FeatureTypeNotFound("Feature type %s is not valid." % self.feat_type)

        return self._features

    # -- Factory --------------------------------------------------------------

    @classmethod
    def select_features(
        cls,
        features_id: str | type[Features],
        file_struct: Any,
        annot_beats: bool,
        framesync: bool,
    ) -> Features:
        """Selects the features from the given parameters."""
        if not annot_beats and framesync:
            feat_type = FeatureTypes.framesync
        elif annot_beats and not framesync:
            feat_type = FeatureTypes.ann_beatsync
        elif not annot_beats and not framesync:
            feat_type = FeatureTypes.est_beatsync
        else:
            raise FeatureTypeNotFound("Type of features not valid.")

        if features_id in features_registry:
            feature = features_registry[features_id]
        elif isinstance(features_id, MetaFeatures) and issubclass(
            features_id, Features
        ):
            feature = features_id
        else:
            raise FeatureTypeNotFound(
                "The features '%s' are invalid (valid features are %s)"
                % (features_id, list(features_registry.keys()))
            )

        return feature(file_struct, feat_type)

    # -- Abstract methods -----------------------------------------------------

    def compute_features(self) -> np.ndarray:
        raise NotImplementedError(
            "This method must contain the actual implementation of the features"
        )

    @classmethod
    def get_id(cls) -> str:
        raise NotImplementedError(
            "This method must return a string identifier of the features"
        )
