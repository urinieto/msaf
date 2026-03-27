"""Base module containing parent classes for the Features."""

import logging
import os
from enum import Enum

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
features_registry = {}


class MetaFeatures(type):
    """Meta-class to register the available features."""

    def __new__(meta, name, bases, class_dict):
        cls = type.__new__(meta, name, bases, class_dict)
        # Register classes that inherit from the base class Features
        if "Features" in [base.__name__ for base in bases]:
            features_registry[cls.get_id()] = cls
        return cls


class Features(metaclass=MetaFeatures):
    """This is the base class for all the features in MSAF.

    It contains functions to automatically estimate beats, read
    annotated beats, compute beat-synchronous features, and compute
    features on the fly from audio.

    It should be straightforward to add features in MSAF, simply by
    writing classes that inherit from this one.

    The `features` getter does the main job, and it returns a matrix
    `(N, F)`, where `N` is the number of frames an `F` is the number of
    features per frames.
    """

    def __init__(self, file_struct, sr, hop_length, feat_type):
        """Init function for the base class to make sure all features have at
        least these parameters as attributes.

        Parameters
        ----------
        file_struct: `msaf.input_output.FileStruct`
            Object containing the paths to the files.
        sr: int > 0
            Sampling rate of the audio file.
        hop_length: int > 0
            Hop in frames of the features to be computed.
        feat_type: `FeatureTypes`
            Enum containing the type of feature.
        """
        self.file_struct = file_struct
        self.sr = sr
        self.hop_length = hop_length
        self.feat_type = feat_type

        # The following attributes will be populated, if needed,
        # once the `features` getter is called
        self.dur = None
        self._features = None
        self._framesync_features = None
        self._est_beatsync_features = None
        self._ann_beatsync_features = None
        self._audio = None
        self._audio_harmonic = None
        self._audio_percussive = None
        self._framesync_times = None
        self._est_beatsync_times = None
        self._est_beats_times = None
        self._est_beats_frames = None
        self._ann_beatsync_times = None
        self._ann_beats_times = None
        self._ann_beats_frames = None

    def compute_HPSS(self):
        """Computes harmonic-percussive source separation.

        Returns
        -------
        audio_harmonic: np.array
            The harmonic component of the audio signal
        audio_percussive: np.array
            The percussive component of the audio signal
        """
        return librosa.effects.hpss(self._audio)

    def estimate_beats(self):
        """Estimates the beats using librosa.

        Returns
        -------
        times: np.array
            Times of estimated beats in seconds.
        frames: np.array
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

    def read_ann_beats(self):
        """Reads the annotated beats if available.

        Returns
        -------
        times: np.array
            Times of annotated beats in seconds.
        frames: np.array
            Frame indices of annotated beats.
        """
        times, frames = (None, None)

        if os.path.isfile(self.file_struct.ref_file):
            try:
                jam = jams.load(self.file_struct.ref_file)
            except TypeError:
                logging.warning(
                    "Can't read JAMS file %s. Maybe it's not "
                    "compatible with current JAMS version?" % self.file_struct.ref_file
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

    def compute_beat_sync_features(self, beat_frames, beat_times, pad):
        """Make the features beat-synchronous.

        Parameters
        ----------
        beat_frames: np.array
            The frame indices of the beat positions.
        beat_times: np.array
            The time points of the beat positions (in seconds).
        pad: boolean
            If `True`, `beat_frames` is padded to span the full range.

        Returns
        -------
        beatsync_feats: np.array
            The beat-synchronized features.
            `None` if the beat_frames was `None`.
        beatsync_times: np.array
            The beat-synchronized times.
            `None` if the beat_frames was `None`.
        """
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

    def _compute_framesync_times(self):
        """Computes the framesync times based on the framesync features."""
        self._framesync_times = librosa.core.frames_to_time(
            np.arange(self._framesync_features.shape[0]),
            sr=self.sr,
            hop_length=self.hop_length,
        )

    def _compute_all_features(self):
        """Computes all the features (beatsync, framesync) from the audio."""
        logging.info("Loading audio: %s", self.file_struct.audio_file)
        self._audio, _ = librosa.load(self.file_struct.audio_file, sr=self.sr)

        self.dur = len(self._audio) / float(self.sr)

        logging.info("Computing %s features...", self.get_id())
        feat_type = self.feat_type
        self.feat_type = FeatureTypes.framesync
        self._framesync_features = self.compute_features()
        self.feat_type = feat_type

        self._compute_framesync_times()

        logging.info("Estimating beats...")
        self._est_beats_times, self._est_beats_frames = self.estimate_beats()
        self._ann_beats_times, self._ann_beats_frames = self.read_ann_beats()

        # Beat-Synchronize
        pad = True
        (
            self._est_beatsync_features,
            self._est_beatsync_times,
        ) = self.compute_beat_sync_features(
            self._est_beats_frames, self._est_beats_times, pad
        )
        (
            self._ann_beatsync_features,
            self._ann_beatsync_times,
        ) = self.compute_beat_sync_features(
            self._ann_beats_frames, self._ann_beats_times, pad
        )

    @property
    def frame_times(self):
        """This getter returns the frame times, for the corresponding type of
        features."""
        frame_times = None
        # Make sure we have already computed the features
        self.features
        if self.feat_type is FeatureTypes.framesync:
            self._compute_framesync_times()
            frame_times = self._framesync_times
        elif self.feat_type is FeatureTypes.est_beatsync:
            frame_times = self._est_beatsync_times
        elif self.feat_type is FeatureTypes.ann_beatsync:
            frame_times = self._ann_beatsync_times

        return frame_times

    @property
    def features(self):
        """This getter will compute the actual features if they haven't been
        computed yet.

        Returns
        -------
        features: np.array
            The actual features. Each row corresponds to a feature vector.
        """
        if self._features is None:
            try:
                self._compute_all_features()
            except OSError:
                raise NoAudioFileError(
                    "Couldn't find audio file in %s" % self.file_struct.audio_file
                )

        # Choose features based on type
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

    @classmethod
    def select_features(cls, features_id, file_struct, annot_beats, framesync):
        """Selects the features from the given parameters.

        Parameters
        ----------
        features_id: str or `msaf.features.Features` class
            The identifier of the features (it must be a key inside the
            `features_registry`)
        file_struct: msaf.io.FileStruct
            The file struct containing the files to extract the features from
        annot_beats: boolean
            Whether to use annotated (`True`) or estimated (`False`) beats
        framesync: boolean
            Whether to use framesync (`True`) or beatsync (`False`) features

        Returns
        -------
        features: obj
            The actual features object that inherits from `msaf.Features`
        """
        if not annot_beats and framesync:
            feat_type = FeatureTypes.framesync
        elif annot_beats and not framesync:
            feat_type = FeatureTypes.ann_beatsync
        elif not annot_beats and not framesync:
            feat_type = FeatureTypes.est_beatsync
        else:
            raise FeatureTypeNotFound("Type of features not valid.")

        if features_id in features_registry.keys():
            feature = features_registry[features_id]
        elif isinstance(features_id, MetaFeatures) and issubclass(
            features_id, Features
        ):
            feature = features_id
        else:
            raise FeatureTypeNotFound(
                "The features '%s' are invalid (valid features are %s)"
                % (features_id, features_registry.keys())
            )

        return feature(file_struct, feat_type)

    def compute_features(self):
        raise NotImplementedError(
            "This method must contain the actual implementation of the features"
        )

    @classmethod
    def get_id(cls):
        raise NotImplementedError(
            "This method must return a string identifier of the features"
        )
