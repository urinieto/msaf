"""Base module containing parent classes for the Features.

In following versions, base classes for algorithms should also be
included here.
"""

import collections
import datetime
import json
import logging
import os
from enum import Enum

import jams
import librosa
import numpy as np

import msaf
from msaf.exceptions import (
    FeatureParamsError,
    FeaturesNotFound,
    FeatureTypeNotFound,
    NoAudioFileError,
    NoFeaturesFileError,
    WrongFeaturesFormatError,
    FramePerBeatTooHigh,
)

# Five types of features at the moment:
#   - framesync: Frame-wise synchronous.
#   - est_beatsync: Beat-synchronous using estimated beats with librosa
#   - ann_beatsync: Beat-synchronous using annotated beats from ground-truth
#   - est_mutlibeat: Multiple frames per beat-synchronous using estimated beats 
#   - ann_multibeat: Multiple frames per beat-synchronous using annotated beats from ground-truth

FeatureTypes = Enum("FeatureTypes", "framesync est_beatsync ann_beatsync est_multibeat ann_multibeat")

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
    annotated beats, compute beat-synchronous features, read and write
    features.

    It should be straightforward to add features in MSAF, simply by
    writing classes that inherit from this one.

    The `features` getter does the main job, and it returns a matrix
    `(N, F)`, where `N` is the number of frames an `F` is the number of
    features per frames.
    """

    def __init__(self, file_struct, sr, hop_length, feat_type, frames_per_beat=3):
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
        frame_per_beat: int > 0
            Number of frames per beat used in multibeat features
        """
        # Set the global parameters
        self.file_struct = file_struct
        self.sr = sr
        self.hop_length = hop_length
        self.feat_type = feat_type
        self.frames_per_beat = frames_per_beat # The number of frames per beat computed for mfpb features

        # The following attributes will be populated, if needed,
        # once the `features` getter is called
        self.dur = None  # The duration of the audio file in seconds
        self._features = None  # The actual features
        self._framesync_features = None  # Frame-sync features
        self._est_beatsync_features = None  # Estimated Beat-sync features
        self._ann_beatsync_features = None  # Annotated Beat-sync features
        self._est_multibeat_features = None  # Estimated Beat-sync features with multiple frames per beat
        self._ann_mutlibeat_features = None  # Annotated Beat-sync features with multiple frames per beat
        self._audio = None  # Actual audio signal
        self._audio_harmonic = None  # Harmonic audio signal
        self._audio_percussive = None  # Percussive audio signal
        self._framesync_times = None  # The times of the framesync features
        self._est_beatsync_times = None  # Estimated beat-sync times
        self._est_beats_times = None  # Estimated beat times
        self._est_beats_frames = None  # Estimated beats in frames
        self._ann_beatsync_times = None  # Annotated beat-sync times
        self._ann_beats_times = None  # Annotated beat times
        self._ann_beats_frames = None  # Annotated beats in frames
        self._est_multibeat_frames = None # Estimated multibeat frames
        self._est_multibeat_times = None # Estimated multibeat times
        self._ann_multibeat_frames = None # Annotated multibeat frames
        self._ann_multibeat_times = None # Annotated multibeat times

        # Differentiate global params from subclass attributes.
        # This is a bit hacky... I accept Pull Requests ^_^
        self._global_param_names = [
            "file_struct",
            "sr",
            "feat_type",
            "hop_length",
            "dur",
            "frames_per_beat",
        ]

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
        # Compute harmonic-percussive source separation if needed
        if self._audio_percussive is None:
            self._audio_harmonic, self._audio_percussive = self.compute_HPSS()

        # Compute beats
        tempo, frames = librosa.beat.beat_track(
            y=self._audio_percussive, sr=self.sr, hop_length=self.hop_length
        )

        # To times
        times = librosa.frames_to_time(frames, sr=self.sr, hop_length=self.hop_length)

        # TODO: Is this really necessary?
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

        # Read annotations if they exist in correct folder
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

            # If beat annotations exist, get times and frames
            if len(beat_annot) > 0:
                beats_inters, _ = beat_annot[0].to_interval_values()
                times = beats_inters[:, 0]
                frames = librosa.time_to_frames(
                    times, sr=self.sr, hop_length=self.hop_length
                )
        return times, frames

    def compute_beat_sync_features(self, beat_frames, pad):
        """Make the features beat-synchronous.

        Parameters
        ----------
        beat_frames: np.array
            The frame indices of the beat positions.
        pad: boolean
            If `True`, `beat_frames` is padded to span the full range.

        Returns
        -------
        beatsync_feats: np.array
            The beat-synchronized features.
            `None` if the beat_frames was `None`.
        """
        if beat_frames is None:
            return None
        
        # Make beat synchronous
        beatsync_feats = librosa.util.utils.sync(
            self._framesync_features.T, beat_frames, pad=pad
        ).T

        return beatsync_feats
    
    def _compute_multibeat(self, beat_frames):
        """Compute frames index  evenly distributed between beats
        Parameters
        ----------
            beat_frames: np.array
                the frames index of beats
        Returns
        -------
            multibeat_frames: np.array
                the frames index of multibeats
        """
        if beat_frames is None:
            return None
        if beat_frames == []:
            return []
        multibeat_frames = np.empty(0,dtype=int)
        for idx in range(len(beat_frames)-1):
            this_beat_frame = beat_frames[idx]
            next_beat_frame = beat_frames[idx+1]
            subdivision = (next_beat_frame - this_beat_frame)
            if not (self.frames_per_beat < subdivision):
                raise FramePerBeatTooHigh
            frames_in_beat = [int(k * subdivision/self.frames_per_beat + this_beat_frame) for k in range(self.frames_per_beat)]
            multibeat_frames = np.concatenate((multibeat_frames, frames_in_beat), dtype=int )

        return multibeat_frames

    def _shape_beatwise(self, multibeat_features):
        """Transform the multibeat_features matrix into a beatwise features matrix
        Parameters
        -----------
        multibeat_features: np.array
            The features to transform
        Returns
        ----------
        beatwise_feature: np.array
            The features shaped as a beatwise matrix
            or None if multibeat_features is None
        """
        if multibeat_features is None:
            return None
        if multibeat_features.shape[0] == 0:
            return multibeat_features
        assert(multibeat_features.shape[0]%self.frames_per_beat == 0,"The size of array must be a multiple of self.frames_per_beat")
        nummber_of_beats = int(multibeat_features.shape[0]/self.frames_per_beat)
        tensor = []
        for k in range(nummber_of_beats):
            beat=[]
            for f in range(self.frames_per_beat):
                beat.append(multibeat_features[k*self.frames_per_beat + f])
            tensor.append(beat)
        tensor = np.array(tensor)
        return np.reshape(tensor,(tensor.shape[0], -1), order='C')
    
    def _pad_beats_times(self, beat_times, beatsync_feats):
        """Pad the beat_times with the last frametimes if necessary
        Parameters
        -----------
            beat_times: np.array
                the beats times to pad
            beatsync_features: np.array
                The features corresponding to the beats
        Returns
        ---------
            beatsync_times: np.array
                the padded beats times
        """
        if beatsync_feats is None :
            return None
        beatsync_times = np.copy(beat_times)
        if beatsync_times.shape[0] != beatsync_feats.shape[0]:
            beatsync_times = np.concatenate(
                (beatsync_times, [self._framesync_times[-1]])
            )
        return beatsync_times
    
    def read_features(self, tol=1e-3):
        """Reads the features from a file and stores them in the current
        object.

        Parameters
        ----------
        tol: float
            Tolerance level to detect duration of audio.
        """
        try:
            # Read JSON file
            with open(self.file_struct.features_file) as f:
                feats = json.load(f)

            # Store duration
            if self.dur is None:
                self.dur = float(feats["globals"]["dur"])

            # Check that we have the correct global parameters
            assert np.isclose(self.dur, float(feats["globals"]["dur"]), rtol=tol)
            assert self.sr == int(feats["globals"]["sample_rate"])
            assert self.hop_length == int(feats["globals"]["hop_length"])
            assert os.path.basename(self.file_struct.audio_file) == os.path.basename(
                feats["globals"]["audio_file"]
            )
            assert self.frames_per_beat == int(feats["globals"]["frames_per_beat"])

            # Check for specific features params
            feat_params_err = FeatureParamsError(
                "Couldn't find features for %s id in file %s"
                % (self.get_id(), self.file_struct.features_file)
            )
            if self.get_id() not in feats.keys():
                raise feat_params_err
            for param_name in self.get_param_names():
                value = getattr(self, param_name)
                if hasattr(value, "__call__"):
                    # Special case of functions
                    if value.__name__ != feats[self.get_id()]["params"][param_name]:
                        raise feat_params_err
                else:
                    if str(value) != feats[self.get_id()]["params"][param_name]:
                        raise feat_params_err

            # Store actual features
            self._est_beats_times = np.array(feats["est_beats"])
            self._est_beatsync_times = np.array(feats["est_beatsync_times"])
            self._est_beats_frames = librosa.core.time_to_frames(
                self._est_beats_times, sr=self.sr, hop_length=self.hop_length
            )
            self._framesync_features = np.array(feats[self.get_id()]["framesync"])
            self._est_beatsync_features = np.array(feats[self.get_id()]["est_beatsync"])

            self._est_multibeat_times = np.array(feats[self.get_id()]["est_multibeat_times"])
            self._est_multibeat_features = np.array(feats[self.get_id()]["est_multibeat"])

            # Read annotated beats if available
            if "ann_beats" in feats.keys():
                self._ann_beats_times = np.array(feats["ann_beats"])
                self._ann_beatsync_times = np.array(feats["ann_beatsync_times"])
                self._ann_multibeat_times = np.array(feats["ann_multibeat_times"])
                self._ann_beats_frames = librosa.core.time_to_frames(
                    self._ann_beats_times, sr=self.sr, hop_length=self.hop_length
                )
                self._ann_beatsync_features = np.array(
                    feats[self.get_id()]["ann_beatsync"]
                )
                self._ann_mutlibeat_features = np.array(
                    feats[self.get_id()]["ann_multibeat"]
                )
        except KeyError:
            raise WrongFeaturesFormatError(
                "The features file %s is not correctly formatted"
                % self.file_struct.features_file
            )
        except AssertionError:
            raise FeaturesNotFound(
                "The features for the given parameters were not found in "
                "features file %s" % self.file_struct.features_file
            )
        except OSError:
            raise NoFeaturesFileError(
                "Could not find features file %s", self.file_struct.features_file
            )

    def write_features(self):
        """Saves features to file."""
        out_json = collections.OrderedDict()
        try:
            # Only save the necessary information
            self.read_features()
        except (WrongFeaturesFormatError, FeaturesNotFound, NoFeaturesFileError):
            # We need to create the file or overwrite it
            # Metadata
            out_json = collections.OrderedDict(
                {
                    "metadata": {
                        "versions": {
                            "librosa": librosa.__version__,
                            "msaf": msaf.__version__,
                            "numpy": np.__version__,
                        },
                        "timestamp": datetime.datetime.today().strftime(
                            "%Y/%m/%d %H:%M:%S"
                        ),
                    }
                }
            )

            # Global parameters
            out_json["globals"] = {
                "dur": self.dur,
                "sample_rate": self.sr,
                "hop_length": self.hop_length,
                "audio_file": self.file_struct.audio_file,
                "frames_per_beat": self.frames_per_beat,
            }

            # Beats
            out_json["est_beats"] = self._est_beats_times.tolist()
            out_json["est_beatsync_times"] = self._est_beatsync_times.tolist()
            out_json["est_multibeat_times"] = self._est_multibeat_times.tolist()
            if self._ann_beats_times is not None:
                out_json["ann_beats"] = self._ann_beats_times.tolist()
                out_json["ann_beatsync_times"] = self._ann_beatsync_times.tolist()
                out_json["ann_multibeat_times"] = self._ann_multibeat_times.tolist()
        except FeatureParamsError:
            # We have other features in the file, simply add these ones
            with open(self.file_struct.features_file) as f:
                out_json = json.load(f)
        finally:
            # Specific parameters of the current features
            out_json[self.get_id()] = {}
            out_json[self.get_id()]["params"] = {}
            for param_name in self.get_param_names():
                value = getattr(self, param_name)
                # Check for special case of functions
                if hasattr(value, "__call__"):
                    value = value.__name__
                else:
                    value = str(value)
                out_json[self.get_id()]["params"][param_name] = value

            # Actual features
            out_json[self.get_id()]["framesync"] = self._framesync_features.tolist()
            out_json[self.get_id()]["est_beatsync"] = self._est_beatsync_features.tolist()

            if self._ann_beatsync_features is not None:
                out_json[self.get_id()]["ann_beatsync"] = self._ann_beatsync_features.tolist()
            
            out_json[self.get_id()]["est_multibeat"] = self._est_multibeat_features.tolist()
            if self._ann_mutlibeat_features is not None:
                out_json[self.get_id()]["ann_multibeat"] = self._ann_mutlibeat_features.tolist()
            # Save it
            with open(self.file_struct.features_file, "w") as f:
                json.dump(out_json, f, indent=2)

    def get_param_names(self):
        """Returns the parameter names for these features, avoiding the global
        parameters."""
        return [
            name
            for name in vars(self)
            if not name.startswith("_") and name not in self._global_param_names
        ]

    def _compute_framesync_times(self):
        """Computes the framesync times based on the framesync features."""
        self._framesync_times = librosa.core.frames_to_time(
            np.arange(self._framesync_features.shape[0]),
            sr=self.sr,
            hop_length=self.hop_length,
        )

    def _compute_all_features(self):
        """Computes all the features (beatsync, framesync) from the audio."""
        # Read actual audio waveform
        self._audio, _ = librosa.load(self.file_struct.audio_file, sr=self.sr)

        # Get duration of audio file
        self.dur = len(self._audio) / float(self.sr)

        # Compute actual features
        feat_type = self.feat_type
        self.feat_type = FeatureTypes.framesync
        self._framesync_features = self.compute_features()
        self.feat_type = feat_type

        # Compute framesync times
        self._compute_framesync_times()

        # Compute/Read beats times and frames
        self._est_beats_times, self._est_beats_frames = self.estimate_beats()
        self._ann_beats_times, self._ann_beats_frames = self.read_ann_beats()

        # Compute multibeats timees and frames
        self._est_multibeat_frames = self._compute_multibeat(self._est_beats_frames)
        self._ann_multibeat_frames = self._compute_multibeat(self._ann_beats_frames)

        # Multibeat times is beats time (before padding)
        self._est_multibeat_times = np.copy(self._est_beats_times)
        self._ann_multibeat_times = np.copy(self._ann_beats_times)

        # Compute frames features on beat
        pad = True  # pad the beat frames 
        self._est_beatsync_features = self.compute_beat_sync_features(self._est_beats_frames, pad)
        self._ann_beatsync_features = self.compute_beat_sync_features(self._ann_beats_frames, pad)

        # Compute frames features on multibeat
        self._est_multibeat_features = self.compute_beat_sync_features(self._est_multibeat_frames, pad)
        self._ann_mutlibeat_features = self.compute_beat_sync_features(self._ann_multibeat_frames, pad)

        # Transform multibeat into beatwise matrix
        self._est_multibeat_features = self._shape_beatwise(self._est_multibeat_features)
        self._ann_mutlibeat_features = self._shape_beatwise(self._ann_mutlibeat_features)

        # Pad beatsync times
        self._est_beatsync_times = self._pad_beats_times(self._est_beats_times, self._est_beatsync_features)
        self._ann_beatsync_times = self._pad_beats_times(self._ann_beats_times, self._ann_beatsync_features)

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
        elif self.feat_type is FeatureTypes.est_multibeat:
            frame_times = self._est_multibeat_times
        elif self.feat_type is FeatureTypes.ann_multibeat:
            frame_times = self._ann_multibeat_times
        else:
            raise FeatureTypeNotFound("Type of features not valid.")
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
        # Compute features if needed
        if self._features is None:
            try:
                self.read_features()
            except (
                NoFeaturesFileError,
                FeaturesNotFound,
                WrongFeaturesFormatError,
                FeatureParamsError,
            ) as e:
                try:
                    self._compute_all_features()
                    self.write_features()
                except OSError:
                    if isinstance(e, FeaturesNotFound) or isinstance(
                        e, FeatureParamsError
                    ):
                        msg = (
                            "Computation of the features is needed for "
                            "current parameters but no audio file was found."
                            "Please, change your parameters or add the audio"
                            " file in %s"
                        )
                    else:
                        msg = "Couldn't find audio file in %s"
                    raise NoAudioFileError(msg % self.file_struct.audio_file)

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
        elif self.feat_type is FeatureTypes.est_multibeat:
            self._features = self._est_multibeat_features
        elif self.feat_type is FeatureTypes.ann_multibeat:
            if self._ann_multibeat_features is None:
                raise FeatureTypeNotFound(
                    "Feature type %s is not valid because no annotated beats "
                    "were found" % self.feat_type
                )
            self._features = self._ann_mutlibeat_features
        else:
            raise FeatureTypeNotFound("Feature type %s is not valid." % self.feat_type)

        return self._features

    @classmethod
    def select_features(cls, features_id, file_struct, annot_beats, framesync, multibeat=False):
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
            if multibeat:
                feat_type = FeatureTypes.ann_multibeat
            else:
                feat_type = FeatureTypes.ann_beatsync
        elif not annot_beats and not framesync:
            if multibeat:
                feat_type = FeatureTypes.est_multibeat
            else:
                feat_type = FeatureTypes.est_beatsync
        else:
            raise FeatureTypeNotFound("Type of features not valid.")

        # Select features with default parameters
        if features_id in features_registry.keys():
            feature = features_registry[features_id]
        elif isinstance(features_id, MetaFeatures) and issubclass(
            features_id, Features
        ):
            feature = features_id
        else:
            raise FeaturesNotFound(
                "The features '%s' are invalid (valid features are %s)"
                % (features_id, features_registry.keys())
            )

        return feature(file_struct, feat_type)

    def compute_features(self):
        raise NotImplementedError(
            "This method must contain the actual " "implementation of the features"
        )

    @classmethod
    def get_id(cls):
        raise NotImplementedError(
            "This method must return a string identifier" " of the features"
        )
