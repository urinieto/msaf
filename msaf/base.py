"""
Base module containing parent classes for the Features.
In following versions, base classes for algorithms should also be included
here.
"""

import collections
import datetime
from enum import Enum
import librosa
import jams
import json
import numpy as np
import os
import six

# Local stuff
import msaf
from msaf.exceptions import WrongFeaturesFormatError, NoFeaturesFileError,\
    FeaturesNotFound, FeatureTypeNotFound, FeatureParamsError, NoAudioFileError


# Three types of features at the moment:
#   - framesync: Frame-wise synchronous.
#   - est_beatsync: Beat-synchronous using estimated beats with librosa
#   - ann_beatsync: Beat-synchronous using annotated beats from ground-truth
FeatureTypes = Enum('FeatureTypes', 'framesync est_beatsync ann_beatsync')

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


class Features(six.with_metaclass(MetaFeatures)):
    """This is the base class for all the features in MSAF.

    It contains functions to automatically estimate beats, read annotated
    beats, compute beat-synchronous features, read and write features.

    It should be straightforward to add features in MSAF, simply by writing
    classes that inherit from this one.

    The `features` getter does the main job, and it returns a matrix `(N, F)`,
    where `N` is the number of frames an `F` is the number of features
    per frames.
    """
    def __init__(self, file_struct, sr, hop_length, feat_type):
        """Init function for the base class to make sure all features have
        at least parameters as attributes.

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
        # Set the global parameters
        self.file_struct = file_struct
        self.sr = sr
        self.hop_length = hop_length
        self.feat_type = feat_type

        # The following attributes will be populated, if needed,
        # once the `features` getter is called
        self.dur = None  # The duration of the audio file in seconds
        self._features = None  # The actual features
        self._framesync_features = None  # Frame-sync features
        self._est_beatsync_features = None  # Estimated Beat-sync features
        self._ann_beatsync_features = None  # Annotated Beat-sync features
        self._audio = None  # Actual audio signal
        self._audio_harmonic = None  # Harmonic audio signal
        self._audio_percussive = None  # Percussive audio signal
        self._framesync_times = None  # The frame synced times
        self._est_beats_times = None  # Estimated beat times
        self._est_beats_frames = None  # Estimated beats in frames
        self._ann_beats_times = None  # Annotated beat times
        self._ann_beats_frames = None  # Annotated beats in frames

        # Differentiate global params from sublcass attributes.
        # This is a bit hacky... I accept Pull Requests ^_^
        self._global_param_names = ["file_struct", "sr", "feat_type",
                                    "hop_length", "dur"]

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
            Frame indeces of estimated beats.
        """
        # Compute harmonic-percussive source separiation if needed
        if self._audio_percussive is None:
            self._audio_harmonic, self._audio_percussive = self.compute_HPSS()

        # Compute beats
        tempo, frames = librosa.beat.beat_track(
            y=self._audio_percussive, sr=self.sr,
            hop_length=self.hop_length)

        # To times
        times = librosa.frames_to_time(frames, sr=self.sr,
                                       hop_length=self.hop_length)

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
            Frame indeces of annotated beats.
        """
        times, frames = (None, None)

        # Read annotations if they exist in correct folder
        if os.path.isfile(self.file_struct.ref_file):
            jam = jams.load(self.file_struct.ref_file)
            beat_annot = jam.search(namespace="beat.*")

            # If beat annotations exist, get times and frames
            if len(beat_annot) > 0:
                beats_inters, _ = beat_annot[0].data.to_interval_values()
                times = beats_inters[:, 0]
                frames = librosa.time_to_frames(times, sr=self.sr,
                                                hop_length=self.hop_length)
        return times, frames

    def compute_beat_sync_features(self, beat_frames, pad):
        """Make the features beat-synchronous.

        Parameters
        ----------
        beat_frames: np.array
            The frame indeces of the beat positions.
        pad: boolean
            If `True`, `beat_frames` is padded to span the full range.

        Returns
        -------
        beatsync: np.array
            The beat-synchronized features.
            `None` if the beat_frames was `None`.
        """
        if beat_frames is None:
            return None

        # Make beat synchronous
        beatsync = librosa.feature.sync(self._framesync_features.T,
                                        beat_frames, pad=pad).T

        # TODO: Make sure we have the right size (remove last frame if needed)
        #beatsync = beatsync[:len(beat_frames), :]
        return beatsync

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
            assert(np.isclose(
                self.dur, float(feats["globals"]["dur"]), rtol=tol))
            assert(self.sr == int(feats["globals"]["sample_rate"]))
            assert(self.hop_length == int(feats["globals"]["hop_length"]))
            assert(os.path.basename(self.file_struct.audio_file) ==
                   os.path.basename(feats["globals"]["audio_file"]))

            # Check for specific features params
            feat_params_err = FeatureParamsError(
                "Couldn't find features for %s id in file %s" %
                (self.get_id(), self.file_struct.features_file))
            if self.get_id() not in feats.keys():
                raise feat_params_err
            for param_name in self.get_param_names():
                value = getattr(self, param_name)
                if hasattr(value, '__call__'):
                    if value.__name__ != \
                            feats[self.get_id()]["params"][param_name]:
                        raise feat_params_err
                else:
                    if str(value) != \
                            feats[self.get_id()]["params"][param_name]:
                        raise feat_params_err

            # Store actual features
            self._est_beats_times = np.array(feats["est_beats"])
            self._est_beats_frames = librosa.core.time_to_frames(
                self._est_beats_times, sr=self.sr, hop_length=self.hop_length)
            self._framesync_features = \
                np.array(feats[self.get_id()]["framesync"])
            self._est_beatsync_features = \
                np.array(feats[self.get_id()]["est_beatsync"])

            # Read annotated beats if available
            if "ann_beats" in feats.keys():
                self._ann_beats_times = np.array(feats["ann_beats"])
                self._ann_beats_frames = librosa.core.time_to_frames(
                    self._ann_beats_times, sr=self.sr,
                    hop_length=self.hop_length)
                self._ann_beatsync_features = \
                    np.array(feats[self.get_id()]["ann_beatsync"])
        except KeyError:
            raise WrongFeaturesFormatError(
                "The features file %s is not correctly formatted" %
                self.file_struct.features_file)
        except AssertionError:
            raise FeaturesNotFound(
                "The features for the given parameters were not found in "
                "features file %s" % self.file_struct.features_file)
        except IOError:
            raise NoFeaturesFileError("Could not find features file %s",
                                      self.file_struct.features_file)

    def write_features(self):
        """Saves features to file."""
        out_json = collections.OrderedDict()
        try:
            # Only save the necessary information
            self.read_features()
        except (WrongFeaturesFormatError, FeaturesNotFound,
                NoFeaturesFileError):
            # We need to create the file or overwite it
            # Metadata
            out_json = collections.OrderedDict({"metadata": {
                "versions": {"librosa": librosa.__version__,
                             "msaf": msaf.__version__,
                             "numpy": np.__version__},
                "timestamp": datetime.datetime.today().strftime(
                    "%Y/%m/%d %H:%M:%S")}})

            # Global parameters
            out_json["globals"] = {
                "dur": self.dur,
                "sample_rate": self.sr,
                "hop_length": self.hop_length,
                "audio_file": self.file_struct.audio_file
            }

            # Beats
            out_json["est_beats"] = self._est_beats_times.tolist()
            if self._ann_beats_times is not None:
                out_json["ann_beats"] = self._ann_beats_times.tolist()
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
                if hasattr(value, '__call__'):
                    value = value.__name__
                else:
                    value = str(value)
                out_json[self.get_id()]["params"][param_name] = value

            # Actual features
            out_json[self.get_id()]["framesync"] = \
                self._framesync_features.tolist()
            out_json[self.get_id()]["est_beatsync"] = \
                self._est_beatsync_features.tolist()
            if self._ann_beatsync_features is not None:
                out_json[self.get_id()]["ann_beatsync"] = \
                    self._ann_beatsync_features.tolist()

            # Save it
            with open(self.file_struct.features_file, "w") as f:
                json.dump(out_json, f, indent=2)

    def get_param_names(self):
        """Returns the parameter names for these features, avoiding
        the global parameters."""
        return [name for name in vars(self) if not name.startswith('_') and
                name not in self._global_param_names]

    def _compute_all_features(self):
        """Computes all the features (beatsync, framesync) from the audio."""
        # Read actual audio waveform
        self._audio, _ = librosa.load(self.file_struct.audio_file,
                                      sr=self.sr)

        # Get duration of audio file
        self.dur = len(self._audio) / float(self.sr)

        # Compute actual features
        self._framesync_features = self.compute_features()

        # Compute/Read beats
        self._est_beats_times, self._est_beats_frames = self.estimate_beats()
        self._ann_beats_times, self._ann_beats_frames = self.read_ann_beats()

        # Beat-Synchronize
        pad = False  # Always append to the end of the features
        self._est_beatsync_features = \
            self.compute_beat_sync_features(self._est_beats_frames, pad)
        self._ann_beatsync_features = \
            self.compute_beat_sync_features(self._ann_beats_frames, pad)

    @property
    def frame_times(self):
        """This getter returns the frame times, for the corresponding type of
        features."""
        frame_times = None
        # Make sure we have already computed the features
        features = self.features
        if self.feat_type is FeatureTypes.framesync:
            frame_times = np.array([i * self.hop_length / float(self.sr) for
                                    i in np.arange(features.shape[0])])
        elif self.feat_type is FeatureTypes.est_beatsync:
            frame_times = self._est_beats_times
        elif self.feat_type is FeatureTypes.ann_beatsync:
            if self._ann_beatsync_features is None:
                raise FeatureTypeNotFound(
                    "Feature type %s is not valid because no annotated beats "
                    "were found" % self.feat_type)
            frame_times = self._ann_beats_times
        else:
            raise FeatureTypeNotFound("Feature type %s is not valid"
                                      % self.feat_type)
        return frame_times

    @property
    def features(self):
        """This getter will compute the actual features if they haven't
        been computed yet.

        Returns
        -------
        features: np.array
            The actual features. Each row corresponds to a feature vector.
        """
        # Compute features if needed
        if self._features is None:
            try:
                self.read_features()
            except (NoFeaturesFileError, FeaturesNotFound,
                    WrongFeaturesFormatError, FeatureParamsError) as e:
                try:
                    self._compute_all_features()
                    self.write_features()
                except IOError:
                    if isinstance(e, FeaturesNotFound) or \
                            isinstance(e, FeatureParamsError):
                        msg = "Computation of the features is needed for " \
                            "current parameters but no audio file was found." \
                            "Please, change your parameters or add the audio" \
                            " file in %s"
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
                    "were found" % self.feat_type)
            self._features = self._ann_beatsync_features
        else:
            raise FeatureTypeNotFound("Feature type %s is not valid." %
                                      self.feat_type)

        return self._features

    @classmethod
    def select_features(cls, features_id, file_struct, annot_beats, framesync):
        """Selects the features from the given parameters.

        Parameters
        ----------
        features_id: str
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
        if framesync:
            feat_type = FeatureTypes.est_beatsync
        elif annot_beats and not framesync:
            feat_type = FeatureTypes.ann_beatsync
        elif not annot_beats and not framesync:
            feat_type = FeatureTypes.est_beatsync
        else:
            raise FeatureTypeNotFound("Type of features not valid.")

        # Select features with default parameters
        return features_registry[features_id](file_struct, feat_type)

    def compute_features(self):
        raise NotImplementedError("This method must contain the actual "
                                  "implementation of the features")

    @classmethod
    def get_id(self):
        raise NotImplementedError("This method must return a string identifier"
                                  " of the features")
