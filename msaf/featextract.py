"""
MSAF module to extract the audio features.

Features to be computed:

- MFCC: Mel Frequency Cepstral Coefficients
- PCP: Harmonic Pithc Class Profile
- CQT: Constant-Q Transform
- Tempogram: Rhythmic features
- Beats
"""

import datetime
from builtins import super
from enum import Enum
import librosa
import jams
import json
import logging
import numpy as np
import os
import six

# Local stuff
import msaf
from msaf import utils
from msaf import input_output as io
from msaf.input_output import FileStruct
from msaf.exceptions import WrongFeaturesFormatError, NoFeaturesFileError,\
    FeaturesNotFound, FeatureTypeNotFound


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
            features_registry[cls.__name__] = cls
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
        if times[0] == 0:
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
                times = annot_beats_inters[:, 0]
                frames = librosa.time_to_frames(times, sr=self.sr,
                                                hop_length=self.hop_size)
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
        # beatsync = beatsync[:len(beats_frames), :]
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

            # Check that we have the correct global parameters
            assert(np.isclose(
                self.dur, float(feats["globals"]["dur"]), rtol=tol))
            assert(self.sr == int(feats["globals"]["sample_rate"]))
            assert(self.hop_length == int(feats["globals"]["hop_length"]))
            assert(self.get_id() in feats.keys())

            # Check for specific features params
            for param_name in self.get_param_names():
                value = getattr(self, param_name)
                assert(str(value) ==
                       feats[self.get_id()]["params"][param_name])

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
        except AssertError:
            raise FeaturesNotFound(
                "The features for the given parameters were not found in "
                "features file %s" % self.file_struct.features_file)
        except IOError:
            raise NoFeaturesFileError("Could not find features file %s",
                                      self.file_struct.features_file)

    def write_features(self):
        """Saves features to file."""
        # Metadata
        out_json = {"metadata": {
            "versions": {"librosa": librosa.__version__,
                         "msaf": msaf.__version__,
                         "numpy": np.__version__},
            "timestamp": datetime.datetime.today().strftime(
                "%Y/%m/%d %H:%M:%S")}}

        # Global parameters
        out_json["globals"] = {
            "dur": self.dur,
            "sample_rate": self.sr,
            "hop_length": self.hop_length,
            "audio_file": self.file_struct.audio_file
        }

        # Specific parameters of the current features
        out_json[self.get_id()] = {}
        out_json[self.get_id()]["params"] = {}
        for param_name in self.get_param_names():
            value = getattr(self, param_name)
            out_json[self.get_id()]["params"][param_name] = str(value)

        # Beats
        out_json["est_beats"] = self._est_beats_times.tolist()
        if self._ann_beats_times is not None:
            out_json["ann_beats"] = self._ann_beats_times.tolist()

        # Actual features
        out_json[self.get_id()]["framesync"] = \
            self._framesync_features.tolist()
        out_json[self.get_id()]["est_beatsync"] = \
            self._est_beatsync_features.tolist()
        if self._ann_beatsync_features is not None:
            out_json[self.get_id()]["ann_beatsync"] = \
                self._ann_beatsync_features.tolist()

        # Save it
        with open("caca.json", "w") as f:
            json.dump(out_json, f, indent=2)
            pass

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
        pad = True  # Always append to the end of the features
        self._est_beatsync_features = \
            self.compute_beat_sync_features(self._est_beats_frames, pad)
        self._ann_beatsync_features = \
            self.compute_beat_sync_features(self._ann_beats_frames, pad)

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
                    WrongFeaturesFormatError):
                self._compute_all_features()
                self.write_features()

        # Choose features based on type
        if self.feat_type is FeatureTypes.framesync:
            self._features = self._framesync_features
        elif self.feat_type is FeatureTypes.est_beatsync:
            self._features = self._est_beatsync_features
        elif self.feat_type is FeatureTypes.ann_beatsync:
            if self._ann_beatsync_features is None:
                # TODO: Improve error checking
                logging.warning("NO ANNOTATED BEATS!")
                self._features = self._est_beatsync_features
            else:
                self._features = self._ann_beatsync_features
        else:
            raise FeatureTypeNotFound("Feature type %s is not valid." %
                                      self.feat_type)

        return self._features

    def compute_features(self):
        raise NotImplementedError("This method must contain the actual "
                                  "implementation of the features")

    def get_id(self):
        raise NotImplementedError("This method must return a string identifier"
                                  " of the features")


class CQT(Features):
    """This class contains the implementation of the Constant-Q Transform.

    These features contain both harmonic and timbral content of the given
    audio signal.
    """
    def __init__(self, file_struct, feat_type, sr=msaf.Anal.sample_rate,
                 hop_length=msaf.Anal.hop_size, n_bins=msaf.Anal.cqt_bins,
                 norm=msaf.Anal.cqt_norm,
                 filter_scale=msaf.Anal.cqt_filter_scale, ref_power=np.max):
        """Constructor of the class.

        Parameters
        ----------
        file_struct: `msaf.input_output.FileStruct`
            Object containing the file paths from where to extract/read
            the features.
        feat_type: `FeatureTypes`
            Enum containing the type of features.
        sr: int > 0
            Sampling rate for the analysis.
        hop_length: int > 0
            Hop size in frames for the analysis.
        n_bins: int > 0
            Number of frequency bins for the CQT.
        norm: float
            The normalization coefficient.
        filter_scale: float
            The scale of the filter for the CQT.
        ref_power: function
            The reference power for logarithmic scaling.
        """
        # Init the parent
        super().__init__(file_struct=file_struct, sr=sr, hop_length=hop_length,
                         feat_type=feat_type)
        # Init the CQT parameters
        self.n_bins = n_bins
        self.norm = norm
        self.filter_scale = filter_scale
        self.ref_power = ref_power

    def get_id(self):
        """Identifier of these features."""
        return "cqt"

    def compute_features(self):
        """Actual implementation of the features.

        Returns
        -------
        cqt: np.array(N, F)
            The features, each row representing a feature vector for a give
            time frame/beat.
        """
        linear_cqt = np.abs(librosa.cqt(
            self._audio, sr=self.sr, hop_length=self.hop_length,
            n_bins=self.n_bins, norm=self.norm, filter_scale=self.filter_scale,
            real=False)) ** 2
        cqt = librosa.logamplitude(linear_cqt, ref_power=self.ref_power).T
        return cqt


#def compute_features(audio, y_harmonic):
    #"""Computes the HPCP and MFCC features.

    #Parameters
    #----------
    #audio: np.array(N)
        #Audio samples of the given input.
    #y_harmonic: np.array(N)
        #Harmonic part of the audio signal, in samples.

    #Returns
    #-------
    #mfcc: np.array(N, msaf.Anal.mfcc_coeff)
        #Mel-frequency Cepstral Coefficients.
    #pcp: np.array(N, 12)
        #Pitch Class Profiles.
    #tonnetz: np.array(N, 6)
        #Tonal Centroid features.
    #cqt: np.array(N, msaf.Anal.cqt_bins)
        #Constant-Q log-scale features.
    #tempogram: np.array(N, 192)
        #Tempogram features.
    #"""
    #logging.info("Computing Spectrogram...")
    ##S = librosa.feature.melspectrogram(audio,
                                       ##sr=msaf.Anal.sample_rate,
                                       ##n_fft=msaf.Anal.frame_size,
                                       ##hop_length=msaf.Anal.hop_size,
                                       ##n_mels=msaf.Anal.n_mels)

    #logging.info("Computing Constant-Q...")
    ##cqt = librosa.logamplitude(librosa.cqt(audio, sr=msaf.Anal.sample_rate,
                                           ##hop_length=msaf.Anal.hop_size,
                                           ##n_bins=msaf.Anal.cqt_bins) ** 2,
                               ##ref_power=np.max).T

    ##linear_cqt = np.abs(librosa.cqt(y_harmonic,
                                    ##sr=msaf.Anal.sample_rate,
                                    ##hop_length=msaf.Anal.hop_size,
                                    ##n_bins=msaf.Anal.cqt_bins,
                                    ##norm=np.inf,
                                    ##filter_scale=1,
                                    ##real=False))
    ##cqt = librosa.logamplitude(linear_cqt, ref_power=np.max).T

    #logging.info("Computing MFCCs...")
    ##log_S = librosa.logamplitude(S, ref_power=np.max)
    ##mfcc = librosa.feature.mfcc(S=log_S, n_mfcc=msaf.Anal.mfcc_coeff).T

    #logging.info("Computing PCPs...")
    #pcp_cqt = np.abs(librosa.hybrid_cqt(y_harmonic,
                                        #sr=msaf.Anal.sample_rate,
                                        #hop_length=msaf.Anal.hop_size,
                                        #n_bins=msaf.Anal.cqt_bins,
                                        #norm=1,
                                        #fmin=f_min)) ** 2
    #pcp = librosa.feature.chroma_cqt(C=pcp_cqt,
                                     #sr=msaf.Anal.sample_rate,
                                     #hop_length=msaf.Anal.hop_size,
                                     #n_octaves=msaf.Anal.n_octaves,
                                     #fmin=f_min).T
    ##pcp = librosa.feature.chroma_cqt(C=linear_cqt,
                                     ##sr=msaf.Anal.sample_rate,
                                     ##hop_length=msaf.Anal.hop_size,
                                     ##n_octaves=msaf.Anal.n_octaves,
                                     ##fmin=msaf.Anal.f_min).T
    ##pcp = librosa.feature.chroma_stft(y=y_harmonic,
                                      ##sr=msaf.Anal.sample_rate,
                                      ##n_fft=msaf.Anal.frame_size,
                                      ##hop_length=msaf.Anal.hop_size).T

    #logging.info("Computing Tonnetz...")
    ##tonnetz = utils.chroma_to_tonnetz(pcp)

    ## TODO:
    #tonnetz = pcp
    #mfcc = pcp
    #cqt = pcp

    #logging.info("Computing Tempogram...")
    #tempogram = librosa.feature.tempogram(audio,
                                          #sr=msaf.Anal.sample_rate,
                                          #hop_length=msaf.Anal.hop_size,
                                          #win_length=192).T
    #return mfcc, pcp, tonnetz, cqt, tempogram


#def save_features(out_file, features):
    #"""Saves the features into the specified file using the JSON format.

    #Parameters
    #----------
    #out_file: str
        #Path to the output file to be saved.
    #features: dict
        #Dictionary containing the features.
    #"""
    #logging.info("Saving the JSON file in %s" % out_file)
    #out_json = {"metadata": {"versions":
                             #{"librosa": librosa.__version__,
                              #"msaf": msaf.__version__}}}
    #out_json["analysis"] = {
        #"dur": features["anal"]["dur"],
        #"n_fft": msaf.Anal.frame_size,
        #"hop_size": msaf.Anal.hop_size,
        #"mfcc_coeff": msaf.Anal.mfcc_coeff,
        #"n_mels": msaf.Anal.n_mels,
        #"sample_rate": msaf.Anal.sample_rate,
        #"window_type": msaf.Anal.window_type
    #}
    #out_json["beats"] = {
        #"times": features["beats"].tolist()
    #}
    #out_json["timestamp"] = \
        #datetime.datetime.today().strftime("%Y/%m/%d %H:%M:%S")
    #out_json["framesync"] = {
        #"mfcc": features["mfcc"].tolist(),
        #"pcp": features["pcp"].tolist(),
        #"tonnetz": features["tonnetz"].tolist(),
        #"cqt": features["cqt"].tolist(),
        #"tempogram": features["tempogram"].tolist()
    #}
    #out_json["est_beatsync"] = {
        #"mfcc": features["bs_mfcc"].tolist(),
        #"pcp": features["bs_pcp"].tolist(),
        #"tonnetz": features["bs_tonnetz"].tolist(),
        #"cqt": features["bs_cqt"].tolist(),
        #"tempogram": features["bs_tempogram"].tolist()
    #}
    #try:
        #out_json["ann_beatsync"] = {
            #"mfcc": features["ann_mfcc"].tolist(),
            #"pcp": features["ann_pcp"].tolist(),
            #"tonnetz": features["ann_tonnetz"].tolist(),
            #"cqt": features["ann_cqt"].tolist(),
            #"tempogram": features["ann_tempogram"].tolist()
        #}
    #except:
        #logging.warning("No annotated beats")

    ## Actual save
    #with open(out_file, "w") as f:
        #json.dump(out_json, f, indent=2)


#def compute_beat_sync_features(features, beats_idx):
    #"""Given a dictionary of features, and the estimated index frames,
    #calculate beat-synchronous features."""
    #pad = True  # Always pad til the end of the actual audio file
    #bs_mfcc = librosa.feature.sync(features["mfcc"].T, beats_idx, pad=pad).T
    #bs_pcp = librosa.feature.sync(features["pcp"].T, beats_idx, pad=pad).T
    #bs_tonnetz = librosa.feature.sync(features["tonnetz"].T, beats_idx,
                                      #pad=pad).T
    #bs_cqt = librosa.feature.sync(features["cqt"].T, beats_idx, pad=pad).T
    #bs_tempogram = librosa.feature.sync(features["tempogram"].T, beats_idx,
                                        #pad=pad).T

    ## Make sure we have the right size (remove last frame if needed)
    #bs_mfcc = bs_mfcc[:len(beats_idx), :]
    #bs_pcp = bs_pcp[:len(beats_idx), :]
    #bs_tonnetz = bs_tonnetz[:len(beats_idx), :]
    #bs_cqt = bs_cqt[:len(beats_idx), :]
    #bs_tempogram = bs_tempogram[:len(beats_idx), :]
    #return bs_mfcc, bs_pcp, bs_tonnetz, bs_cqt, bs_tempogram


#def compute_features_for_audio_file(audio_file):
    #"""
    #Parameters
    #----------
    #audio_file: str
        #Path to the audio file.

    #Returns
    #-------
    #features: dict
        #Dictionary of audio features.
    #"""
    ## Load Audio
    #logging.info("Loading audio file %s" % os.path.basename(audio_file))
    #audio, sr = librosa.load(audio_file, sr=msaf.Anal.sample_rate)

    ## Compute harmonic-percussive source separation
    #logging.info("Computing Harmonic Percussive source separation...")
    #y_harmonic, y_percussive = librosa.effects.hpss(audio)

    ## Output features dict
    #features = {}

    ## Compute framesync features
    #features["mfcc"], features["pcp"], features["tonnetz"], \
        #features["cqt"], features["tempogram"] = \
        #compute_features(audio, y_harmonic)

    ## Estimate Beats
    #features["beats_idx"], features["beats"] = compute_beats(
        #y_percussive, sr=msaf.Anal.sample_rate)

    ## Compute Beat-sync features
    #features["bs_mfcc"], features["bs_pcp"], features["bs_tonnetz"], \
        #features["bs_cqt"], features["bs_tempogram"] = \
        #compute_beat_sync_features(features, features["beats_idx"])

    ## Analysis parameters
    #features["anal"] = {}
    #features["anal"]["n_fft"] = msaf.Anal.frame_size
    #features["anal"]["hop_size"] = msaf.Anal.hop_size
    #features["anal"]["mfcc_coeff"] = msaf.Anal.mfcc_coeff
    #features["anal"]["sample_rate"] = msaf.Anal.sample_rate
    #features["anal"]["window_type"] = msaf.Anal.window_type
    #features["anal"]["n_mels"] = msaf.Anal.n_mels
    #features["anal"]["dur"] = audio.shape[0] / float(msaf.Anal.sample_rate)

    #return features


#def compute_all_features(file_struct, sonify_beats=False, overwrite=False,
                         #out_beats="out_beats.wav"):
    #"""Computes all the features for a specific audio file and its respective
        #human annotations. It creates an audio file with the sonified estimated
        #beats if needed.

    #Parameters
    #----------
    #file_struct: FileStruct
        #Object containing all the set of file paths of the input file.
    #sonify_beats: bool
        #Whether to sonify the beats.
    #overwrite: bool
        #Whether to overwrite previous features JSON file.
    #out_beats: str
        #Path to the new file containing the sonified beats.
    #"""

    ## Output file
    #out_file = file_struct.features_file

    #if os.path.isfile(out_file) and not overwrite:
        #return  # Do nothing, file already exist and we are not overwriting it

    ## Compute the features for the given audio file
    #features = compute_features_for_audio_file(file_struct.audio_file)

    ## Save output as audio file
    #if sonify_beats:
        #logging.info("Sonifying beats...")
        #fs = 44100
        #audio, sr = librosa.load(file_struct.audio_file, sr=fs)
        #msaf.utils.sonify_clicks(audio, features["beats"], out_beats, fs,
                                 #offset=0.0)

    ## Read annotations if they exist in path/references_dir/file.jams
    #if os.path.isfile(file_struct.ref_file):
        #jam = jams.load(file_struct.ref_file)
        #beat_annot = jam.search(namespace="beat.*")

        ## If beat annotations exist, compute also annotated beatsync features
        #if len(beat_annot) > 0:
            #logging.info("Reading beat annotations from JAMS")
            #annot_beats_inters, _ = beat_annot[0].data.to_interval_values()
            #annot_beats_times = annot_beats_inters[:, 0]
            #annot_beats_idx = librosa.time_to_frames(
                #annot_beats_times, sr=msaf.Anal.sample_rate,
                #hop_length=msaf.Anal.hop_size)
            #features["ann_mfcc"], features["ann_pcp"], \
                #features["ann_tonnetz"], features["ann_cqt"], \
                #features["ann_tempogram"] = \
                #compute_beat_sync_features(features, annot_beats_idx)

    ## Save output as json file
    #save_features(out_file, features)
