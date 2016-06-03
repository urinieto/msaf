"""
MSAF module to extract the audio features using librosa.

Features to be computed:

- MFCC: Mel Frequency Cepstral Coefficients
- PCP: Harmonic Pithc Class Profile
- CQT: Constant-Q Transform
- Tempogram: Rhythmic features
- Beats
"""

import datetime
import librosa
import jams
from joblib import Parallel, delayed
import logging
import numpy as np
import os
import json

# Local stuff
import msaf
from msaf import utils
from msaf import input_output as io
from msaf.input_output import FileStruct


class Features(object):
    """Features class."""
    def __init__(self, audio_file, json_file, sr, hop_length, beat_sync):
        self.audio_file = audio_file
        self.json_file = json_file
        self.sr = sr
        self.hop_length = hop_length
        self.beat_sync = beat_sync
        self._audio = None
        self._audio_harmonic = None
        self._audio_percussive = None
        self._features = None
        self._beatsync_features = None
        self._framesync_features = None
        self._beats_times = None
        self._beats_frames = None

    def compute_HPSS(self):
        """Compute harmonic-percussive source separation."""
        return librosa.effects.hpss(self._audio)

    def compute_beats(self):
        """Computes the beats using librosa."""
        # Compute harmonic-percussive source separiation if needed
        if self._audio_percussive is None:
            self._audio_harmonic, self._audio_percussive = self.compute_HPSS()

        tempo, beats_idx = librosa.beat.beat_track(
            y=self._audio_percussive, sr=self.sr,
            hop_length=self.hop_length)

        # To times
        times = librosa.frames_to_time(beats_idx, sr=sr,
                                       hop_length=msaf.Anal.hop_size)

        if times[0] == 0:
            times = times[1:]
            beats_idx = beats_idx[1:]

        return times, beats_idx

    def compute_beat_sync_features(self, pad):
        """Makes the features beat-synchronous."""
        beat_sync = librosa.feature.sync(self._framesync_features.T,
                                         self.beats_frames, pad=pad).T

        # TODO: Make sure we have the right size (remove last frame if needed)
        # beat_sync = beat_sync[:len(self._beats_frames), :]
        return beat_sync

    def read_features(self):
        """Reads the features from a file."""
        # TODO: Read features and return True if features could be read.
        return False

    def write_features(self):
        """Saves features to file."""
        out_json = {"metadata": {"versions":
                                 {"librosa": librosa.__version__,
                                  "msaf": msaf.__version__,
                                  "numpy": np.__version__}}}
        out_json["analysis"] = {
            "dur": features["anal"]["dur"],
            "n_fft": msaf.Anal.frame_size,
            "hop_size": msaf.Anal.hop_size,
            "mfcc_coeff": msaf.Anal.mfcc_coeff,
            "n_mels": msaf.Anal.n_mels,
            "sample_rate": msaf.Anal.sample_rate,
            "window_type": msaf.Anal.window_type
        }
        out_json["beats"] = {
            "times": features["beats"].tolist()
        }
        out_json["timestamp"] = \
            datetime.datetime.today().strftime("%Y/%m/%d %H:%M:%S")
        out_json["framesync"] = {
            "mfcc": features["mfcc"].tolist(),
            "pcp": features["pcp"].tolist(),
            "tonnetz": features["tonnetz"].tolist(),
            "cqt": features["cqt"].tolist(),
            "tempogram": features["tempogram"].tolist()
        }
        out_json["est_beatsync"] = {
            "mfcc": features["bs_mfcc"].tolist(),
            "pcp": features["bs_pcp"].tolist(),
            "tonnetz": features["bs_tonnetz"].tolist(),
            "cqt": features["bs_cqt"].tolist(),
            "tempogram": features["bs_tempogram"].tolist()
        }
        try:
            out_json["ann_beatsync"] = {
                "mfcc": features["ann_mfcc"].tolist(),
                "pcp": features["ann_pcp"].tolist(),
                "tonnetz": features["ann_tonnetz"].tolist(),
                "cqt": features["ann_cqt"].tolist(),
                "tempogram": features["ann_tempogram"].tolist()
            }
        except:
            logging.warning("No annotated beats")

        # Actual save
        with open(out_file, "w") as f:
            json.dump(out_json, f, indent=2)
            pass

    @property
    def features(self):
        # Compute features if needed
        if self._features is None:
            if not self.read_features():
                self._audio, _ = librosa.load(self.audio_file, sr=self.sr)
                self._framesync_features = self.compute_features()
                self._beats_times, self._beats_frames = self.compute_beats()
                pad = True  # Always append to the end of the features
                self._beatsync_features = self.compute_beat_sync_features(pad)

        # Choose features based on type (beatsync or framesync)
        self._features = self._beatsync_features \
            if self.beat_sync else self._framesync_features

        return self._features

    def compute_features(self):
        raise NotImplementedError("This method must contain the actual "
                                  "implementation of the features")

    def get_id(self):
        raise NotImplementedError("This method must return a string identifier"
                                  " of the features")


class CQT(Features):

    def __init__(self, audio_file, json_file, beat_sync,
                 sr=msaf.Anal.sample_rate, hop_length=msaf.Anal.hop_size,
                 n_bins=msaf.Anal.sample_rate, norm=msaf.Anal.cqt_norm,
                 filter_scale=msaf.Anal.cqt_filter_scale,
                 ref_power=msaf.Anal.cqt_ref_power):
        # Init the parent
        super().__init__(audio_file=audio_file, json_file=json_file,
                         sr=sr, hop_length=hop_length, beat_sync=beat_sync)

        self.sr = sr

    def compute_features(self):
        linear_cqt = np.abs(librosa.cqt(self.audio,
                                        sr=self.sr,
                                        hop_length=self.hop_length,
                                        n_bins=self.cqt_bins,
                                        norm=self.cqt_norm,
                                        filter_scale=self.filter_scale,
                                        real=False)) ** 2
        cqt = librosa.logamplitude(linear_cqt, ref_power=self.ref_power).T
        return cqt

    def get_id(self):
        return "cqt"


def compute_features(audio, y_harmonic):
    """Computes the HPCP and MFCC features.

    Parameters
    ----------
    audio: np.array(N)
        Audio samples of the given input.
    y_harmonic: np.array(N)
        Harmonic part of the audio signal, in samples.

    Returns
    -------
    mfcc: np.array(N, msaf.Anal.mfcc_coeff)
        Mel-frequency Cepstral Coefficients.
    pcp: np.array(N, 12)
        Pitch Class Profiles.
    tonnetz: np.array(N, 6)
        Tonal Centroid features.
    cqt: np.array(N, msaf.Anal.cqt_bins)
        Constant-Q log-scale features.
    tempogram: np.array(N, 192)
        Tempogram features.
    """
    logging.info("Computing Spectrogram...")
    #S = librosa.feature.melspectrogram(audio,
                                       #sr=msaf.Anal.sample_rate,
                                       #n_fft=msaf.Anal.frame_size,
                                       #hop_length=msaf.Anal.hop_size,
                                       #n_mels=msaf.Anal.n_mels)

    logging.info("Computing Constant-Q...")
    #cqt = librosa.logamplitude(librosa.cqt(audio, sr=msaf.Anal.sample_rate,
                                           #hop_length=msaf.Anal.hop_size,
                                           #n_bins=msaf.Anal.cqt_bins) ** 2,
                               #ref_power=np.max).T

    #linear_cqt = np.abs(librosa.cqt(y_harmonic,
                                    #sr=msaf.Anal.sample_rate,
                                    #hop_length=msaf.Anal.hop_size,
                                    #n_bins=msaf.Anal.cqt_bins,
                                    #norm=np.inf,
                                    #filter_scale=1,
                                    #real=False))
    #cqt = librosa.logamplitude(linear_cqt, ref_power=np.max).T

    logging.info("Computing MFCCs...")
    #log_S = librosa.logamplitude(S, ref_power=np.max)
    #mfcc = librosa.feature.mfcc(S=log_S, n_mfcc=msaf.Anal.mfcc_coeff).T

    logging.info("Computing PCPs...")
    pcp_cqt = np.abs(librosa.hybrid_cqt(y_harmonic,
                                        sr=msaf.Anal.sample_rate,
                                        hop_length=msaf.Anal.hop_size,
                                        n_bins=msaf.Anal.cqt_bins,
                                        norm=1,
                                        fmin=f_min)) ** 2
    pcp = librosa.feature.chroma_cqt(C=pcp_cqt,
                                     sr=msaf.Anal.sample_rate,
                                     hop_length=msaf.Anal.hop_size,
                                     n_octaves=msaf.Anal.n_octaves,
                                     fmin=f_min).T
    #pcp = librosa.feature.chroma_cqt(C=linear_cqt,
                                     #sr=msaf.Anal.sample_rate,
                                     #hop_length=msaf.Anal.hop_size,
                                     #n_octaves=msaf.Anal.n_octaves,
                                     #fmin=msaf.Anal.f_min).T
    #pcp = librosa.feature.chroma_stft(y=y_harmonic,
                                      #sr=msaf.Anal.sample_rate,
                                      #n_fft=msaf.Anal.frame_size,
                                      #hop_length=msaf.Anal.hop_size).T

    logging.info("Computing Tonnetz...")
    #tonnetz = utils.chroma_to_tonnetz(pcp)

    # TODO:
    tonnetz = pcp
    mfcc = pcp
    cqt = pcp

    logging.info("Computing Tempogram...")
    tempogram = librosa.feature.tempogram(audio,
                                          sr=msaf.Anal.sample_rate,
                                          hop_length=msaf.Anal.hop_size,
                                          win_length=192).T
    return mfcc, pcp, tonnetz, cqt, tempogram


def save_features(out_file, features):
    """Saves the features into the specified file using the JSON format.

    Parameters
    ----------
    out_file: str
        Path to the output file to be saved.
    features: dict
        Dictionary containing the features.
    """
    logging.info("Saving the JSON file in %s" % out_file)
    out_json = {"metadata": {"versions":
                             {"librosa": librosa.__version__,
                              "msaf": msaf.__version__}}}
    out_json["analysis"] = {
        "dur": features["anal"]["dur"],
        "n_fft": msaf.Anal.frame_size,
        "hop_size": msaf.Anal.hop_size,
        "mfcc_coeff": msaf.Anal.mfcc_coeff,
        "n_mels": msaf.Anal.n_mels,
        "sample_rate": msaf.Anal.sample_rate,
        "window_type": msaf.Anal.window_type
    }
    out_json["beats"] = {
        "times": features["beats"].tolist()
    }
    out_json["timestamp"] = \
        datetime.datetime.today().strftime("%Y/%m/%d %H:%M:%S")
    out_json["framesync"] = {
        "mfcc": features["mfcc"].tolist(),
        "pcp": features["pcp"].tolist(),
        "tonnetz": features["tonnetz"].tolist(),
        "cqt": features["cqt"].tolist(),
        "tempogram": features["tempogram"].tolist()
    }
    out_json["est_beatsync"] = {
        "mfcc": features["bs_mfcc"].tolist(),
        "pcp": features["bs_pcp"].tolist(),
        "tonnetz": features["bs_tonnetz"].tolist(),
        "cqt": features["bs_cqt"].tolist(),
        "tempogram": features["bs_tempogram"].tolist()
    }
    try:
        out_json["ann_beatsync"] = {
            "mfcc": features["ann_mfcc"].tolist(),
            "pcp": features["ann_pcp"].tolist(),
            "tonnetz": features["ann_tonnetz"].tolist(),
            "cqt": features["ann_cqt"].tolist(),
            "tempogram": features["ann_tempogram"].tolist()
        }
    except:
        logging.warning("No annotated beats")

    # Actual save
    with open(out_file, "w") as f:
        json.dump(out_json, f, indent=2)


def compute_beat_sync_features(features, beats_idx):
    """Given a dictionary of features, and the estimated index frames,
    calculate beat-synchronous features."""
    pad = True  # Always pad til the end of the actual audio file
    bs_mfcc = librosa.feature.sync(features["mfcc"].T, beats_idx, pad=pad).T
    bs_pcp = librosa.feature.sync(features["pcp"].T, beats_idx, pad=pad).T
    bs_tonnetz = librosa.feature.sync(features["tonnetz"].T, beats_idx,
                                      pad=pad).T
    bs_cqt = librosa.feature.sync(features["cqt"].T, beats_idx, pad=pad).T
    bs_tempogram = librosa.feature.sync(features["tempogram"].T, beats_idx,
                                        pad=pad).T

    # Make sure we have the right size (remove last frame if needed)
    bs_mfcc = bs_mfcc[:len(beats_idx), :]
    bs_pcp = bs_pcp[:len(beats_idx), :]
    bs_tonnetz = bs_tonnetz[:len(beats_idx), :]
    bs_cqt = bs_cqt[:len(beats_idx), :]
    bs_tempogram = bs_tempogram[:len(beats_idx), :]
    return bs_mfcc, bs_pcp, bs_tonnetz, bs_cqt, bs_tempogram


def compute_features_for_audio_file(audio_file):
    """
    Parameters
    ----------
    audio_file: str
        Path to the audio file.

    Returns
    -------
    features: dict
        Dictionary of audio features.
    """
    # Load Audio
    logging.info("Loading audio file %s" % os.path.basename(audio_file))
    audio, sr = librosa.load(audio_file, sr=msaf.Anal.sample_rate)

    # Compute harmonic-percussive source separation
    logging.info("Computing Harmonic Percussive source separation...")
    y_harmonic, y_percussive = librosa.effects.hpss(audio)

    # Output features dict
    features = {}

    # Compute framesync features
    features["mfcc"], features["pcp"], features["tonnetz"], \
        features["cqt"], features["tempogram"] = \
        compute_features(audio, y_harmonic)

    # Estimate Beats
    features["beats_idx"], features["beats"] = compute_beats(
        y_percussive, sr=msaf.Anal.sample_rate)

    # Compute Beat-sync features
    features["bs_mfcc"], features["bs_pcp"], features["bs_tonnetz"], \
        features["bs_cqt"], features["bs_tempogram"] = \
        compute_beat_sync_features(features, features["beats_idx"])

    # Analysis parameters
    features["anal"] = {}
    features["anal"]["n_fft"] = msaf.Anal.frame_size
    features["anal"]["hop_size"] = msaf.Anal.hop_size
    features["anal"]["mfcc_coeff"] = msaf.Anal.mfcc_coeff
    features["anal"]["sample_rate"] = msaf.Anal.sample_rate
    features["anal"]["window_type"] = msaf.Anal.window_type
    features["anal"]["n_mels"] = msaf.Anal.n_mels
    features["anal"]["dur"] = audio.shape[0] / float(msaf.Anal.sample_rate)

    return features


def compute_all_features(file_struct, sonify_beats=False, overwrite=False,
                         out_beats="out_beats.wav"):
    """Computes all the features for a specific audio file and its respective
        human annotations. It creates an audio file with the sonified estimated
        beats if needed.

    Parameters
    ----------
    file_struct: FileStruct
        Object containing all the set of file paths of the input file.
    sonify_beats: bool
        Whether to sonify the beats.
    overwrite: bool
        Whether to overwrite previous features JSON file.
    out_beats: str
        Path to the new file containing the sonified beats.
    """

    # Output file
    out_file = file_struct.features_file

    if os.path.isfile(out_file) and not overwrite:
        return  # Do nothing, file already exist and we are not overwriting it

    # Compute the features for the given audio file
    features = compute_features_for_audio_file(file_struct.audio_file)

    # Save output as audio file
    if sonify_beats:
        logging.info("Sonifying beats...")
        fs = 44100
        audio, sr = librosa.load(file_struct.audio_file, sr=fs)
        msaf.utils.sonify_clicks(audio, features["beats"], out_beats, fs,
                                 offset=0.0)

    # Read annotations if they exist in path/references_dir/file.jams
    if os.path.isfile(file_struct.ref_file):
        jam = jams.load(file_struct.ref_file)
        beat_annot = jam.search(namespace="beat.*")

        # If beat annotations exist, compute also annotated beatsync features
        if len(beat_annot) > 0:
            logging.info("Reading beat annotations from JAMS")
            annot_beats_inters, _ = beat_annot[0].data.to_interval_values()
            annot_beats_times = annot_beats_inters[:, 0]
            annot_beats_idx = librosa.time_to_frames(
                annot_beats_times, sr=msaf.Anal.sample_rate,
                hop_length=msaf.Anal.hop_size)
            features["ann_mfcc"], features["ann_pcp"], \
                features["ann_tonnetz"], features["ann_cqt"], \
                features["ann_tempogram"] = \
                compute_beat_sync_features(features, annot_beats_idx)

    # Save output as json file
    save_features(out_file, features)


def process(in_path, sonify_beats=False, n_jobs=1, overwrite=False,
            out_file="out.json", out_beats="out_beats.wav",
            ds_name="*"):
    """Main process to compute features.

    Parameters
    ----------
    in_path: str
        Path to the file or dataset to compute the features.
    sonify_beats: bool
        Whether to sonify the beats on top of the audio file
        (single file mode only).
    n_jobs: int
        Number of threads (collection mode only).
    overwrite: bool
        Whether to overwrite the previously computed features.
    out_file: str
        Path to the output json file (single file mode only).
    out_beats: str
        Path to the new file containing the sonified beats.
    ds_name: str
        Name of the prefix of the dataset (e.g., Beatles)
    """

    # If in_path it's a file, we only compute one file
    if os.path.isfile(in_path):
        file_struct = FileStruct(in_path)
        file_struct.features_file = out_file
        compute_all_features(file_struct, sonify_beats, overwrite, out_beats)

    elif os.path.isdir(in_path):
        # Check that in_path exists
        utils.ensure_dir(in_path)

        # Get files
        file_structs = io.get_dataset_files(in_path, ds_name=ds_name)

        # Compute features using joblib
        Parallel(n_jobs=n_jobs)(delayed(compute_all_features)(
            file_struct, sonify_beats, overwrite, out_beats)
            for file_struct in file_structs)
