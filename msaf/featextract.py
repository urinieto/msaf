"""
MSAF module to extract the audio features using librosa.

Features to be computed:

- MFCC: Mel Frequency Cepstral Coefficients
- HPCP: Harmonic Pithc Class Profile
- Beats
"""

import datetime
import librosa
from joblib import Parallel, delayed
import logging
import numpy as np
import os
import json

# Local stuff
import msaf
from msaf import jams2
from msaf import utils
from msaf import input_output as io
from msaf.input_output import FileStruct


def compute_beats(y_percussive, sr=22050):
    """Computes the beats using librosa.

    Parameters
    ----------
    y_percussive: np.array
        Percussive part of the audio signal in samples.
    sr: int
        Sample rate.

    Returns
    -------
    beats_idx: np.array
        Indeces in frames of the estimated beats.
    beats_times: np.array
        Time of the estimated beats.
    """
    logging.info("Estimating Beats...")
    tempo, beats_idx = librosa.beat.beat_track(y=y_percussive, sr=sr,
                                               hop_length=msaf.Anal.hop_size)
    return beats_idx, librosa.frames_to_time(beats_idx, sr=sr,
                                             hop_length=msaf.Anal.hop_size)


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
    hpcp: np.array(N, 12)
        Pitch Class Profiles.
    tonnetz: np.array(N, 6)
        Tonal Centroid features.
    """
    logging.info("Computing Spectrogram...")
    S = librosa.feature.melspectrogram(audio,
                                       sr=msaf.Anal.sample_rate,
                                       n_fft=msaf.Anal.frame_size,
                                       hop_length=msaf.Anal.hop_size,
                                       n_mels=msaf.Anal.n_mels)

    logging.info("Computing MFCCs...")
    log_S = librosa.logamplitude(S, ref_power=np.max)
    mfcc = librosa.feature.mfcc(S=log_S, n_mfcc=msaf.Anal.mfcc_coeff).T

    logging.info("Computing HPCPs...")
    hpcp = librosa.feature.chroma_cqt(y=y_harmonic,
                                      sr=msaf.Anal.sample_rate,
                                      hop_length=msaf.Anal.hop_size).T

    #plt.imshow(hpcp.T, interpolation="nearest", aspect="auto"); plt.show()
    logging.info("Computing Tonnetz...")
    tonnetz = utils.chroma_to_tonnetz(hpcp)
    return mfcc, hpcp, tonnetz


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
    out_json = {"metadata": {"version": {"librosa": librosa.__version__}}}
    out_json["analysis"] = {
        "dur": features["anal"]["dur"],
        "frame_rate": msaf.Anal.frame_size,
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
        "hpcp": features["hpcp"].tolist(),
        "tonnetz": features["tonnetz"].tolist()
    }
    out_json["est_beatsync"] = {
        "mfcc": features["bs_mfcc"].tolist(),
        "hpcp": features["bs_hpcp"].tolist(),
        "tonnetz": features["bs_tonnetz"].tolist()
    }
    try:
        out_json["ann_beatsync"] = {
            "mfcc": features["ann_mfcc"].tolist(),
            "hpcp": features["ann_hpcp"].tolist(),
            "tonnetz": features["ann_tonnetz"].tolist()
        }
    except:
        logging.warning("No annotated beats")

    # Actual save
    with open(out_file, "w") as f:
        json.dump(out_json, f, indent=2)


def compute_beat_sync_features(features, beats_idx):
    """Given a dictionary of features, and the estimated index frames,
    calculate beat-synchronous features."""
    bs_mfcc = librosa.feature.sync(features["mfcc"].T, beats_idx, pad=False).T
    bs_hpcp = librosa.feature.sync(features["hpcp"].T, beats_idx, pad=False).T
    bs_tonnetz = librosa.feature.sync(features["tonnetz"].T, beats_idx,
                                      pad=False).T
    return bs_mfcc, bs_hpcp, bs_tonnetz


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
    features["mfcc"], features["hpcp"], features["tonnetz"] = \
        compute_features(audio, y_harmonic)

    # Estimate Beats
    features["beats_idx"], features["beats"] = compute_beats(
        y_percussive, sr=msaf.Anal.sample_rate)

    # Compute Beat-sync features
    features["bs_mfcc"], features["bs_hpcp"], features["bs_tonnetz"] = \
        compute_beat_sync_features(features, features["beats_idx"])

    # Analysis parameters
    features["anal"] = {}
    features["anal"]["frame_rate"] = msaf.Anal.frame_size
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
        jam = jams2.load(file_struct.ref_file)

        # If beat annotations exist, compute also annotated beatsync features
        if jam.beats != []:
            logging.info("Reading beat annotations from JAMS")
            annot = jam.beats[0]
            annot_beats = []
            for data in annot.data:
                annot_beats.append(data.time.value)
            annot_beats = np.unique(annot_beats)
            annot_beats_idx = librosa.time_to_frames(
                annot_beats, sr=msaf.Anal.sample_rate,
                hop_length=msaf.Anal.hop_size)
            features["ann_mfcc"], features["ann_hpcp"], \
                features["ann_tonnetz"] = \
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
