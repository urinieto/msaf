"""
MSAF module to extract the audio features using librosa.

Features to be computed:

- MFCC: Mel Frequency Cepstral Coefficients
- HPCP: Harmonic Pithc Class Profile
- Beats
"""

import argparse
import datetime
import librosa
from joblib import Parallel, delayed
import logging
import numpy as np
import os
import time
import json

# Local stuff
import msaf
from msaf import jams2
from msaf import utils
from msaf import input_output as io


def compute_beats(y_percussive):
    """Computes the beats using librosa.

    Parameters
    ----------
    y_percussive: np.array
        Percussive part of the audio signal in samples.

    Returns
    -------
    beats_idx: np.array
        Indeces in frames of the estimated beats.
    beats_times: np.array
        Time of the estimated beats.
    """
    logging.info("Estimating Beats...")
    tempo, beats_idx = librosa.beat.beat_track(y=y_percussive,
                                               sr=msaf.Anal.sample_rate,
                                               hop_length=msaf.Anal.hop_size)
    return beats_idx, librosa.frames_to_time(beats_idx,
                                             sr=msaf.Anal.sample_rate,
                                             hop_length=msaf.Anal.hop_size)


def compute_features(audio, y_harmonic):
    """Computes the HPCP and MFCC features."""
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
    hpcp = librosa.feature.chromagram(y=y_harmonic,
                                      sr=msaf.Anal.sample_rate,
                                      n_fft=msaf.Anal.frame_size,
                                      hop_length=msaf.Anal.hop_size).T

    #plt.imshow(hpcp.T, interpolation="nearest", aspect="auto"); plt.show()
    logging.info("Computing Tonnetz...")
    tonnetz = utils.chroma_to_tonnetz(hpcp)
    return mfcc, hpcp, tonnetz


def save_features(out_file, features):
    """Saves the features into the specified file using the JSON format."""
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
        logging.warning("No annotated beats for %s" % out_file)

    # Actual save
    with open(out_file, "w") as f:
        json.dump(out_json, f, indent=2)


def compute_beat_sync_features(features, beats_idx):
    """Given a dictionary of features, and the estimated index frames,
    calculate beat-synchronous features."""
    bs_mfcc = librosa.feature.sync(features["mfcc"].T, beats_idx).T
    bs_hpcp = librosa.feature.sync(features["hpcp"].T, beats_idx).T
    bs_tonnetz = librosa.feature.sync(features["tonnetz"].T, beats_idx).T
    return bs_mfcc, bs_hpcp, bs_tonnetz


def compute_features_for_audio_file(audio_file):
    """
    Parameters
    ----------
    audio_file: str
        Path to the audio file.

    Returns
    -------
    audio: np.array
        Audio samples.
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
    features["beats_idx"], features["beats"] = compute_beats(y_percussive)

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

    return audio, features


def compute_all_features(file_struct, audio_beats=False, overwrite=False):
    """Computes all the features for a specific audio file and its respective
        human annotations. It creates an audio file with the estimated
        beats if needed."""

    # Output file
    out_file = file_struct.features_file

    if os.path.isfile(out_file) and not overwrite:
        return  # Do nothing, file already exist and we are not overwriting it

    # Compute the features for the given audio file
    audio, features = compute_features_for_audio_file(file_struct.audio_file)

    # Save output as audio file
    if audio_beats:
        logging.info("Sonifying beats... (TODO)")
        #TODO

    # Read annotations if they exist in path/references_dir/file.jams
    if os.path.isfile(file_struct.ref_file):
        jam = jams2.load(file_struct.ref_file)

        # If beat annotations exist, compute also annotated beatsyn features
        if jam.beats != []:
            logging.info("Reading beat annotations from JAMS")
            annot = jam.beats[0]
            annot_beats = []
            for data in annot.data:
                annot_beats.append(data.time.value)
            annot_beats = np.unique(annot_beats).tolist()
            annot_beats_idx = librosa.time_to_frames(annot_beats,
                                                     sr=msaf.Anal.sample_rate,
                                                     hop_length=msaf.Anal.hop_size)
            features["annot_mfcc"], features["annot_hpcp"], \
                features["annot_tonnetz"] = \
                compute_beat_sync_features(features, annot_beats_idx)

    # Save output as json file
    save_features(out_file, features)


def process(in_path, audio_beats=False, n_jobs=1, overwrite=False):
    """Main process."""

    # If in_path it's a file, we only compute one file
    if os.path.isfile(in_path):
        compute_all_features(in_path, audio_beats, overwrite)

    elif os.path.isdir(in_path):
        # Check that in_path exists
        utils.ensure_dir(in_path)

        # Get files
        file_structs = io.get_dataset_files(in_path)

        # Compute features using joblib
        Parallel(n_jobs=n_jobs)(delayed(compute_all_features)(
            file_struct, audio_beats, overwrite)
            for file_struct in file_structs)


def main():
    """Main function to parse the arguments and call the main process."""
    parser = argparse.ArgumentParser(description=
        "Extracts a set of features from the Segmentation dataset or a given "
        "audio file and saves them into the 'features' folder of the dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("in_path",
                        action="store",
                        help="Input dataset dir or audio file")
    parser.add_argument("-a",
                        action="store_true",
                        dest="audio_beats",
                        help="Output audio file with estimated beats",
                        default=False)
    parser.add_argument("-j",
                        action="store",
                        dest="n_jobs",
                        type=int,
                        help="Number of jobs (threads)",
                        default=4)
    parser.add_argument("-ow",
                        action="store_true",
                        dest="overwrite",
                        help="Overwrite the previously computed features",
                        default=False)
    args = parser.parse_args()
    start_time = time.time()

    # Setup the logger
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s',
        level=logging.INFO)

    # Run the algorithm
    process(args.in_path, args.audio_beats, n_jobs=args.n_jobs,
            overwrite=args.overwrite)

    # Done!
    logging.info("Done! Took %.2f seconds." % (time.time() - start_time))

if __name__ == '__main__':
    main()
