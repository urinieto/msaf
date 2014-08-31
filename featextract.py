#!/usr/bin/env python
"""
This script uses Essentia in order to extract a set of features to be used
as input for the segmentation algorithms.

Features to be computed:

- MFCC: Mel Frequency Cepstral Coefficients
- HPCP: Harmonic Pithc Class Profile
- Beats
"""

__author__ = "Oriol Nieto"
__copyright__ = "Copyright 2014, Music and Audio Research Lab (MARL)"
__license__ = "GPL"
__version__ = "1.0"
__email__ = "oriol@nyu.edu"

import argparse
import datetime
import essentia
import essentia.standard as ES
from essentia.standard import YamlOutput
from joblib import Parallel, delayed
import logging
import numpy as np
import os
import time

# Local stuff
import msaf
from msaf import jams2
from msaf import utils
from msaf import input_output as io


class STFTFeature:
    """Class to easily compute the features that require a frame based
        spectrum process (or STFT)."""
    def __init__(self, frame_size, hop_size, window_type, feature, sample_rate,
                 beats=None):
        """STFTFeature constructor."""
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.window_type = window_type
        self.w = ES.Windowing(type=window_type)
        self.spectrum = ES.Spectrum()
        self.feature = feature  # Essentia feature object
        self.beats = beats
        self.sample_rate = sample_rate

    def compute_features(self, audio):
        """Computes the specified Essentia features from the audio array."""
        features = []

        for frame in ES.FrameGenerator(audio,
                frameSize=self.frame_size, hopSize=self.hop_size):
            if self.feature.name() == "MFCC":
                bands, coeffs = self.feature(self.spectrum(self.w(frame)))
            elif self.feature.name() == "HPCP":
                spectral_peaks = ES.SpectralPeaks()
                freqs, mags = spectral_peaks(self.spectrum(self.w(frame)))
                coeffs = self.feature(freqs, mags)
            features.append(coeffs)

        # Convert to Essentia Numpy array
        features = essentia.array(features)

        # Make beat-synchronous if we have the beats
        if self.beats is not None and self.beats != []:
            framerate = self.sample_rate / float(self.hop_size)
            tframes = np.arange(features.shape[0]) / float(framerate)
            features = utils.resample_mx(features.T, tframes, self.beats).T

        return features


def double_beats(beats):
    """Double the beats."""
    new_beats = []
    for i, beat in enumerate(beats[:-1]):
        new_beats.append(beat)
        new_beats.append((beat + beats[i + 1]) / 2.0)
    return essentia.array(new_beats)


def compute_beats(audio):
    """Computes the beats using Essentia."""
    logging.info("Computing Beats...")
    conf = 1.0
    beats, conf = ES.BeatTrackerMultiFeature()(audio)
    # beats = ES.BeatTrackerDegara()(audio)
    beats *= 44100 / msaf.Anal.sample_rate  # Essentia requires 44100 input

    # Double the beats if found beats are too little
    th = 0.9  # 1 would equal to at least 1 beat per second
    while beats.shape[0] / (audio.shape[0] / float(msaf.Anal.sample_rate)) < th \
            and beats.shape[0] > 2:
        beats = double_beats(beats)
    return beats, conf


def compute_features(audio, beats=None):
    """Computes the HPCP and MFCC beat-synchronous features given a set
        of beats (beats)."""
    beatsync_str = ""
    if beats is not None:
        beatsync_str = "Beat-synchronous "

    MFCC = STFTFeature(msaf.Anal.frame_size, msaf.Anal.hop_size,
                       msaf.Anal.window_type,
                       ES.MFCC(numberCoefficients=msaf.Anal.mfcc_coeff),
                       msaf.Anal.sample_rate, beats)
    HPCP = STFTFeature(msaf.Anal.frame_rate, msaf.Anal.hop_size,
                       msaf.Anal.window_type, ES.HPCP(), msaf.Anal.sample_rate,
                       beats)
    logging.info("Computing %sMFCCs..." % beatsync_str)
    mfcc = MFCC.compute_features(audio)
    logging.info("Computing %sHPCPs..." % beatsync_str)
    hpcp = HPCP.compute_features(audio)
    #plt.imshow(hpcp.T, interpolation="nearest", aspect="auto"); plt.show()
    logging.info("Computing %sTonnetz..." % beatsync_str)
    tonnetz = utils.chroma_to_tonnetz(hpcp)
    return mfcc, hpcp, tonnetz


def save_features(key, pool, mfcc, hpcp, tonnetz):
    """Saves the features into the specified pool under the given key."""
    [pool.add(key + ".mfcc", essentia.array(mfcc_coeff))
        for mfcc_coeff in mfcc]
    [pool.add(key + ".hpcp", essentia.array(hpcp_coeff))
        for hpcp_coeff in hpcp]
    [pool.add(key + ".tonnetz", essentia.array(tonnetz_coeff))
        for tonnetz_coeff in tonnetz]


def compute_features_for_audio_file(audio_file):
    # Load Audio
    logging.info("Loading audio file %s" % os.path.basename(audio_file))
    audio = ES.MonoLoader(filename=audio_file,
                          sampleRate=msaf.Anal.sample_rate)()

    # Output features dict
    features = {}

    # Compute framesync features
    features["mfcc"], features["hpcp"], features["tonnetz"] = \
        compute_features(audio)

    # Estimate Beats
    features["beats"], features["beats_conf"] = compute_beats(audio)

    # Compute Beat-sync features
    features["bs_mfcc"], features["bs_hpcp"], features["bs_tonnetz"] = \
        compute_features(audio, features["beats"])

    # Analysis parameters
    features["anal"] = {}
    features["anal"]["frame_rate"] = msaf.Anal.frame_rate
    features["anal"]["hop_size"] = msaf.Anal.hop_size
    features["anal"]["mfcc_coeff"] = msaf.Anal.mfcc_coeff
    features["anal"]["sample_rate"] = msaf.Anal.sample_rate
    features["anal"]["window_type"] = msaf.Anal.window_type
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
        logging.info("Saving Beats as an audio file")
        marker = ES.AudioOnsetsMarker(onsets=features["beats"], type='beep',
                                      sampleRate=msaf.Anal.sample_rate)
        marked_audio = marker(audio)
        ES.MonoWriter(filename='beats.wav',
                      sampleRate=msaf.Anal.sample_rate)(marked_audio)

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
            annot_beats = essentia.array(np.unique(annot_beats).tolist())
            annot_mfcc, annot_hpcp, annot_tonnetz = compute_features(
                audio, annot_beats)

    # Save output as json file
    logging.info("Saving the JSON file in %s" % out_file)
    yaml = YamlOutput(filename=out_file, format='json')
    pool = essentia.Pool()
    pool.add("beats.times", features["beats"])
    pool.add("beats.confidence", features["beats_conf"])
    pool.set("analysis.sample_rate", msaf.Anal.sample_rate)
    pool.set("analysis.frame_rate", msaf.Anal.frame_size)
    pool.set("analysis.hop_size", msaf.Anal.hop_size)
    pool.set("analysis.window_type", msaf.Anal.window_type)
    pool.set("analysis.mfcc_coeff", msaf.Anal.mfcc_coeff)
    pool.set("timestamp",
             datetime.datetime.today().strftime("%Y/%m/%d %H:%M:%S"))
    save_features("framesync", pool, features["mfcc"], features["hpcp"],
                  features["tonnetz"])
    save_features("est_beatsync", pool, features["bs_mfcc"],
                  features["bs_hpcp"], features["bs_tonnetz"])
    if os.path.isfile(file_struct.ref_file) and jam.beats != []:
        save_features("ann_beatsync", pool, annot_mfcc, annot_hpcp,
                      annot_tonnetz)
    yaml(pool)


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
