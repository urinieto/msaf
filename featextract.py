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
import essentia
import essentia.standard as ES
from essentia.standard import YamlOutput
import glob
import jams
import logging
import os
import pylab as plt
import numpy as np
import time
import utils

from joblib import Parallel, delayed

# Setup main params
SAMPLE_RATE = 11025
FRAME_SIZE = 2048
HOP_SIZE = 512
WINDOW_TYPE = "blackmanharris74"

class STFTFeature:
    """Class to easily compute the features that require a frame based
        spectrum process (or STFT)."""
    def __init__(self, frame_size, hop_size, window_type, feature,
            beats, sample_rate):
        """STFTFeature constructor."""
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.window_type = window_type
        self.w = ES.Windowing(type=window_type)
        self.spectrum = ES.Spectrum()
        self.feature = feature # Essentia feature object
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

        if self.beats != []:
            framerate = self.sample_rate / float(self.hop_size)
            tframes = np.arange(features.shape[0]) / float(framerate)
            features = utils.resample_mx(features.T, tframes, self.beats).T

        return features


def double_beats(ticks):
    """Double the beats."""
    new_ticks = []
    for i, tick in enumerate(ticks[:-1]):
        new_ticks.append(tick)
        new_ticks.append((tick + ticks[i+1]) / 2.0)
    return essentia.array(new_ticks)


def compute_beats(audio):
    """Computes the beats using Essentia."""
    logging.info("Computing Beats...")
    conf = 1.0
    ticks, conf = ES.BeatTrackerMultiFeature()(audio)
    # ticks = ES.BeatTrackerDegara()(audio)
    ticks *= 44100 / SAMPLE_RATE # Essentia requires 44100 input (-.-')

    # Double the beats if found beats are too little
    th = 0.9 # 1 would equal to at least 1 beat per second
    while ticks.shape[0] / (audio.shape[0]/float(SAMPLE_RATE)) < th and \
            ticks.shape[0] > 2:
        ticks = double_beats(ticks)
    return ticks, conf


def compute_beatsync_features(ticks, audio):
    """Computes the HPCP and MFCC beat-synchronous features given a set
        of beats (ticks)."""
    MFCC = STFTFeature(FRAME_SIZE, HOP_SIZE, WINDOW_TYPE, ES.MFCC(numberCoefficients=14),
        ticks, SAMPLE_RATE)
    HPCP = STFTFeature(FRAME_SIZE, HOP_SIZE, WINDOW_TYPE, ES.HPCP(),
        ticks, SAMPLE_RATE)
    logging.info("Computing Beat-synchronous MFCCs...")
    mfcc = MFCC.compute_features(audio)
    logging.info("Computing Beat-synchronous HPCPs...")
    hpcp = HPCP.compute_features(audio)
    #plt.imshow(hpcp.T, interpolation="nearest", aspect="auto"); plt.show()

    return mfcc, hpcp


def compute_all_features(audio_file, audio_beats, overwrite):
    """Computes all the features for a specific audio file and its respective
        human annotations. It creates an audio file with the estimated
        beats if needed."""

    # Output file
    out_file = os.path.join(os.path.dirname(os.path.dirname(audio_file)),
                    "features", os.path.basename(audio_file)[:-4] + ".json")
    if os.path.isfile(out_file) and not overwrite:
        return # Do nothing, file already exist and we are not overwriting it

    # Load Audio
    logging.info("Loading audio file %s" % os.path.basename(audio_file))
    audio = ES.MonoLoader(filename=audio_file, sampleRate=SAMPLE_RATE)()

    # Estimate Beats
    ticks, conf = compute_beats(audio)
    ticks = np.concatenate(([0], ticks)) # Add first time
    ticks = essentia.array(np.unique(ticks))

    # Compute Beat-sync features
    mfcc, hpcp = compute_beatsync_features(ticks, audio)

    # Save output as audio file
    if audio_beats:
        logging.info("Saving Beats as an audio file")
        marker = ES.AudioOnsetsMarker(onsets=ticks, type='beep',
                                      sampleRate=SAMPLE_RATE)
        marked_audio = marker(audio)
        ES.MonoWriter(filename='beats.wav', sampleRate=SAMPLE_RATE)(marked_audio)

    # Save boundaries as an audio file if needed
    # if audio_beats:
    #     logging.info("Saving Boundaries as an audio file")
    #     bounds = []
    #     for data in jam.sections[0].data:
    #         bounds.append(data.start.value)
    #         bounds.append(data.end.value)
    #     bounds = essentia.array(np.unique(bounds))
    #     marker = ES.AudioOnsetsMarker(onsets=bounds, type='beep',
    #                                   sampleRate=SAMPLE_RATE)
    #     marked_audio = marker(audio)
    #     ES.MonoWriter(filename='bounds.wav', sampleRate=SAMPLE_RATE)(marked_audio)

    # Save beats as an audio file if needed
    if audio_beats:
        logging.info("Saving Beats as an audio file")
        marker = ES.AudioOnsetsMarker(onsets=annot_ticks, type='beep',
                                      sampleRate=SAMPLE_RATE)
        marked_audio = marker(audio)
        ES.MonoWriter(filename='beats.wav', sampleRate=SAMPLE_RATE)\
                                                            (marked_audio)

    # Read annotations if they exist in path/annotations/file.jams
    jam_file = os.path.join(os.path.dirname(os.path.dirname(audio_file)),
                 "annotations", os.path.basename(audio_file)[:-4] + ".jams")
    if os.path.isfile(jam_file):
        jam = jams.load(jam_file)

        # If beat annotations exist, compute also annotated beatsyn features
        if jam.beats != []:
            logging.info("Reading beat annotations from JAMS")
            annot = jam.beats[0]
            annot_ticks = []
            for data in annot.data:
                annot_ticks.append(data.time.value)
            annot_ticks = essentia.array(np.unique(annot_ticks).tolist())
            annot_mfcc, annot_hpcp = compute_beatsync_features(annot_ticks,
                                                                    audio)

    # Save output as json file
    logging.info("Saving the JSON file in %s" % out_file)
    yaml = YamlOutput(filename=out_file, format='json')
    pool = essentia.Pool()
    pool.add("beats.ticks", ticks)
    pool.add("beats.confidence", conf)
    [pool.add("est_beatsync.mfcc", essentia.array(mfcc_coeff)) \
        for mfcc_coeff in mfcc]
    [pool.add("est_beatsync.hpcp", essentia.array(hpcp_coeff)) \
        for hpcp_coeff in hpcp]
    pool.set("analysis.sample_rate", SAMPLE_RATE)
    pool.set("analysis.frame_rate", FRAME_SIZE)
    pool.set("analysis.hop_size", HOP_SIZE)
    pool.set("analysis.window_type", WINDOW_TYPE)
    if os.path.isfile(jam_file) and jam.beats != []:
        [pool.add("ann_beatsync.mfcc", essentia.array(mfcc_coeff)) \
            for mfcc_coeff in annot_mfcc]
        [pool.add("ann_beatsync.hpcp", essentia.array(hpcp_coeff)) \
            for hpcp_coeff in annot_hpcp]
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
        audio_files = glob.glob(os.path.join(in_path, "audio",
                                    "*.[wm][ap][v3]"))

        # Compute features using joblib
        Parallel(n_jobs=n_jobs)(delayed(compute_all_features)( \
                audio_file, audio_beats, overwrite) \
                for audio_file in audio_files)


def main():
    """Main function to parse the arguments and call the main process."""
    parser = argparse.ArgumentParser(description=
        "Extracts a set of features from the Segmentation dataset or a given " \
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
                        default=1)
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
