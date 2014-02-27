#!/usr/bin/env python
"""
Makes sure that the datasets that constitute the Segmentation Dataset are 
consistent with the names of the annotations and the names of their audio files.

More specifically, it makes sure that the JAMS have the same name as the
audio files.

e.g. 1:
    /Isophonics/audio/01 I Feel The Earth Move.mp3

    is consistent with

    /Isophonics/audio/01 I Feel The Earth Move.jams

e.g. 2:
    /Isophonics/audio/(01)  Carole King - I Feel The Earth Move.mp3

    is NOT consistent with

    /Isophonics/audio/01 I Feel The Earth Move.jams

This script modifies the name of the audio files in order to correspond with
the annotated JAMS (not the other way arround).
"""

__author__ = "Oriol Nieto"
__copyright__ = "Copyright 2014, Music and Audio Research Lab (MARL)"
__license__ = "GPL"
__version__ = "1.0"
__email__ = "oriol@nyu.edu"

import argparse
import glob
import logging
import os
import re
import shutil
import time

def split_name(name):
    """Split the string name based on a specific regular expression."""
    return re.split("_|-| |[0-9]|\(|\)", name)

def remove_empty(words):
    """Removes empty words."""
    return filter(lambda w: w != "", words)

def ensure_audio_names(audio_dir, jams_dir):
    """Makes sure that the names of the audio file corresponds with 
        their JAMS."""

    jam_files = glob.glob(os.path.join(jams_dir, "*.jams"))
    # Allow wav and/or mp3
    audio_files = glob.glob(os.path.join(audio_dir, "*.[wm][ap][v3]"))

    #print audio_dir, len(audio_files)

    for jam_file in jam_files:
        base_jam = os.path.basename(jam_file).replace(".jams","").lower()
        jam_words = split_name(base_jam)
        jam_words = remove_empty(jam_words)

        max_k = 0
        for audio_file in audio_files:
            base_audio = os.path.basename(audio_file)[:-4].lower()
            audio_words = split_name(base_audio)
            audio_words = remove_empty(audio_words)
            k = len(set(jam_words).intersection(audio_words))
            if k >= max_k:
                if k == max_k and k > 0:
                    if len(correct_audio_file) < len(audio_file):
                        continue
                correct_audio_file = audio_file
                max_k = k
        if base_jam == "CD2_-_08_-_Revolution_1".lower():
            correct_audio_file = "CD2_-_08_-_Revolution_1.wav"
        print os.path.basename(jam_file), "\t\t", os.path.basename(correct_audio_file)






def process(in_dir):
    """Modify the audio files in order to match the JAMS files if needed."""

    datasets = glob.glob(os.path.join(in_dir, "*"))
    for dataset in datasets:
        if not os.path.isdir(dataset): continue

        # Ignore some temporary data
        if os.path.basename(dataset) == "Beatles": continue
        if os.path.basename(dataset) == "Isophonics_audio": continue

        audio_dir = os.path.join(dataset, "audio")
        jams_dir = os.path.join(dataset, "jams")

        if os.path.basename(dataset) == "Isophonics":
            ensure_audio_names(audio_dir, jams_dir)


def main():
    """Main function to ensure the correct names of the original datasets."""
    parser = argparse.ArgumentParser(description=
        "Ensures that the original datasets have their correct " \
        "audio file names",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("in_dir",
                        action="store",
                        help="Folder to the original datasets")
    args = parser.parse_args()
    start_time = time.time()
   
    # Setup the logger
    logging.basicConfig(format='%(asctime)s: %(message)s', level=logging.INFO)

    # Run the parser
    process(args.in_dir)

    # Done!
    logging.info("Done! Took %.2f seconds." % (time.time() - start_time))

if __name__ == '__main__':
    main()