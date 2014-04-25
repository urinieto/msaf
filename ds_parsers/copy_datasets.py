#!/usr/bin/env python
"""
Copies the audio and annotations of the original datasets into the
Segmentation Dataset.

Additionally, it makes sure that the datasets that constitute the Segmentation
Dataset are consistent with the names of the annotations and the names of their
audio files.

More specifically, it modifies the audio name when copying it in order to match
the one of the annotation.

e.g. 1:
    /Isophonics/audio/01 I Feel The Earth Move.mp3

    is consistent with

    /Isophonics/jams/01 I Feel The Earth Move.jams

e.g. 2:
    /Isophonics/audio/(01)  Carole King - I Feel The Earth Move.mp3

    is NOT consistent with

    /Isophonics/jams/01 I Feel The Earth Move.jams

    so it will copy it as

    /Isophonics/audio/01 I Feel The Earth Move.mp3

It saves the audio files in:
    /output_folder/audio
And it saves the JAMS files in
    /output_folder/annotations

"output_folder" is taken as an argument.
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


def find_correct_name(audio_files, jam_file):
    """ Finds the correct audio file for a given jam file."""
    base_jam = os.path.basename(jam_file).replace(".jams", "").lower()
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

    # Little hack, oh uri, you bad
    if base_jam == "CD2_-_08_-_Revolution_1".lower():
        correct_audio_file = os.path.join(os.path.dirname(audio_files[0]),
                                "CD2_-_08_-_Revolution_1.wav")

    return correct_audio_file


def copy_data(audio_dir, jams_dir, out_dir, ds_name):
    """Copies audio and JAMS in the correct path and makes sure that the names
        of the audio file correspond with their JAMS."""

    # Check if output subfolders exist and create them if not
    final_audio_dir = os.path.join(out_dir, "audio")
    final_jam_dir = os.path.join(out_dir, "annotations")
    if not os.path.exists(final_audio_dir):
        os.makedirs(final_audio_dir)
    if not os.path.exists(final_jam_dir):
        os.makedirs(final_jam_dir)

    jam_files = glob.glob(os.path.join(jams_dir, "*.jams"))
    # Allow wav and/or mp3
    audio_files = glob.glob(os.path.join(audio_dir, "*.[wm][ap][v3]"))

    # Ensure audio name for each JAMS file
    for jam_file in jam_files:
        # Find best string match in audio for the Isophonics dataset
        if ds_name == "Isophonics":
            correct_audio_file = find_correct_name(audio_files, jam_file)
        elif ds_name == "SALAMI":
            correct_audio_file = os.path.join(audio_dir,
                os.path.basename(jam_file).replace(".jams", ""),
                "audio.mp3")
        else:
            correct_audio_file = os.path.join(audio_dir,
                os.path.basename(jam_file).replace(".jams", "") +
                audio_files[0][-4:])
            assert os.path.isfile(correct_audio_file)

        # Define final file paths
        final_audio_file = os.path.join(
            final_audio_dir, ds_name + "_" +
            os.path.basename(jam_file).replace(".jams", "") +
            os.path.basename(correct_audio_file)[-4:])
        final_jam_file = os.path.join(final_jam_dir,
                ds_name + "_" + os.path.basename(jam_file))

        print "origin jam:\t", jam_file
        print "final jam:\t", final_jam_file
        print "origin audio:\t", correct_audio_file
        print "final audio:\t", final_audio_file
        print "------"

        # Copy files
        shutil.copy(jam_file, final_jam_file)
        shutil.copy(correct_audio_file, final_audio_file)


def process(in_dir, out_dir):
    """Copies thre audio and annotations to the final dataset and
        modify the audio files in order to match the JAMS files if needed."""

    datasets = glob.glob(os.path.join(in_dir, "*"))
    for dataset in datasets:
        if not os.path.isdir(dataset):
            continue

        # Ignore some temporary data
        if os.path.basename(dataset) == "Beatles":
            continue
        if os.path.basename(dataset) == "Isophonics_audio":
            continue

        # Define dirs
        audio_dir = os.path.join(dataset, "audio")
        jams_dir = os.path.join(dataset, "jams")

        # Copy the data into the new Segmentation Dataset
        copy_data(audio_dir, jams_dir, out_dir, os.path.basename(dataset))


def main():
    """Main function to copy the audio and annotations to the Segmentation
        dataset."""
    parser = argparse.ArgumentParser(description=
        "Copies the audio and annotations of the original datasets and "
        "modifies the audio file names accordingly",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("in_dir",
                        action="store",
                        help="Folder to the original datasets")
    parser.add_argument("out_dir",
                        action="store",
                        help="Output Segmentation Dataset folder")
    args = parser.parse_args()
    start_time = time.time()

    # Setup the logger
    logging.basicConfig(format='%(asctime)s: %(message)s', level=logging.INFO)

    # Run the parser
    process(args.in_dir, args.out_dir)

    # Done!
    logging.info("Done! Took %.2f seconds." % (time.time() - start_time))

if __name__ == '__main__':
    main()
