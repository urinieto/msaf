#!/usr/bin/env python
"""
Enriches the JAMS of the Segmentation Dataset by doing the following:

- Setting duration field in JAMS
- Adding EchoNest ID
- Adding MD5
- Adding last segment to Cerulean dataset

"""

__author__ = "Oriol Nieto"
__copyright__ = "Copyright 2014, Music and Audio Research Lab (MARL)"
__license__ = "GPL"
__version__ = "1.0"
__email__ = "oriol@nyu.edu"

import argparse
import glob
import jams2
import json
import logging
import os
import time
import subprocess
from pyechonest import track

# Set Uri's Echo Nest API Key
from pyechonest import config
config.ECHO_NEST_API_KEY = "RRIGOW0BG5NJTM5M4"


def add_last_segment(jam, jam_file, duration):
    """Adds the last segment of the Cerulean dataset to the JAMS file."""
    dataset_path = os.path.dirname(os.path.dirname(jam_file))

    # Update both annotations
    for annot in jam.sections:
        # Read original JSON file
        annotator = annot.annotation_metadata.annotator.name
        original_json = os.path.join(dataset_path, "originalDatasets",
                        "Cerulean", annotator, "gt",
                        os.path.basename(jam_file).replace(".jams",
                                        ".json").replace("Cerulean_", ""))
        original_annot = json.load(open(original_json, "r"))

        # Only add last segment if there's enough duration difference
        if duration - annot.data[-1]["end"]["value"] > 1:
            start_time = annot.data[-1]["end"]["value"]
            end_time = duration
            label = original_annot["sections"][-1]["label"]
            function_label = original_annot["sections"][-1]["opt"]

            section = annot.create_datapoint()
            section.start.value = float(start_time)
            section.start.confidence = 1.0
            section.end.value = float(end_time)
            section.end.confidence = 1.0
            section.label.value = function_label
            section.label.confidence = 1.0
            section.label.context = "function"

            section = annot.create_datapoint()
            section.start.value = float(start_time)
            section.start.confidence = 1.0
            section.end.value = float(end_time)
            section.end.confidence = 1.0
            section.label.value = label
            section.label.confidence = 1.0
            section.label.context = "large_scale"


def enrich_jam(audio_file, jam_file):
    """Enriches the JAMS file with the audio file."""

    # Read JAMS and annotation
    jam = jams2.load(jam_file)

    # Read Echo Nest Info
    #while True:
        #try:
            #pytrack = track.track_from_filename(audio_file)
            #break
        #except:
            #logging.warning("Connection lost. Retrying in 5 seconds...")
            #time.sleep(5)

    ## Fill data
    #dur = pytrack.duration
    #jam.metadata.duration = float(pytrack.duration)
    #jam.metadata.md5 = pytrack.md5
    #jam.metadata.echonest_id = pytrack.id
    #try:
        #jam.metadata.artist = pytrack.artist
    #except AttributeError:
        #pass

    # Use sox instead of Echo Nest
    parsed_audio_file = audio_file.replace(",", "\,")
    parsed_audio_file = parsed_audio_file.replace(" ", "\ ")
    parsed_audio_file = parsed_audio_file.replace("&", "\&")
    parsed_audio_file = parsed_audio_file.replace("'", "\\'")
    parsed_audio_file = parsed_audio_file.replace(")", "\)")
    parsed_audio_file = parsed_audio_file.replace("(", "\(")
    cmd = "soxi -D " + parsed_audio_file
    dur = float(subprocess.check_output(cmd, shell=True))
    jam.metadata.duration = dur

    # Add the last segment
    if os.path.basename(audio_file)[:9] == "Cerulean_":
        add_last_segment(jam, jam_file, dur)

    # Save JAMS
    with open(jam_file, "w") as f:
        json.dump(jam, f, indent=2)


def process(in_dir):
    """Main process."""

    # Get Ground Truth JSON files
    audio_files = glob.glob(os.path.join(in_dir, "audio", "*.[wm][ap][v3]"))
    jam_files = glob.glob(os.path.join(in_dir, "annotations", "*.jams"))

    for audio_file, jam_file in zip(audio_files, jam_files):
        assert os.path.basename(audio_file)[:-4] == \
            os.path.basename(jam_file)[:-5], "Files names %s and %s must be "\
            "the same." % (os.path.basename(audio_file),
                          os.path.basename(jam_file))

        logging.info("Processing %s...", os.path.basename(jam_file))

        #Enrich JAMS with audio file
        enrich_jam(audio_file, jam_file)


def main():
    """Main function to enrich the JAMS of the Segmentation Dataset."""
    parser = argparse.ArgumentParser(description=
        "Enriches the JAMS of the Segmentation Dataset by reading "
        "audio information",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("in_dir",
                        action="store",
                        help="Segmentation main folder")
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
