#!/usr/bin/env python
"""
Saves all the TUT Beatles dataset into a jams. The structure of the TUT Beatles
dataset is:

/BeatlesTUT
    /Album
        /lab (or text) files

Example:

/TUT
    /01_-_Please_Please_Me
        /01_-_I_Saw_Her_Standing_There.lab

To parse the entire dataset, you simply need the path to the TUT Beatles dataset
and an output folder.

Example:
./tut_parser.py ~/datasets/BeatlesTUT -o ~/datasets/BeatlesTUT/outJAMS

"""

__author__ = "Oriol Nieto"
__copyright__ = "Copyright 2014, Music and Audio Research Lab (MARL)"
__license__ = "MIT"
__version__ = "1.0"
__email__ = "oriol@nyu.edu"

import argparse
import glob
import json
import logging
import os
import time

from msaf import jams2


def fill_global_metadata(jam, lab_file):
    """Fills the global metada into the JAMS jam."""
    jam.metadata.artist = "The Beatles"
    jam.metadata.duration = -1  # In seconds
    jam.metadata.title = os.path.basename(lab_file).replace(".lab", "")

    # TODO: extra info
    #jam.metadata.genre = metadata[14]


def fill_annotation_metadata(annot, attribute):
    """Fills the annotation metadata."""
    annot.annotation_metadata.attribute = attribute
    annot.annotation_metadata.corpus = "Isophonics"
    annot.annotation_metadata.version = "1.0"
    annot.annotation_metadata.annotation_tools = "Sonic Visualizer"
    annot.annotation_metadata.annotation_rules = "TODO"  # TODO
    annot.annotation_metadata.validation_and_reliability = "TODO"  # TODO
    annot.annotation_metadata.origin = "Centre for Digital Music"
    annot.annotation_metadata.annotator.name = "TODO"
    annot.annotation_metadata.annotator.email = "TODO"  # TODO
    #TODO:
    #time = "TODO"


def fill_section_annotation(lab_file, annot):
    """Fills the JAMS annot annotation given a lab file."""

    # Annotation Metadata
    fill_annotation_metadata(annot, "sections")

    # Open lab file
    try:
        f = open(lab_file, "r")
    except IOError:
        logging.warning("Annotation doesn't exist: %s", lab_file)
        return

    # Convert to JAMS
    lines = f.readlines()
    for line in lines:
        # Hacky stuff to accept inconsistencies in the dataset
        section_raw = line.strip("\n").split(" ")
        if len(section_raw) == 0 or section_raw[0] == '':
            continue
        start_time = section_raw[0]
        if '\t' in section_raw[1]:
            end_time = section_raw[1].split('\t')[0]
            label = section_raw[1].split('\t')[1]
        else:
            end_time = section_raw[1]
            label = section_raw[2]
        if float(end_time) <= float(start_time):
            logging.warning("Start time is after end time in file %s" %
                            lab_file)
            continue
        section = annot.create_datapoint()
        section.start.value = float(start_time)
        section.start.confidence = 1.0
        section.end.value = float(end_time)
        section.end.confidence = 1.0
        section.label.value = label
        section.label.confidence = 1.0
        section.label.context = "function"  # Only function level of annotation

    f.close()


def create_JAMS(lab_file, out_file, parse_beats=False):
    """Creates a JAMS file given the Isophonics lab file."""

    # New JAMS and annotation
    jam = jams2.Jams()

    # Global file metadata
    fill_global_metadata(jam, lab_file)

    logging.info("Parsing %s..." % lab_file)

    # Create Section annotations
    annot = jam.sections.create_annotation()
    fill_section_annotation(lab_file, annot)

    # Save JAMS
    with open(out_file, "w") as f:
        json.dump(jam, f, indent=2)


def process(in_dir, out_dir):
    """Converts the TUT Beatles files into the JAMS format, and saves
    them in the out_dir folder."""

    # Check if output folder and create it if needed:
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Get all the lab files
    lab_files = glob.glob(os.path.join(in_dir, "*", "*.lab"))

    for lab_file in lab_files:
        #Create a JAMS file for this track
        create_JAMS(lab_file,
                    os.path.join(out_dir,
                                 os.path.basename(lab_file).replace(".lab", "")
                                 + ".jams"))

    logging.info("Parsed %d files" % len(lab_files))


def main():
    """Main function to convert the dataset into JAMS."""
    parser = argparse.ArgumentParser(description=
        "Converts the TUT Beatles dataset to the JAMS format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("in_dir",
                        action="store",
                        help="TUT Beatles main folder")
    parser.add_argument("-o",
                        action="store",
                        dest="out_dir",
                        default="outJAMS",
                        help="Output JAMS folder")
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
