#!/usr/bin/env python
"""
Saves all the Cerulean dataset into jams. The structure of the Cerulean
dataset is:

/Cerulean
    /annotator
        /gt
            /json files

Only section annotations are available.

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


def fill_global_metadata(jam, json_file):
    """Fills the global metada into the JAMS jam."""
    # Open JSON
    f = open(json_file, "r")
    annot = json.load(f)
    basename = os.path.basename(json_file).replace(".json", "")

    if "-" in basename:
        token = "-"
    elif "," in basename:
        token = ","

    # Assign Metadata
    jam.metadata.artist = basename.split(token)[0]
    jam.metadata.duration = -1  # In seconds
    jam.metadata.title = basename.split(token)[1]
    jam.metadata.md5 = ""  # TODO
    jam.metadata.echonest_id = annot["TRID"]
    jam.metadata.mbid = ""  # TODO
    jam.metadata.version = -1  # TODO

    f.close()


def fill_annoatation_metadata(annot, attribute, name):
    """Fills the annotation metadata."""
    annot.annotation_metadata.attribute = attribute
    annot.annotation_metadata.corpus = "Cerulean"
    annot.annotation_metadata.version = "1.0"
    annot.annotation_metadata.annotation_tools = "Sonic Visualizer"
    annot.annotation_metadata.annotation_rules = "TODO"  # TODO
    annot.annotation_metadata.validation_and_reliability = "TODO"  # TODO
    annot.annotation_metadata.origin = "Cerulean Mountain Trust"
    annot.annotation_metadata.annotator.name = name
    annot.annotation_metadata.annotator.email = "TODO"  # TODO
    #TODO:
    #time = "TODO"


def fill_section_annotation(json_file, annot):
    """Fills the JAMS annot annotation given a json file."""

    # Annotation Metadata
    fill_annoatation_metadata(annot, "sections", json_file.split("/")[-3])

    # Open JSON
    f = open(json_file, "r")
    json_annot = json.load(f)

    # Convert to JAMS
    for i, json_section in enumerate(json_annot["sections"][:-1]):
        start_time = json_section["start"]
        end_time = json_annot["sections"][i + 1]["start"]
        label = json_section["label"]
        function_label = json_section["opt"]

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

    f.close()


def create_JAMS(json_files, out_file):
    """Creates a JAMS file given the Cerulean json files."""

    # New JAMS and annotation
    jam = jams2.Jams()

    # Global file metadata
    fill_global_metadata(jam, json_files[0])

    # Create Section annotations
    for json_file in json_files:
        annot = jam.sections.create_annotation()
        fill_section_annotation(json_file, annot)

    # Save JAMS
    f = open(out_file, "w")
    json.dump(jam, f, indent=2)
    f.close()


def process(in_dir, out_dir):
    """Converts the original Isophonic files into the JAMS format, and saves
    them in the out_dir folder."""

    # Check if output folder and create it if needed:
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Two annotators in Cerulean dataset
    annot1_path = os.path.join(in_dir, "charlie", "gt")
    annot2_path = os.path.join(in_dir, "sam", "gt")

    # Get Ground Truth JSON files
    annot1_files = glob.glob(os.path.join(annot1_path, "*.json"))
    annot2_files = glob.glob(os.path.join(annot2_path, "*.json"))

    for json_file1, json_file2 in zip(annot1_files, annot2_files):
        assert os.path.basename(json_file1) == os.path.basename(json_file2)

        #Create a JAMS file for this track
        create_JAMS([json_file1, json_file2],
                    os.path.join(out_dir,
                        os.path.basename(json_file1).replace(".json", "") +
                                 ".jams"))


def main():
    """Main function to convert the dataset into JAMS."""
    parser = argparse.ArgumentParser(description=
        "Converts the Cerulean dataset to the JAMS format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("in_dir",
                        action="store",
                        help="Cerulean main folder")
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
