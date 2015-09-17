#!/usr/bin/env python
"""
Converts the jams2 files to the 0.2.0 version of JAMS.

jams2 was initially contained in MSAF and is deprecated and should not be
confused with 2.0 or above JAMS version (sorry for the confusion!).

If you're new to MSAF, you probably don't need this.
"""
from __future__ import print_function
import argparse
import glob
import json
import logging
import os
import time

import jams

__author__ = "Oriol Nieto"
__copyright__ = "Copyright 2015, Music and Audio Research Lab (MARL)"
__license__ = "GPL"
__version__ = "1.0"
__email__ = "oriol.nieto@gmail.com"


def get_contexts(section_ann):
    """Returns the unique contexts in the given section annotation."""
    contexts = []
    for data in section_ann["data"]:
        contexts.append(data["label"]["context"])
    return list(set(contexts))


def convert_am(am2, curator_name="", curator_email=""):
    """Converts the annotation metadata 2 and returns it."""
    am = jams.AnnotationMetadata()
    am.data_source = am2["origin"]
    am.annotation_rules = am2["annotation_rules"]
    am.validation = am2["validation_and_reliability"]
    am.annotation_tools = am2["annotation_tools"]
    am.annotator.name = am2["annotator"]["name"]
    am.version = am2["version"]
    am.corpus = am2["corpus"]
    am.curator.name = curator_name
    am.curator.email = curator_email
    return am


def convert_fm(fm2, jam):
    """Converts the file_metada and puts it into the new jams."""
    jam.file_metadata.title = fm2["title"]
    jam.file_metadata.artist = fm2["artist"]
    jam.file_metadata.duration = fm2["duration"]


def convert_sections(sections, jam):
    """Converts the given sections and puts them into the new jams."""
    for section_ann in sections:
        contexts = get_contexts(section_ann)
        for context in contexts:
            if context == "function":
                ns = "segment_open"
            elif context == "large_scale":
                ns = "segment_salami_upper"
            ann = jams.Annotation(namespace=ns)
            ann.annotation_metadata = \
                convert_am(section_ann["annotation_metadata"])

            # Add data
            for data in section_ann["data"]:
                if data["label"]["context"] != context:
                    continue
                dur = data["end"]["value"] - data["start"]["value"]

                # Hack to to convert to upper case
                value = data["label"]["value"]
                if context == "large_scale":
                    value = value.upper()

                # Append actual data point
                # TODO: confidence?
                ann.append(time=data["start"]["value"], duration=dur,
                           value=value)

            # Save annotation in JAMS
            jam.annotations.append(ann)


def convert_beats(beats, jam):
    """Converts the given beats and puts them into the new jams."""
    # TODO
    pass


def convert_JAMS(jams2_file):
    """Converts the given jams2 file into a jams file"""
    # Open jams2 file
    with open(jams2_file) as f:
        jams2_data = json.load(f)

    # Create new jams
    jam = jams.JAMS()

    # File Metadata
    convert_fm(jams2_data["metadata"], jam)

    # Convert the different tasks (only segment-related for now)
    convert_sections(jams2_data["sections"], jam)
    convert_beats(jams2_data["beats"], jam)

    return jam


def process(in_dir, out_dir):
    """Converts the jams2 files into the JAMS 0.2 format, and saves
    them in the out_dir folder."""

    # Check if output folder and create it if needed:
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Get all jams2 files
    jams2_files = glob.glob(os.path.join(in_dir, "*.jams"))

    # Convert files
    for jams2_file in jams2_files[:]:
        print("Converting %s..." % jams2_file)

        # Set output file
        jams_file = os.path.join(out_dir, os.path.basename(jams2_file))

        # Convert JAMS
        jam = convert_JAMS(jams2_file)

        # Save JAMS
        jam.save(jams_file)


def main():
    """Main function to convert the dataset into JAMS."""
    parser = argparse.ArgumentParser(description=
        "Converts old jams2 files to the JAMS 0.2 version.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("in_dir",
                        action="store",
                        help="Folder where jams2 files are found")
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
