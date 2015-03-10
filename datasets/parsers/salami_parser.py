#!/usr/bin/env python
"""
 TODO: Write me, I'm a lonely docstring!
"""

__author__ = "Oriol Nieto"
__copyright__ = "Copyright 2014, Music and Audio Research Lab (MARL)"
__license__ = "MIT"
__version__ = "1.0"
__email__ = "oriol@nyu.edu"

import argparse
import csv
from msaf import jams2
import json
import logging
import os
import time
import pandas
import glob


def parse_annotation_level(annot, path, annotation_id, level):
    """Parses one specific level of annotation and puts it into annot.

    Parameters
    ----------

    annot: Annotation
    path: str
        path to the track in the SALAMI dataset
    annotation_id: int
        Whether to use the first or the second annotation
    level: str
        Level of annotation
    """
    level_dict = {
        "function": "_functions",
        #"large_scale": "_uppercase",
        #"small_scale": "_lowercase",
        "large_scale": "_large",
        "small_scale": "_small"
    }

    # File to open
    file_path = os.path.join(path, "parsed",
        "textfile" + str(annotation_id + 1) + level_dict[level] + ".lab")

    # Open file
    try:
        data = pandas.read_csv(file_path, sep="\t", header=None)
    except IOError:
        logging.warning("Annotation doesn't exist: %s", file_path)
        return

    # Parse file
    for row in data.iterrows():
        interval_data = row[1]
        start_time = interval_data[0]
        end_time = interval_data[1]
        label = interval_data[2]
        section = annot.create_datapoint()
        section.start.value = float(start_time)
        section.start.confidence = 0.0
        section.end.value = float(end_time)
        section.end.confidence = 0.0
        section.label.value = label
        section.label.confidence = 0.0
        section.label.context = level


def fill_global_metadata(jam, metadata):
    """Fills the global metada into the JAMS jam."""
    if metadata[5] == "":
        metadata[5] = -1
    jam.metadata.artist = metadata[8]
    jam.metadata.duration = float(metadata[5])  # In seconds
    jam.metadata.title = metadata[7]

    # TODO: extra info
    #jam.metadata.genre = metadata[14]


def fill_annotation(path, annot, annotation_id, metadata):
    """Fills the JAMS annot annotation given a path to the original
    SALAMI annotations. The variable "annotator" let's you choose which
    SALAMI annotation to use.

    Parameters
    ----------

    annotation_id: int
        0 or 1 depending on which annotation to use

    """

    # Annotation Metadata
    annot.annotation_metadata.attribute = "sections"
    annot.annotation_metadata.corpus = "SALAMI"
    annot.annotation_metadata.version = "1.2"
    annot.annotation_metadata.annotation_tools = "Sonic Visualizer"
    annot.annotation_metadata.annotation_rules = "TODO"  # TODO
    annot.annotation_metadata.validation_and_reliability = "TODO"  # TODO
    annot.annotation_metadata.origin = metadata[1]
    annot.annotation_metadata.annotator.name = metadata[annotation_id + 2]
    annot.annotation_metadata.annotator.email = "TODO"  # TODO
    #TODO:
    #time = metadata[annotation_id + 15]

    # Parse all level annotations
    levels = ["function", "large_scale", "small_scale"]
    [parse_annotation_level(annot, path, annotation_id, level)
        for level in levels]


def create_JAMS(in_dir, metadata, out_file):
    """Creates a JAMS file given the path to a SALAMI track."""
    path = os.path.join(in_dir, "data_cleaned", metadata[0])

    # Sanity check
    if not os.path.exists(path):
        logging.warning("Path not found %s", path)
        return

    # New JAMS and annotation
    jam = jams2.Jams()

    # Global file metadata
    fill_global_metadata(jam, metadata)

    # Create Annotations if they exist
    # Maximum 3 annotations per file
    for possible_annot in xrange(3):
        if os.path.isfile(os.path.join(path,
                "textfile" + str(possible_annot + 1) + ".txt")):
            annot = jam.sections.create_annotation()
            fill_annotation(path, annot, possible_annot, metadata)

    # Save JAMS
    with open(out_file, "w") as f:
        json.dump(jam, f, indent=2)


def process(in_dir, out_dir):
    """Converts the original SALAMI files into the JAMS format, and saves
    them in the out_dir folder."""

    # Check if output folder and create it if needed:
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Open CSV with metadata
    fh = open(os.path.join(in_dir, "metadata.csv"))
    csv_reader = csv.reader(fh)

    for metadata in csv_reader:
        # Create a JAMS file for this track
        create_JAMS(in_dir, metadata,
                    os.path.join(out_dir,
                        os.path.basename(metadata[0]) + ".jams"))
    # Close metadata
    fh.close()


def main():
    """Main function to convert the dataset into JAMS."""
    parser = argparse.ArgumentParser(description=
        "Converts the SALAMI dataset to the JAMS format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("in_dir",
                        action="store",
                        help="SALAMI main folder")
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
