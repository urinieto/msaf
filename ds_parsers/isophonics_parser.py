#!/usr/bin/env python
"""
 TODO: Write me, I'm a lonely docstring!
"""

__author__ = "Oriol Nieto"
__copyright__ = "Copyright 2014, Music and Audio Research Lab (MARL)"
__license__ = "GPL"
__version__ = "1.0"
__email__ = "oriol@nyu.edu"

import argparse
import csv
import glob
import jams
import json
import logging
import os
import sys
import time

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
        "large_scale": "_uppercase",
        "small_scale": "_lowercase"
    }

    # File to open
    file_path = os.path.join(path, "parsed", 
        "textfile" + str(annotation_id+1) + level_dict[level] + ".txt")

    # Open file
    try:
        f = open(file_path, "r")
    except IOError:
        logging.warning("Annotation doesn't exist: %s", file_path)
        return

    # Parse file
    lines = f.readlines()
    for i, line in enumerate(lines[:-1]):
        start_time, label = line.strip("\n").split("\t")
        end_time = lines[i+1].split("\t")[0]
        section = annot.create_datapoint()
        section.start.value = float(start_time)
        section.start.confidence = 1.0
        section.end.value = float(end_time)
        section.end.confidence = 1.0
        section.label.value = label
        section.label.confidence = 1.0
        section.level = level
        #print start_time, end_time, label

    f.close()


def fill_global_metadata(jam, lab_file):
    """Fills the global metada into the JAMS jam."""
    jam.metadata.artist = lab_file.split("/")[-3]
    jam.metadata.duration = -1 # In seconds
    jam.metadata.title = os.path.basename(lab_file).strip(".lab")

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
    annot.annotation_metadata.annotation_rules = "TODO" #TODO
    annot.annotation_metadata.validation_and_reliability = "TODO" #TODO
    annot.annotation_metadata.origin = metadata[1]
    annot.annotation_metadata.annotator.name = metadata[annotation_id + 2]
    annot.annotation_metadata.annotator.email = "TODO" #TODO
    #TODO:
    #time = metadata[annotation_id + 15]

    # Parse all level annotations
    levels = ["function", "large_scale", "small_scale"]
    [parse_annotation_level(annot, path, annotation_id, level) \
        for level in levels]


def create_JAMS(lab_file, out_file):
    """Creates a JAMS file given the Isophonics lab file."""

    # New JAMS and annotation
    jam = jams.Jams()

    print lab_file.split("/")

    # # Global file metadata
    fill_global_metadata(jam, lab_file)

    # # Create Annotations if they exist
    # # Maximum 3 annotations per file
    # for possible_annot in xrange(3):
    #     if os.path.isfile(os.path.join(path, 
    #             "textfile" + str(possible_annot+1) + ".txt")):
    #         annot = jam.sections.create_annotation()
    #         fill_annotation(path, annot, possible_annot, metadata)

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

    # Get all the higher level folders
    folders = glob.glob(os.path.join(in_dir, "*"))
    for folder in folders:
        if not os.path.isdir(folder): continue

        # Get all the subfolders (where the lab files are)
        subfolders = glob.glob(os.path.join(folder, "*"))
        for subfolder in subfolders:
            if not os.path.isdir(subfolder): continue
            if os.path.basename(subfolder) == "audio": continue

            # Get all the lab files
            lab_files = glob.glob(os.path.join(subfolder, "*.lab"))
            for lab_file in lab_files:

                #Create a JAMS file for this track
                create_JAMS(lab_file, 
                            os.path.join(out_dir, 
                                "Isophonics_" + os.path.basename(lab_file) + \
                                ".jams"))


def main():
    """Main function to convert the dataset into JAMS."""
    parser = argparse.ArgumentParser(description=
        "Converts the Isophonics dataset to the JAMS format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("in_dir",
                        action="store",
                        help="Isophonics main folder")
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