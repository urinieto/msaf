#!/usr/bin/env python
"""
Saves all the Isophonics dataset into a jams. The structure of the Isophonics
dataset is:

/Isophonics
    /Artist Annotations
        /feature
            /Artist
                /Album
                    /lab (or text) files

Example:

/Isophonics
    /The Beatles Annotations
        /seglab
            /The Beatles
                /01_-_Please_Please_Me
                    /01_-_I_Saw_Her_Standing_There.lab
        /beat
            /The Beatles
                /01_-_Please_Please_Me
                    /01_-_I_Saw_Her_Standing_There.txt

To parse the entire dataset, you simply need the path to the Isophonics dataset
and an output folder.

Example:
./isohpnics_parser.py ~/datasets/Isophonics -o ~/datasets/Isophonics/outJAMS

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


def fill_global_metadata(jam, lab_file):
    """Fills the global metada into the JAMS jam."""
    jam.metadata.artist = lab_file.split("/")[-3]
    jam.metadata.duration = -1  # In seconds
    jam.metadata.title = os.path.basename(lab_file).replace(".lab", "")

    # TODO: extra info
    #jam.metadata.genre = metadata[14]


def fill_annoatation_metadata(annot, attribute):
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
    fill_annoatation_metadata(annot, "sections")

    # Open lab file
    try:
        f = open(lab_file, "r")
    except IOError:
        logging.warning("Annotation doesn't exist: %s", lab_file)
        return

    # Convert to JAMS
    lines = f.readlines()
    for line in lines:
        section_raw = line.strip("\n").split("\t")
        start_time = section_raw[0]
        end_time = section_raw[1]
        label = section_raw[3]
        section = annot.create_datapoint()
        section.start.value = float(start_time)
        section.start.confidence = 1.0
        section.end.value = float(end_time)
        section.end.confidence = 1.0
        section.label.value = label
        section.label.confidence = 1.0
        section.label.context = "function"  # Only function level of annotation
        if float(end_time) < float(start_time):
            logging.warning("Start time is after end time in file %s" %
                            lab_file)

    f.close()


def fill_beat_annotation(txt_file, annot):
    """Fills the JAMS annot annotation given a txt file."""

    # Annotation Metadata
    fill_annoatation_metadata(annot, "beats")

    # Open txt file
    try:
        f = open(txt_file, "r")
    except IOError:
        logging.warning("Annotation doesn't exist: %s", txt_file)
        return

    # Convert to JAMS
    lines = f.readlines()
    for line in lines:
        line = line.strip("\n")
        if " " in line:
            time = line.split(" ")[0]
            downbeat = line.split(" ")[-1]
        elif "\t" in line:
            time = line.split("\t")[0]
            downbeat = line.split("\t")[-1]
        beat = annot.create_datapoint()
        try:
            # Problem with 11_-_When_I_Get_Home (starting with upbeat)
            beat.time.value = float(time)
        except ValueError:
            beat.time.value = float(time[0])
        beat.time.confidence = 1.0
        try:
            beat.label.value = int(downbeat)
        except ValueError:
            beat.label.value = int(-1)
        beat.label.confidence = 1.0

    f.close()


def create_JAMS(lab_file, out_file, parse_beats=False):
    """Creates a JAMS file given the Isophonics lab file."""

    # New JAMS and annotation
    jam = jams2.Jams()

    # Global file metadata
    fill_global_metadata(jam, lab_file)

    # Create Section annotations
    annot = jam.sections.create_annotation()
    fill_section_annotation(lab_file, annot)

    # Create Beat annotations if needed
    if parse_beats:
        annot = jam.beats.create_annotation()
        txt_file = lab_file.replace("seglab", "beat").replace(".lab", ".txt")
        fill_beat_annotation(txt_file, annot)

    # TODO: Create Chord and Key annotations

    # Save JAMS
    with open(out_file, "w") as f:
        json.dump(jam, f, indent=2)


def process(in_dir, out_dir):
    """Converts the original Isophonic files into the JAMS format, and saves
    them in the out_dir folder."""

    # Check if output folder and create it if needed:
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Get all the higher level annotation folders
    annots_folder = glob.glob(os.path.join(in_dir, "*"))
    for annot_folder in annots_folder:
        if not os.path.isdir(annot_folder):
            continue
        if os.path.basename(annot_folder) == "jams":
            continue
        if os.path.basename(annot_folder) == "audio":
            continue

        # Check whether we need to parse the beats
        parse_beats = os.path.isdir(os.path.join(annot_folder, "beat"))

        # Step into the segments folder
        annot_folder = os.path.join(annot_folder, "seglab")

        # Step into the artist folder
        artist_folder = glob.glob(os.path.join(annot_folder, "*"))[0]

        # Get all the subfolders (where the lab/txt files are)
        album_folder = glob.glob(os.path.join(artist_folder, "*"))

        for subfolder in album_folder:
            if not os.path.isdir(subfolder):
                continue
            if os.path.basename(subfolder) == "audio":
                continue

            logging.info("Parsing album %s" % os.path.basename(subfolder))

            # Get all the lab files
            lab_files = glob.glob(os.path.join(subfolder, "*.lab"))
            for lab_file in lab_files:
                #Create a JAMS file for this track
                create_JAMS(lab_file,
                            os.path.join(out_dir,
                                os.path.basename(lab_file).replace(".lab", "")
                                         + ".jams"),
                            parse_beats)


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
