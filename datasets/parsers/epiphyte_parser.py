#!/usr/bin/env python
"""
Saves all the Epiphyte dataset into a jams. The structure of the Epiphyte
dataset is:

/Epiphyte
    /originalSegments
        /text files
    /originalBeats
        /text files


To parse the entire dataset, you simply need the path to the Epiphyte dataset
and an output folder.

"""

__author__ = "Oriol Nieto"
__copyright__ = "Copyright 2014, Music and Audio Research Lab (MARL)"
__license__ = "GPL"
__version__ = "1.0"
__email__ = "oriol@nyu.edu"

import argparse
import glob
import jams
import json
import logging
import os
import sys
import time


def fill_global_metadata(jam, txt_file):
    """Fills the global metada into the JAMS jam."""
    jam.metadata.artist = "" #TODO
    jam.metadata.duration = -1 # In seconds
    jam.metadata.title = \
        os.path.basename(txt_file).replace(".txt", "").split("_")[-1]
    jam.metadata.duration = -1 # In seconds
    jam.metadata.md5 = "" #TODO
    jam.metadata.echonest_id = "" #TODO
    jam.metadata.mbid = "" #TODO
    jam.metadata.version = -1 #TODO

    # TODO: extra info
    #jam.metadata.genre = metadata[14]

def fill_annoatation_metadata(annot, attribute):
    """Fills the annotation metadata."""
    annot.annotation_metadata.attribute = attribute
    annot.annotation_metadata.corpus = "Epiphyte"
    annot.annotation_metadata.version = "1.0"
    annot.annotation_metadata.annotation_tools = "Sonic Visualizer"
    annot.annotation_metadata.annotation_rules = "TODO" #TODO
    annot.annotation_metadata.validation_and_reliability = "TODO" #TODO
    annot.annotation_metadata.origin = "Epiphyte Corp"
    annot.annotation_metadata.annotator.name = "TODO"
    annot.annotation_metadata.annotator.email = "TODO" #TODO
    #TODO:
    #time = "TODO"

def fill_section_annotation(txt_file, annot):
    """Fills the JAMS annot annotation given a text file."""

    # Annotation Metadata
    fill_annoatation_metadata(annot, "sections")
    
    # Open txt file
    try:
        f = open(txt_file, "r")
    except IOError:
        logging.warning("Annotation doesn't exist: %s", txt_file)
        return

    # Convert to JAMS
    lines = f.readlines()
    for i, line in enumerate(lines[:-1]):
        start_time = line.strip("\n").split(" ")[0]
        end_time = lines[i+1].strip("\n").split(" ")[0]
        label = line.strip("\n").split(" ")[-1]
        section = annot.create_datapoint()
        section.start.value = float(start_time)
        section.start.confidence = 1.0
        section.end.value = float(end_time)
        section.end.confidence = 1.0
        section.label.value = label
        section.label.confidence = 1.0
        section.level = "function" # Only function level of annotation

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
        time = line.split("\t")[0]
        downbeat = line.split("\t")[1]
        beat = annot.create_datapoint()
        beat.time.value = float(time)
        beat.time.confidence = 1.0
        beat.label.value = int(downbeat)
        beat.label.confidence = 1.0

    f.close()


def create_JAMS(segment_file, beat_file, out_file):
    """Creates a JAMS file given the Epiphyte lab file."""

    # New JAMS and annotation
    jam = jams.Jams()

    # Global file metadata
    fill_global_metadata(jam, segment_file)

    # Create Section annotations
    annot = jam.sections.create_annotation()
    fill_section_annotation(segment_file, annot)

    annot = jam.beats.create_annotation()
    fill_beat_annotation(beat_file, annot)

    # Save JAMS
    f = open(out_file, "w")
    json.dump(jam, f, indent=2)
    f.close()


def process(in_dir, out_dir):
    """Converts the original Epiphyte files into the JAMS format, and saves
    them in the out_dir folder."""

    # Check if output folder and create it if needed:
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Get all the higher level annotation folders
    segment_files = glob.glob(os.path.join(in_dir, "originalSegments", "*.txt"))
    beat_files = glob.glob(os.path.join(in_dir, "originalBeats", "*.txt"))
    for i, [segment_file, beat_file] in enumerate(zip(segment_files, beat_files)):
        if i % 100 == 0:
            logging.info("Parsed %.1f%%...", 100*i/float(len(segment_files)))
        #Create a JAMS file for this track
        create_JAMS(segment_file, beat_file,
                    os.path.join(out_dir,  
                        os.path.basename(segment_file).replace(".txt","") + \
                            ".jams"))


def main():
    """Main function to convert the dataset into JAMS."""
    parser = argparse.ArgumentParser(description=
        "Converts the Epiphyte dataset to the JAMS format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("in_dir",
                        action="store",
                        help="Epiphyte main folder")
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