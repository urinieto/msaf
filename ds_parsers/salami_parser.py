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

def fill_global_metadata(jam, metadata):
    """Fills the global metada into the JAMS jam."""
    jam.metadata.artist = metadata[8]
    jam.metadata.duration = metadata[5] # In seconds
    jam.metadata.title = metadata[7]

    # TODO: extra info
    #jam.metadata.genre = metadata[14]

def fill_annotation(path, annot, annotator):
    """Fills the JAMS annot annotation given a path to the original 
    SALAMI annotations. The variable "annotator" let's you choose which
    SALAMI annotation to use."""

    pass

def create_JAMS(in_dir, metadata, out_file):
    """Creates a JAMS file given the path to a SALAMI track."""
    path = os.path.join(in_dir, "data", metadata[0])

    # Sanity check
    if not os.path.exists(path):
        logging.warning("Path not found %s", path)
        return

    # New JAMS
    jam = jams.Jams()

    # Global file metadata
    update_global_metadata(jam, metadata)

    # Create Annotation 1
    annot1 = jam.sections.create_annotation()

    # Create Annotation 2
    annot2 = jam.sections.create_annotation()



def process(in_dir, out_dir):
    """Converts the original SALAMI files into the JAMS format, and saves
    them un the out_dir folder."""

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