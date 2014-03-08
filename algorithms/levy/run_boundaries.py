#!/usr/bin/env python
'''Runs the Levy segmenter for boundaries across the Segmentation dataset

'''

__author__ = "Oriol Nieto"
__copyright__ = "Copyright 2014, Music and Audio Research Lab (MARL)"
__license__ = "GPL"
__version__ = "1.0"
__email__ = "oriol@nyu.edu"

import sys
import glob
import os
import argparse
import json
import numpy as np
import time
import logging
import datetime
import jams
import subprocess

import sys
sys.path.append( "../../" )
import msaf_io as MSAF


def process(in_path, annot_beats=False):
    """Main process."""

    # Get relevant files
    feat_files = glob.glob(os.path.join(in_path, "features", "*.json"))
    jam_files = glob.glob(os.path.join(in_path, "annotations", "*.jams"))

    for feat_file, jam_file in zip(feat_files, jam_files):

        # Only analize files with annotated beats
        if annot_beats:
            jam = jams.load(jam_file)
            if jam.beats == []:
                continue
            if jam.beats[0].data == []:
                continue 

        if annot_beats:
            annot_beats_str = "1"
        else:
            annot_beats_str = "0"

        logging.info("Segmenting %s" % feat_file)

        # Levy segmenter call
        cmd = ["./segmenter", feat_file.replace(" ", "\ "), annot_beats_str]
        print cmd
        subprocess.call(cmd)


def main():
    """Main function to parse the arguments and call the main process."""
    parser = argparse.ArgumentParser(description=
        "Runs the Levy segmenter across a the Segmentation dataset and "\
            "stores the results in the estimations folder",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("in_path",
                        action="store",
                        help="Input dataset")
    parser.add_argument("-b", 
                        action="store_true", 
                        dest="annot_beats",
                        help="Use annotated beats",
                        default=False)
    args = parser.parse_args()
    start_time = time.time()
    
    # Setup the logger
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s', 
        level=logging.INFO)

    # Run the algorithm
    process(args.in_path, annot_beats=args.annot_beats)

    # Done!
    logging.info("Done! Took %.2f seconds." % (time.time() - start_time))


if __name__ == '__main__':
    main()
