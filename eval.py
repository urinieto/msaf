#!/usr/bin/env python
"""
Evaluates the estimated results of the Segmentation dataset
"""

__author__ = "Oriol Nieto"
__copyright__ = "Copyright 2014, Music and Audio Research Lab (MARL)"
__license__ = "GPL"
__version__ = "1.0"
__email__ = "oriol@nyu.edu"

import argparse
import essentia
import essentia.standard as ES
from essentia.standard import YamlOutput
import glob
import jams
import logging
import os
import pylab as plt
import numpy as np
import time
import utils


def process(in_path, annot_beats=False):
    """Main process."""

    # Get files
    jam_files = glob.glob(os.path.join(in_path, "annotations", "*.jams"))
    audio_files = glob.glob(os.path.join(in_path, "audio", 
                                                "*.[wm][ap][v3]"))

    # Compute features for each file
    for jam_file, audio_file in zip(jam_files, audio_files):
        assert os.path.basename(audio_file)[:-4] == \
            os.path.basename(jam_file)[:-5]
        #TODO


def main():
    """Main function to parse the arguments and call the main process."""
    parser = argparse.ArgumentParser(description=
        "Evaluates the estimated results of the Segmentation dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("in_path",
                        action="store",
                        help="Input dataset dir or audio file")
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
    process(args.in_path, args.annot_beats)

    # Done!
    logging.info("Done! Took %.2f seconds." % (time.time() - start_time))

if __name__ == '__main__':
    main()