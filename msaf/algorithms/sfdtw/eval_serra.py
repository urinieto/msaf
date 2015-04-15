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
import glob
import json
import logging
import os
import pylab as plt
import numpy as np
import time
import sys
import sqlite3

import mir_eval
import jams

import sys
sys.path.append( "../../" )
import msaf_io as MSAF



def process(in_path, win=3, trim=False):
    """Main process."""

    # Get files
    est_files = glob.glob(os.path.join("./", "smga1", "*.lab"))

    logging.info("Evaluating %d tracks..." % len(est_files))

    # Compute features for each file
    PRF = []  # Results: Precision, Recall, F-measure
    for est_file in est_files:

        jam_file = os.path.join(in_path, "annotations", 
                    "Isophonics_" + os.path.basename(est_file)[:-4] + ".jams")

        assert "Isophonics_" + os.path.basename(est_file)[:-4] == \
            os.path.basename(jam_file)[:-5]

        try:
            ann_times, ann_labels = mir_eval.input_output.load_jams_range(
                                    jam_file, 
                                    "sections", context="function")
        except:
            logging.warning("No annotations for file: %s" % jam_file)
            continue

        est_times, est_labels = mir_eval.input_output.load_annotation(est_file)
        if est_times == []: continue

        P, R, F = mir_eval.segment.boundary_detection(ann_times, est_times, 
                                                window=win, trim=trim)
        
        PRF.append([P,R,F])

    PRF = np.asarray(PRF)
    res = 100*PRF.mean(axis=0)
    logging.info("F: %.2f, P: %.2f, R: %.2f" % (res[2], res[0], res[1]))
    logging.info("%d tracks analized" % PRF.shape[0])


def main():
    """Main function to parse the arguments and call the main process."""
    parser = argparse.ArgumentParser(description=
        "Evaluates the estimated results of the Segmentation dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("in_path",
                        action="store",
                        help="Input dataset")
    parser.add_argument("-s",
                        action="store",
                        dest="win",
                        default=3,
                        type=float,
                        help="Time window in seconds")
    parser.add_argument("-t", 
                        action="store_true", 
                        dest="trim",
                        help="Trim the first and last boundaries",
                        default=False)
    args = parser.parse_args()
    start_time = time.time()
   
    # Setup the logger
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s', 
        level=logging.INFO)

    # Run the algorithm
    process(args.in_path, args.win, trim=args.trim)

    # Done!
    logging.info("Done! Took %.2f seconds." % (time.time() - start_time))

if __name__ == '__main__':
    main()