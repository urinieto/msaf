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

import utils
import mir_eval
import jams

import msaf_io as MSAF


def process(in_path, alg_id, ds_name="*", annot_beats=False, win=3, **params):
    """Main process."""

    # The Beatles hack
    beatles = False
    if ds_name == "Beatles":
        beatles = True
        ds_name = "Isophonics"

    # The SALAMI internet hack
    salamii = False
    if ds_name == "SALAMI-i":
        salamii = True
        ds_name = "SALAMI"

    # Get files
    jam_files = glob.glob(os.path.join(in_path, "annotations", 
                                            "%s_*.jams" % ds_name))
    est_files = glob.glob(os.path.join(in_path, "estimations", 
                                            "%s_*.json" % ds_name))

    prefix_dict = {
        "Cerulean"      : "large_scale",
        "Epiphyte"      : "function",
        "Isophonics"    : "function",
        "SALAMI"        : "large_scale"
    }

    logging.info("Evaluating %d tracks..." % len(jam_files))

    # Compute features for each file
    PRF = []  # Results: Precision, Recall, F-measure
    #for jam_file, est_file in zip(jam_files, est_files):
    for est_file in est_files:

        # Get corresponding estimation files
        idx = [i for i, s in enumerate(jam_files) if \
                os.path.basename(est_file)[:-5] in s][0]
        jam_file = jam_files[idx]

        assert os.path.basename(est_file)[:-5] == \
            os.path.basename(jam_file)[:-5]

        ds_prefix = os.path.basename(est_file).split("_")[0]
        try:
            ann_times, ann_labels = mir_eval.input_output.load_jams_range(
                                    jam_file, 
                                    "sections", context=prefix_dict[ds_prefix])
        except:
            logging.warning("No annotations for file: %s" % jam_file)
            continue

        if beatles:
            jam = jams.load(jam_file)
            if jam.metadata.artist != "The Beatles":
                continue

        # if ann_times[0][0] != 0:
        #     ann_times.insert(0, 0)

        if salamii:
            num = int(os.path.basename(jam_file).split("_")[1].split(".")[0])
            if num < 956 or num > 1498:
                continue

        est_times = MSAF.read_boundaries(est_file, alg_id, annot_beats, **params)
        if est_times == []: continue

        # if est_times[0] != 0:
        #     est_times = np.concatenate(([0], est_times))

        P, R, F = mir_eval.segment.boundary_detection(ann_times, est_times, 
                                                        window=win, trim=False)
        
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
    parser.add_argument("alg_id",
                        action="store",
                        help="Algorithm identifier (e.g. olda, siplca)")
    parser.add_argument("-d",
                        action="store",
                        dest="ds_name",
                        default="*",
                        help="The prefix of the dataset to use "\
                            "(e.g. Isophonics, SALAMI")
    parser.add_argument("-b", 
                        action="store_true", 
                        dest="annot_beats",
                        help="Use annotated beats",
                        default=False)
    parser.add_argument("-s",
                        action="store",
                        dest="win",
                        default=3,
                        type=float,
                        help="Time window in seconds")
    parser.add_argument("-f",
                        action="store",
                        dest="feature",
                        default="",
                        type=str,
                        help="Type of features (e.g. mfcc, hpcp")
    args = parser.parse_args()
    start_time = time.time()
   
    # Setup the logger
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s', 
        level=logging.INFO)

    # Run the algorithm
    process(args.in_path, args.alg_id, args.ds_name, args.annot_beats,
                    args.win, feature=args.feature)

    # Done!
    logging.info("Done! Took %.2f seconds." % (time.time() - start_time))

if __name__ == '__main__':
    main()