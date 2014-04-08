#!/usr/bin/env python
"""
Converts an svl file into a JAMS annotation
"""

__author__      = "Oriol Nieto"
__copyright__   = "Copyright 2014, Music and Audio Research Lab (MARL)"
__license__     = "GPL"
__version__     = "1.0"
__email__       = "oriol@nyu.edu"

import argparse
import glob
import logging
import os
import numpy as np
import time
import sqlite3
import itertools

import mir_eval
import jams

import msaf_io as MSAF


def main():
    """Main function to convert the annotation."""
    parser = argparse.ArgumentParser(description=
        "Converst a Sonic Visualizer annotation into a JAMS.",
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
                        help="The prefix of the dataset to use "
                        "(e.g. Isophonics, SALAMI")
    parser.add_argument("-b",
                        action="store_true",
                        dest="annot_beats",
                        help="Use annotated beats",
                        default=False)
    parser.add_argument("-f",
                        action="store",
                        dest="feature",
                        default="",
                        type=str,
                        help="Type of features (e.g. mfcc, hpcp")
    parser.add_argument("-t",
                        action="store_true",
                        dest="trim",
                        help="Trim the first and last boundaries",
                        default=False)
    parser.add_argument("-m",
                        action="store_true",
                        dest="mma",
                        help="Compute the mean mutual agreement between,"
                        "results",
                        default=False)
    parser.add_argument("-a",
                        action="store_true",
                        dest="all",
                        help="Compute all the results.",
                        default=False)
    args = parser.parse_args()
    start_time = time.time()

    # Setup the logger
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s',
                        level=logging.INFO)

    if args.all:
        # 5 boundary algorithms
        process(args.in_path, "siplca", "*", args.annot_beats, trim=args.trim,
                mma=False, feature="")
        process(args.in_path, "olda", "*", args.annot_beats, trim=args.trim,
                mma=False, feature="")
        process(args.in_path, "serra", "*", args.annot_beats, trim=args.trim,
                mma=False, feature="mix")
        process(args.in_path, "levy", "*", args.annot_beats, trim=args.trim,
                mma=False, feature="hpcp")
        process(args.in_path, "levy", "*", args.annot_beats, trim=args.trim,
                mma=False, feature="mfcc")
        process(args.in_path, "foote", "*", args.annot_beats, trim=args.trim,
                mma=False, feature="hpcp")
        # MMA
        process(args.in_path, "", "*", args.annot_beats,
                trim=args.trim, mma=True, feature=args.feature)
    else:
        # Run the algorithm
        process(args.in_path, args.alg_id, args.ds_name, args.annot_beats,
                trim=args.trim, mma=args.mma, feature=args.feature)

    # Done!
    logging.info("Done! Took %.2f seconds." % (time.time() - start_time))

if __name__ == '__main__':
    main()
