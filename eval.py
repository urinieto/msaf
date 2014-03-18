#!/usr/bin/env python
"""
Evaluates the estimated results of the Segmentation dataset
"""

__author__      = "Oriol Nieto"
__copyright__   = "Copyright 2014, Music and Audio Research Lab (MARL)"
__license__     = "GPL"
__version__     = "1.0"
__email__       = "oriol@nyu.edu"

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

import utils
import mir_eval
import jams

import msaf_io as MSAF


def print_results(PRF, window):
    """Print the results."""
    PRF = np.asarray(PRF)
    res = 100*PRF.mean(axis=0)
    logging.info("Window: %.1f\tF: %.2f, P: %.2f, R: %.2f" % (window, 
                                                    res[2], res[0], res[1]))

def save_one_result_ds(cursor, alg_id, file_name, PRF3, PRF05, annot_beats, trim, 
                    feature):
    """Saves one result for one track into the SQLite dataset. If result exist, 
    ignore it."""

    all_values = (file_name, PRF05[2], PRF05[0], PRF05[1], 
                PRF3[2], PRF3[0], PRF3[1], annot_beats, feature, "none", trim)
    select_values = (file_name, annot_beats, feature, trim)

    # Check if exists
    cursor.execute("SELECT * FROM %s_bounds WHERE track_id=? AND " \
            "annot_beat=? AND feature==? AND trim=?" % alg_id, select_values)

    # Save if it doesn't exist
    if cursor.fetchone() is None:
        sql_cmd = 'INSERT INTO %s_bounds VALUES (?, ?, ?, ?, ?, ?, ' \
                    '?, ?, ?, ?, ?)' % (alg_id)
        cursor.execute(sql_cmd, all_values)


def save_avg_results_ds(cursor, alg_id, PRF3, PRF05, annot_beats, trim, 
                    feature, est_files):
    """Saves the results into the SQLite dataset. If result exist, 
    ignore it."""

    datasets = ["Cerulean", "Epiphyte", "Isophonics", "SALAMI"]
    PRF3 = np.asarray(PRF3)
    PRF05 = np.asarray(PRF05)
    for dataset in datasets:
        idxs = np.asarray([i for i,f in enumerate(est_files) if \
                os.path.basename(f).split("_")[0] == dataset])
        print idxs
        PRF05_ds = np.mean(PRF05[idxs], axis=0)
        PRF3_ds = np.mean(PRF3[idxs], axis=0)
        all_values = (alg_id, dataset, PRF05_ds[2], PRF05_ds[0], PRF05_ds[1], 
                PRF3_ds[2], PRF3_ds[0], PRF3_ds[1], annot_beats, feature, 
                "none", trim)
        select_values = (alg_id, dataset, annot_beats, feature, trim)

        # Check if exists
        cursor.execute("SELECT * FROM boundaries WHERE algo_id=? AND " \
                "ds_name=? AND annot_beat=? AND feature==? AND trim=?",
                select_values)

        # Save if it doesn't exist
        if cursor.fetchone() is None:
            sql_cmd = 'INSERT INTO boundaries VALUES (?, ?, ?, ?, ?, ?, ' \
                        '?, ?, ?, ?, ?, ?)'
            cursor.execute(sql_cmd, all_values)

    

def process(in_path, alg_id, ds_name="*", annot_beats=False, win=3, 
                trim=False, **params):
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

    conn = sqlite3.connect("results/results.sqlite")
    conn.text_factory = str # Fixes the 8-bit encoding string problem
    c = conn.cursor()

    logging.info("Evaluating %d tracks..." % len(jam_files))

    # Compute features for each file
    PRF3 = []  # Results: Precision, Recall, F-measure 3 seconds
    PRF05 = []  # Results: Precision, Recall, F-measure 0.5 seconds

    try:
        feature = params["feature"]
    except:
        feature = ""

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
                                    jam_file, "sections", 
                                    context=MSAF.prefix_dict[ds_prefix])
        except:
            logging.warning("No annotations for file: %s" % jam_file)
            continue

        if beatles:
            jam = jams.load(jam_file)
            if jam.metadata.artist != "The Beatles":
                continue

        if salamii:
            num = int(os.path.basename(jam_file).split("_")[1].split(".")[0])
            if num < 956 or num > 1498:
                continue

        est_times = MSAF.read_boundaries(est_file, alg_id, annot_beats, 
                                         **params)
        if est_times == []: continue

        P3, R3, F3 = mir_eval.segment.boundary_detection(ann_times, est_times, 
                                                window=3, trim=trim)
        P05, R05, F05 = mir_eval.segment.boundary_detection(ann_times, 
                                    est_times, window=0.5, trim=trim)
        
        PRF3.append([P3,R3,F3])
        PRF05.append([P05,R05,F05])

        # Save Track Result to database
        save_one_result_ds(c, alg_id, os.path.basename(est_file), PRF05[-1], 
                        PRF3[-1], annot_beats, trim, feature)

    # Save average results
    save_avg_results_ds(c, alg_id, PRF3, PRF05, annot_beats, trim, 
                    feature, est_files)

    # Print results
    print_results(PRF3, 3)
    print_results(PRF05, 0.5)

    # Commit changes to database and close
    conn.commit()
    conn.close()
    
    logging.info("%d tracks analized" % len(PRF3))
        


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
    process(args.in_path, args.alg_id, args.ds_name, args.annot_beats,
                    args.win, trim=args.trim, feature=args.feature)

    # Done!
    logging.info("Done! Took %.2f seconds." % (time.time() - start_time))

if __name__ == '__main__':
    main()
