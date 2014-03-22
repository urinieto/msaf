#!/usr/bin/env python
"""
Evaluates the estimated results of the Segmentation dataset using the
mean mutual agreement (MMA) method.
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
import itertools

import utils
import mir_eval
import jams

import msaf_io as MSAF

feat_dict = {
    'serra' :   'mix',
    'levy'  :   'hpcp',
    'foote' :   'hpcp',
    'siplca':   '',
    'olda'  :   ''
}

def print_results(results):
    """Print the results."""
    results = np.asarray(results)
    res = results.mean(axis=0)
    logging.info("F3: %.2f, P3: %2.f, R3: %2.f, F05: %.2f, P05: %.2f, " \
        "R05: %.2f, D: %.4f" % (100*res[2], 100*res[0], 100*res[1], 100*res[5], 
            100*res[3], 100*res[4], res[6])) 
        

def save_results_ds(cursor, alg_id, results, annot_beats, trim,
                    feature, track_id=None, ds_name=None):
    """Saves the results into the dataset.

    Parameters
    ==========

    TODO
    """
    # Sanity Check
    if track_id is None and ds_name is None:
        logging.error("You should pass at least a track id or a dataset name")
        return

    # Make sure that the results are stored in numpy arrays
    res = np.asarray(results)

    if track_id is not None:
        all_values = (track_id, res[5], res[3], res[4], res[2], res[0], res[1],
                                res[6], annot_beats, feature, "none", trim)
        table = "%s_bounds" % alg_id
        select_where = "track_id=?"
        select_values = (track_id, annot_beats, feature, trim)
    elif ds_name is not None:
        # Aggregate results
        res = np.mean(res, axis=0)
        all_values = (alg_id, ds_name, res[5], res[3], res[4], res[2], res[0], 
                        res[1], res[6], annot_beats, feature, "none", trim)
        table = "boundaries"
        select_where = "algo_id=? AND ds_name=?"
        select_values = (alg_id, ds_name, annot_beats, feature, trim)

    # Check if exists
    cursor.execute("SELECT * FROM %s WHERE %s AND annot_beat=? AND " \
            "feature=? AND trim=?"% (table, select_where), select_values)

    # Insert new if it doesn't exist
    if cursor.fetchone() is None:
        questions = "?," * len(all_values)
        sql_cmd = "INSERT INTO %s VALUES (%s)" % (table, questions[:-1])
        cursor.execute(sql_cmd, all_values)
    else:
        # Update row
        evaluations = (res[5], res[3], res[4], res[2], res[0], 
                        res[1], res[6])
        evaluations += select_values
        sql_cmd = "UPDATE %s SET F05=?, P05=?, R05=?, F3=?, " \
                    "P3=?, R3=?, D=?  WHERE %s AND annot_beat=? AND " \
                    "feature=? AND trim=?" % (table, select_where)
        cursor.execute(sql_cmd, evaluations)



def process(in_path, ds_name="*", annot_beats=False, trim=False, **params):
    """Main process."""

    # The SALAMI internet hack
    salamii = False
    if ds_name == "SALAMI-i":
        salamii = True
        ds_name = "SALAMI"

    # Get estimation files (each one contains the results of all the 
    # algorithms)
    est_files = glob.glob(os.path.join(in_path, "estimations",
                                            "%s_*.json" % ds_name))

    conn = sqlite3.connect("results/results.sqlite")
    conn.text_factory = str # Fixes the 8-bit encoding string problem
    c = conn.cursor()

    logging.info("Evaluating %d tracks..." % len(est_files))

    # Compute features for each file
    results = np.empty((0,7))   # Results: P3, R3, F3, P05, R05, F05, D

    # Dataset results
    results_ds = []
    alg_id = "mma"

    try:
        feature = params["feature"]
    except:
        feature = ""

    curr_ds = os.path.basename(est_files[0]).split("_")[0]

    for est_file in est_files:

        ds_prefix = os.path.basename(est_file).split("_")[0]

        if salamii:
            num = int(os.path.basename(est_file).split("_")[1].split(".")[0])
            if num < 956 or num > 1498:
                continue

        # Get all the permutations in order to compute the MMA
        results_mma = []
        for algorithms in itertools.permutations(MSAF.get_algo_ids(est_file)):
            print algorithms

            # Read estimated times from both algorithms
            est_times1 = MSAF.read_boundaries(est_file, algorithms[0], 
                                annot_beats, feature=feat_dict[algorithms[0]])
            est_times2 = MSAF.read_boundaries(est_file, algorithms[1], 
                                annot_beats, feature=feat_dict[algorithms[1]])
            if est_times1 == [] or est_times2 == []: continue

            # F-measures
            P3, R3, F3 = mir_eval.segment.boundary_detection(est_times1, 
                                est_times2, window=3, trim=trim)
            P05, R05, F05 = mir_eval.segment.boundary_detection(est_times1,
                                est_times2, window=0.5, trim=trim)

            # Information gain
            try:
                D = mir_eval.beat.information_gain(est_times1, 
                                                est_times2, bins=10)
            except:
                logging.warning("Couldn't compute the Information Gain for " \
                                                        "file %s" % est_file)
                D = 0
            # Store partial results
            results_mma.append([P3, R3, F3, P05, R05, F05, D])

        # Compute the averages
        results_ds.append(np.mean(np.asarray(results_mma), axis=0))

        # Save Track Result to database
        save_results_ds(c, alg_id, results_ds[-1],
            annot_beats, trim, feature, track_id=os.path.basename(est_file))

        # Store dataset results if needed
        actual_ds_name = os.path.basename(est_file).split("_")[0]
        if curr_ds != actual_ds_name:
            save_results_ds(c, alg_id, results_ds, annot_beats, 
                    trim, feature, ds_name=curr_ds)
            curr_ds = actual_ds_name
            # Add to global results
            results = np.concatenate((results, np.asarray(results_ds)))
            results_ds = []

    # Save and add last results
    save_results_ds(c, alg_id, results_ds, annot_beats, trim,
                    feature, ds_name=curr_ds)
    results = np.concatenate((results, np.asarray(results_ds)))

    # Save all results
    save_results_ds(c, alg_id, results, annot_beats, trim,
                    feature, ds_name="all")

    # Print results
    print_results(results)
    
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
    process(args.in_path, args.ds_name, args.annot_beats,
                    trim=args.trim, feature=args.feature)

    # Done!
    logging.info("Done! Took %.2f seconds." % (time.time() - start_time))

if __name__ == '__main__':
    main()
