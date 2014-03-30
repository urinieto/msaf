#!/usr/bin/env python
"""
Evaluates the estimated results of the Segmentation dataset against the
ground truth (human annotated data).
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

import mir_eval
import jams

import msaf_io as MSAF


def print_results(PRF, window):
    """Print the results."""
    PRF = np.asarray(PRF)
    res = 100 * PRF.mean(axis=0)
    logging.info("Window: %.1f\tF: %.2f, P: %.2f, R: %.2f" % (window, res[2],
                                                              res[0], res[1]))


def compute_information_gain(ann_times, est_times, est_file, bins=10):
    """Computes the information gain of the est_file from the annotated times
    and the estimated times."""
    ann_times = np.concatenate((ann_times.flatten()[::2],
                                [ann_times[-1, -1]]))
    try:
        D = mir_eval.beat.information_gain(ann_times, est_times, bins=bins)
    except:
        logging.warning("Couldn't compute the Information Gain for file "
                        "%s" % est_file)
        D = 0
    return D


def save_results_ds(cursor, alg_id, PRF3, PRF05, D, annot_beats, trim,
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
    PRF3 = np.asarray(PRF3)
    PRF05 = np.asarray(PRF05)

    if track_id is not None:
        all_values = (track_id, PRF05[2], PRF05[0], PRF05[1], PRF3[2],
                      PRF3[0], PRF3[1], D, annot_beats, feature, "none", trim)
        table = "%s_bounds" % alg_id
        select_where = "track_id=?"
        select_values = (track_id, annot_beats, feature, trim)
    elif ds_name is not None:
        # Aggregate results
        PRF05 = np.mean(PRF05, axis=0)
        PRF3 = np.mean(PRF3, axis=0)
        D = np.mean(np.asarray(D))
        all_values = (alg_id, ds_name, PRF05[2], PRF05[0], PRF05[1], PRF3[2],
                      PRF3[0], PRF3[1], D, annot_beats, feature, "none", trim)
        table = "boundaries"
        select_where = "algo_id=? AND ds_name=?"
        select_values = (alg_id, ds_name,
                annot_beats, feature, trim)

    # Check if exists
    cursor.execute("SELECT * FROM %s WHERE %s AND annot_beat=? AND "
                   "feature=? AND trim=?" % (table, select_where),
                   select_values)

    # Insert new if it doesn't exist
    if cursor.fetchone() is None:
        questions = "?," * len(all_values)
        sql_cmd = "INSERT INTO %s VALUES (%s)" % (table, questions[:-1])
        cursor.execute(sql_cmd, all_values)
    else:
        # Update row
        evaluations = (PRF05[2], PRF05[0], PRF05[1], PRF3[2], PRF3[0],
                       PRF3[1], D)
        evaluations += select_values
        sql_cmd = "UPDATE %s SET F05=?, P05=?, R05=?, F3=?, " \
            "P3=?, R3=?, D=?  WHERE %s AND annot_beat=? AND " \
            "feature=? AND trim=?" % (table, select_where)
        cursor.execute(sql_cmd, evaluations)


def process(in_path, alg_id, ds_name="*", annot_beats=False,
            trim=False, mma=False, **params):
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
    conn.text_factory = str     # Fixes the 8-bit encoding string problem
    c = conn.cursor()

    logging.info("Evaluating %d tracks..." % len(jam_files))

    # Compute features for each file
    PRF3 = np.empty((0, 3))   # Results: Precision, Recall, F-measure 3 seconds
    PRF05 = np.empty((0, 3))  # Results: Precision, Recall,
                              # F-measure 0.5 seconds
    D = np.empty((0))         # Information Gain

    # Dataset results
    PRF3_ds = []
    PRF05_ds = []
    D_ds = []

    try:
        feature = params["feature"]
    except:
        feature = ""

    curr_ds = os.path.basename(est_files[0]).split("_")[0]

    for est_file in est_files:
        # Get corresponding annotation files
        idx = [i for i, s in enumerate(jam_files) if
               os.path.basename(est_file)[:-5] in s][0]
        jam_file = jam_files[idx]

        assert os.path.basename(est_file)[:-5] == \
            os.path.basename(jam_file)[:-5]

        ds_prefix = os.path.basename(est_file).split("_")[0]
        try:
            ann_times, ann_labels = mir_eval.input_output.load_jams_range(
                jam_file, "sections", context=MSAF.prefix_dict[ds_prefix])
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
        if est_times == []:
            continue

        P3, R3, F3 = mir_eval.segment.boundary_detection(ann_times, est_times,
                                                         window=3, trim=trim)
        P05, R05, F05 = mir_eval.segment.boundary_detection(ann_times,
                                    est_times, window=0.5, trim=trim)

        # Information gain
        D_ds.append(compute_information_gain(ann_times, est_times, est_file,
                                             bins=10))

        PRF3_ds.append([P3, R3, F3])
        PRF05_ds.append([P05, R05, F05])

        # Save Track Result to database
        save_results_ds(c, alg_id, PRF3_ds[-1], PRF05_ds[-1], D_ds[-1],
            annot_beats, trim, feature, track_id=os.path.basename(est_file))

        # Store dataset results if needed
        actual_ds_name = os.path.basename(est_file).split("_")[0]
        if curr_ds != actual_ds_name:
            save_results_ds(c, alg_id, PRF3_ds, PRF05_ds, D_ds, annot_beats,
                            trim, feature, ds_name=curr_ds)
            curr_ds = actual_ds_name
            # Add to global results
            PRF3 = np.concatenate((PRF3, np.asarray(PRF3_ds)))
            PRF05 = np.concatenate((PRF05, np.asarray(PRF05_ds)))
            D = np.concatenate((D, np.asarray(D_ds)))
            PRF3_ds = []
            PRF05_ds = []
            D_ds = []

    # Save and add last results
    save_results_ds(c, alg_id, PRF3_ds, PRF05_ds, D_ds, annot_beats, trim,
                    feature, ds_name=curr_ds)
    PRF3 = np.concatenate((PRF3, np.asarray(PRF3_ds)))
    PRF05 = np.concatenate((PRF05, np.asarray(PRF05_ds)))
    D = np.concatenate((D, np.asarray(D_ds)))

    # Save all results
    save_results_ds(c, alg_id, PRF3, PRF05, D, annot_beats, trim,
                    feature, ds_name="all")

    # Print results
    print_results(PRF3, 3)
    print_results(PRF05, 0.5)

    D = np.asarray(D)
    logging.info("Information gain: %.5f (of %d files)" % (D.mean(),
                                                           D.shape[0]))

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
    args = parser.parse_args()
    start_time = time.time()

    #import vimpdb; vimpdb.set_trace()

    # Setup the logger
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s',
                        level=logging.INFO)

    # Run the algorithm
    process(args.in_path, args.alg_id, args.ds_name, args.annot_beats,
            trim=args.trim, mma=args.mma, feature=args.feature)

    # Done!
    logging.info("Done! Took %.2f seconds." % (time.time() - start_time))

if __name__ == '__main__':
    main()
