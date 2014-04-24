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
import itertools

import mir_eval
import jams2

import msaf_io as MSAF


feat_dict = {
    'serra' :   'mix',
    'levy'  :   'hpcp',
    'foote' :   'hpcp',
    'siplca':   '',
    'olda'  :   ''
}


def print_results(results):
    """Print all the results.

    Parameters
    ----------
    results: np.array(9)
        Results in the following format:
            0   :   Precision 3 seconds
            1   :   Recall 3 seconds
            2   :   F-measure 3 seconds
            3   :   Precision 0.5 seconds
            4   :   Recall 0.5 seconds
            5   :   F-measure 0.5 seconds
            6   :   Information Gain
            7   :   Median Deviation from Annotated to Estimated boundary
            8   :   Median Deviation from Estimated to Annotated boundary
    """
    results = np.asarray(results)
    res = results.mean(axis=0)
    logging.info("F3: %.2f, P3: %.2f, R3: %.2f, F05: %.2f, P05: %.2f, "
                 "R05: %.2f, D: %.4f, Ann2EstDev: %.2f, Est2AnnDev: %.2f" %
                 (100 * res[2], 100 * res[0], 100 * res[1], 100 * res[5],
                  100 * res[3], 100 * res[4], res[6], res[7], res[8]))


def times_to_intervals(times):
    """Given a set of times, convert them into intervals.

    Parameters
    ----------
    times: np.array(N)
        A set of times.

    Returns
    -------
    inters: np.array(N-1, 2)
        A set of intervals.
    """
    return np.asarray(zip(times[:-1], times[1:]))


def intervals_to_times(inters):
    """Given a set of intervals, convert them into times.

    Parameters
    ----------
    inters: np.array(N-1, 2)
        A set of intervals.

    Returns
    -------
    times: np.array(N)
        A set of times.
    """
    return np.concatenate((inters.flatten()[::2], [inters[-1, -1]]), axis=0)


def compute_results(ann_inter, est_inter, trim, bins, est_file):
    """Compute the results using all the available evaluations."""
    # F-measures
    logging.info("Analyzing: %s" % est_file)
    P3, R3, F3 = mir_eval.boundary.detection(ann_inter, est_inter,
                                             window=3, trim=trim)
    P05, R05, F05 = mir_eval.boundary.detection(ann_inter, est_inter,
                                                window=0.5, trim=trim)

    # Information gain
    D = compute_information_gain(ann_inter, est_inter, est_file, bins=bins)
    #D = compute_conditional_entropy(ann_inter, est_inter, window=3, trim=trim)
    #D = R05

    # Median Deviations
    ann_to_est, est_to_ann = mir_eval.boundary.deviation(ann_inter, est_inter,
                                                         trim=trim)

    return [P3, R3, F3, P05, R05, F05, D, ann_to_est, est_to_ann]


def compute_gt_results(est_file, trim, annot_beats, jam_files, alg_id,
                       beatles=False, bins=10, **params):
    """Computes the results by using the ground truth dataset."""

    # Get the ds_prefix
    ds_prefix = os.path.basename(est_file).split("_")[0]

    # Get corresponding annotation file
    jam_file = get_annotation(est_file, jam_files)

    if beatles:
        jam = jams2.load(jam_file)
        if jam.metadata.artist != "The Beatles":
            return []

    try:
        ann_inter, ann_labels = jams2.converters.load_jams_range(jam_file,
                            "sections", context=MSAF.prefix_dict[ds_prefix])
    except:
        logging.warning("No annotations for file: %s" % jam_file)
        return []

    est_inter = MSAF.read_boundaries(est_file, alg_id, annot_beats, **params)
    if est_inter == []:
        return []

    # Compute the results and return
    return compute_results(ann_inter, est_inter, trim, bins, est_file)


def compute_mma_results(est_file, trim, annot_beats, bins=10):
    """Compute the Mean Measure Agreement for all the algorithms of the given
    file est_file."""
    results_mma = []
    for algorithms in itertools.combinations(MSAF.get_algo_ids(est_file), 2):
        # Read estimated times from both algorithms
        est_times1 = MSAF.read_boundaries(est_file, algorithms[0],
                            annot_beats, feature=feat_dict[algorithms[0]])
        est_times2 = MSAF.read_boundaries(est_file, algorithms[1],
                            annot_beats, feature=feat_dict[algorithms[1]])
        if est_times1 == [] or est_times2 == []:
            continue

        # Compute results
        results = compute_results(est_times1, est_times2, trim, bins, est_file)
        results_mma.append(results)

    return results_mma


def compute_information_gain(ann_inter, est_inter, est_file, bins):
    """Computes the information gain of the est_file from the annotated
    intervals and the estimated intervals."""
    ann_times = intervals_to_times(ann_inter)
    est_times = intervals_to_times(est_inter)
    #print est_file
    #D = mir_eval.beat.information_gain(ann_times, est_times, bins=bins)
    try:
        D = mir_eval.beat.information_gain(ann_times, est_times, bins=bins)
    except:
        logging.warning("Couldn't compute the Information Gain for file "
                        "%s" % est_file)
        D = 0
    return D


def binary_entropy(score):
    """Binary entropy for the given score. Since it's binary,
    the entropy will be maximum (1.0) when score=0.5"""
    scores = np.asarray([score, 1 - score])
    entropy = 0
    for s in scores:
        if s == 0:
            s += 1e-17
        entropy += s * np.log2(s)
    entropy = -entropy
    if entropy < 1e-10:
        entropy = 0
    return entropy


def compute_conditional_entropy(ann_times, est_times, window=3, trim=False):
    """Computes the conditional recall entropies."""
    P, R, F = mir_eval.boundary.detection(ann_times, est_times, window=window,
                                          trim=trim)
    # Recall can be seen as the conditional probability of how likely it is
    # for the algorithm to find all the annotated boundaries.
    return binary_entropy(R)


def save_results_ds(cursor, alg_id, results, annot_beats, trim,
                    feature, track_id=None, ds_name=None):
    """Saves the results into the dataset.

    Parameters
    ----------
    cursor: obj
        Cursor connected to the results SQLite dataset.
    alg_id: str
        Identifier of the algorithm. E.g. serra, olda.
    results: np.array
        Array containing the results of this current algorithm to be saved.
    annot_beats: bool
        Whether to use the annotated beats or not.
    trim: bool
        Whether to trim the first and last boundaries or not.
    feature: str
        What type of feature to use for the specific algo_id. E.g. hpcp
    track_id: str
        The identifier of the current track, which is its filename.
    ds_name: str
        The name of the dataset (e.g. SALAMI, Cerulean). Use all datasets if
        None.
    """
    # Sanity Check
    if track_id is None and ds_name is None:
        logging.error("You should pass at least a track id or a dataset name")
        return

    # Make sure that the results are stored in numpy arrays
    res = np.asarray(results)

    if track_id is not None:
        all_values = (track_id, res[5], res[3], res[4], res[2], res[0], res[1],
                      res[6], res[7], res[8], annot_beats, feature, "none",
                      trim)
        table = "%s_bounds" % alg_id
        select_where = "track_id=?"
        select_values = (track_id, annot_beats, feature, trim)
    elif ds_name is not None:
        # Aggregate results
        res = np.mean(res, axis=0)
        all_values = (alg_id, ds_name, res[5], res[3], res[4], res[2], res[0],
                      res[1], res[6], res[7], res[8], annot_beats, feature,
                      "none", trim)
        table = "boundaries"
        select_where = "algo_id=? AND ds_name=?"
        select_values = (alg_id, ds_name, annot_beats, feature, trim)

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
        evaluations = (res[5], res[3], res[4], res[2], res[0],
                       res[1], res[6], res[7], res[8])
        evaluations += select_values
        sql_cmd = "UPDATE %s SET F05=?, P05=?, R05=?, F3=?, " \
            "P3=?, R3=?, D=?, DevA2E=?, DevE2A=?  WHERE %s AND annot_beat=? " \
            "AND feature=? AND trim=?" % (table, select_where)
        cursor.execute(sql_cmd, evaluations)


def get_annotation(est_file, jam_files):
    """Gets the JAMS annotation given an estimation file."""
    idx = [i for i, s in enumerate(jam_files) if
           os.path.basename(est_file)[:-5] in s][0]
    jam_file = jam_files[idx]

    assert os.path.basename(est_file)[:-5] == \
        os.path.basename(jam_file)[:-5]

    return jam_file


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
    results = np.empty((0, 9))      # Results: P3, R3, F3, P05, R05, F05, D,
                                    # median deviations

    # Dataset results
    results_ds = []

    try:
        feature = params["feature"]
    except:
        feature = ""

    curr_ds = os.path.basename(est_files[0]).split("_")[0]
    bins = 250

    for est_file in est_files:
        if salamii:
            num = int(os.path.basename(est_file).split("_")[1].split(".")[0])
            if num < 956 or num > 1498:
                continue

        if mma:
            alg_id = "mma"
            feature = ""

            # Compute the MMA for the current file
            results_mma = compute_mma_results(est_file, trim, annot_beats)

            # Compute the averages
            results_ds.append(np.mean(np.asarray(results_mma), axis=0))

        else:
            results_gt = compute_gt_results(est_file, trim, annot_beats,
                                            jam_files, alg_id, beatles,
                                            bins=bins, **params)
            if results_gt == []:
                continue
            results_ds.append(results_gt)

        # Save Track Result to database
        save_results_ds(c, alg_id, results_ds[-1], annot_beats, trim, feature,
                        track_id=os.path.basename(est_file))

        # Store dataset results if needed
        actual_ds_name = os.path.basename(est_file).split("_")[0]
        if curr_ds != actual_ds_name:
            save_results_ds(c, alg_id, results_ds, annot_beats, trim, feature,
                            ds_name=curr_ds)
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

    logging.info("%d tracks analized" % len(results))


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
