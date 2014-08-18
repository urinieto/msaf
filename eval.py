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
from joblib import Parallel, delayed
import logging
import mir_eval
import numpy as np
import os
import pandas as pd
import sqlite3
import time

# Local stuff
import msaf
from msaf import jams2
from msaf import input_output as io
from msaf import utils
from msaf import algorithms


def print_results(results):
    """Print all the results.

    Parameters
    ----------
    results: pd.DataFrame
        Dataframe with all the results
    """
    res = results.mean()
    logging.info("F3: %.2f, P3: %.2f, R3: %.2f, F05: %.2f, P05: %.2f, "
                 "R05: %.2f, D: %.4f, Ann2EstDev: %.2f, Est2AnnDev: %.2f, "
                 "PWF: %.2f, PWP: %.2f, PWR: %.2f, Sf: %.2f, So: %.2f, "
                 "Su: %.2f" %
                 (100 * res["F3"], 100 * res["P3"], 100 * res["R3"],
                  100 * res["F0.5"], 100 * res["P0.5"], 100 * res["R0.5"],
                  res["D"], res["DevA2E"], res["DevE2A"],
                  100 * res["PWF"], 100 * res["PWP"], 100 * res["PWR"],
                  100 * res["Sf"], 100 * res["So"], 100 * res["Su"]))


def compute_results(ann_inter, est_inter, ann_labels, est_labels, trim, bins,
                    est_file):
    """Compute the results using all the available evaluations.

    Return
    ------
    results : dict
        Contains the results of all the evaluations for the given file.
        Keys are the following:
            track_name  : Name of the track
            ds_name :   Name of the data set
            F3  :   F-measure of hit rate at 3 seconds
            P3  :   Precision of hit rate at 3 seconds
            R3  :   Recall of hit rate at 3 seconds
            F0.5  :   F-measure of hit rate at 0.5 seconds
            P0.5  :   Precision of hit rate at 0.5 seconds
            R0.5  :   Recall of hit rate at 0.5 seconds
            DevA2E  :   Median deviation of annotation to estimation
            DevE2A  :   Median deviation of estimation to annotation
            D   :   Information gain
            PWF : F-measure of pair-wise frame clustering
            PWP : Precision of pair-wise frame clustering
            PWR : Recall of pair-wise frame clustering
            Sf  : F-measure normalized entropy score
            So  : Oversegmentation normalized entropy score
            Su  : Undersegmentation normalized entropy score
    """
    logging.info("Evaluating %s" % os.path.basename(est_file))
    res = {}

    ### Boundaries ###
    # Hit Rate
    res["P3"], res["R3"], res["F3"] = mir_eval.boundary.detection(
        ann_inter, est_inter, window=3, trim=trim)
    res["P0.5"], res["R0.5"], res["F0.5"] = mir_eval.boundary.detection(
        ann_inter, est_inter, window=.5, trim=trim)

    # Information gain
    res["D"] = compute_information_gain(ann_inter, est_inter, est_file,
                                        bins=bins)

    # Median Deviations
    res["DevA2E"], res["DevE2A"] = mir_eval.boundary.deviation(
        ann_inter, est_inter, trim=trim)

    ### Labels ###
    if est_labels != []:
        # TODO: Remove silence?
        #last_time = ann_inter[-1][-1]
        #ann_inter = ann_inter[1:-1]
        #ann_inter[0][0] = 0
        #ann_inter[-1][-1] = last_time
        #ann_labels = ann_labels[1:-1]

        #est_inter = est_inter[1:-1]
        #print "Analyzing", est_file
        #ann_labels = list(ann_labels)
        #est_labels = list(est_labels)
        #print est_labels
        #print est_inter
        #print len(ann_labels), len(ann_inter)
        #ann_inter, ann_labels = mir_eval.util.adjust_intervals(ann_inter,
                                                            #ann_labels)
        #est_inter, est_labels = mir_eval.util.adjust_intervals(
            #est_inter, est_labels, t_min=0, t_max=ann_inter.max())
        #print len(ann_labels), len(ann_inter)
        #print len(est_labels), len(est_inter)
        #print est_labels

        ## Pair-wise frame clustering
        #res["PWP"], res["PWR"], res["PWF"] = mir_eval.structure.pairwise(
            #ann_inter, ann_labels, est_inter, est_labels)

        ## Normalized Conditional Entropies
        #res["So"], res["Su"], res["Sf"] = mir_eval.structure.nce(
            #ann_inter, ann_labels, est_inter, est_labels)
        try:
            # Align labels with intervals
            #print est_inter, est_labels
            ann_labels = list(ann_labels)
            est_labels = list(est_labels)
            ann_inter, ann_labels = mir_eval.util.adjust_intervals(ann_inter,
                                                                ann_labels)
            est_inter, est_labels = mir_eval.util.adjust_intervals(
                est_inter, est_labels, t_min=0, t_max=ann_inter.max())

            # Pair-wise frame clustering
            res["PWP"], res["PWR"], res["PWF"] = mir_eval.structure.pairwise(
                ann_inter, ann_labels, est_inter, est_labels)

            # Normalized Conditional Entropies
            res["So"], res["Su"], res["Sf"] = mir_eval.structure.nce(
                ann_inter, ann_labels, est_inter, est_labels)
        except:
            logging.warning("Labeling evaluation failed in file: %s" %
                            est_file)
            return {}

    # Names
    base = os.path.basename(est_file)
    res["track_id"] = base[:-5]
    res["ds_name"] = base.split("_")[0]

    return res


def compute_gt_results(est_file, trim, annot_beats, jam_file, boundaries_id,
                       labels_id, annotator="GT", bins=251, **params):
    """Computes the results by using the ground truth dataset identified by
    the annotator parameter.

    Return
    ------
    results : dict
        Dictionary of the results (see function compute_results).
    """

    # Get the ds_prefix
    ds_prefix = os.path.basename(est_file).split("_")[0]

    try:
        if annotator == "GT":
            ref_inter, ref_labels = jams2.converters.load_jams_range(
                jam_file, "sections", annotator=0,
                context=msaf.prefix_dict[ds_prefix])
        else:
            ref_inter, ref_labels = jams2.converters.load_jams_range(
                jam_file, "sections", annotator_name=annotator,
                context="large_scale")
    except:
        logging.warning("No annotations for file: %s" % jam_file)
        return {}

    # Set up configuration based on algorithms parameters
    algo_id = boundaries_id
    if algo_id == "gt":
        algo_id = labels_id
    config = eval(algorithms.__name__ + "." + algo_id).config.config
    config["annot_beats"] = annot_beats
    for key in params.keys():
        config[key] = params[key]

    # Read estimations with correct configuration
    est_inter, est_labels = io.read_estimations(est_file, boundaries_id,
                                                labels_id, **config)

    if est_inter == [] or len(est_inter) == 0:
        logging.warning("No estimations for file: %s" % est_file)
        return {}

    # Compute the results and return
    return compute_results(ref_inter, est_inter, ref_labels, est_labels, trim,
                           bins, est_file)


def compute_information_gain(ann_inter, est_inter, est_file, bins):
    """Computes the information gain of the est_file from the annotated
    intervals and the estimated intervals."""
    ann_times = utils.intervals_to_times(ann_inter)
    est_times = utils.intervals_to_times(est_inter)
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


def process_track(est_file, jam_file, salamii, beatles, trim, annot_beats,
                  boundaries_id, labels_id, annotator, **params):
    """Processes a single track."""

    #if est_file != "/Users/uri/datasets/Segments/estimations/SALAMI_576.json":
        #return {}
    if est_file == "/Users/uri/datasets/Segments/estimations/SALAMI_920.json":
        return {}

    # Sanity check
    assert os.path.basename(est_file)[:-4] == \
        os.path.basename(jam_file)[:-4]

    if salamii:
        num = int(os.path.basename(est_file).split("_")[1].split(".")[0])
        if num < 956 or num > 1498:
            return []

    if beatles:
        jam = jams2.load(jam_file)
        if jam.metadata.artist != "The Beatles":
            return []

    one_res = compute_gt_results(est_file, trim, annot_beats, jam_file,
                                 boundaries_id, labels_id, annotator, **params)

    return one_res


def process(in_path, boundaries_id, labels_id=None, ds_name="*",
            annot_beats=False, trim=False, save=False, annotator="GT",
            sql_file="results/results.sqlite", n_jobs=4, **params):
    """Main process.

    Parameters
    ----------
    in_path : str
        Path to the dataset root folder.
    boundaries_id : str
        Boundaries algorithm identifier (e.g. siplca, cnmf)
    labels_id : str
        Labels algorithm identifier (e.g. siplca, cnmf)
    ds_name : str
        Name of the dataset to be evaluated (e.g. SALAMI). * stands for all.
    annot_beats : boolean
        Whether to use the annotated beats or not.
    annot_bounds : boolean
        Whether to use the annotated bounds or not.
    trim : boolean
        Whether to trim the first and last boundary when evaluating boundaries.
    save: boolean
        Whether to save the results into the SQLite database.
    annotator: str
        Annotator identifier of the JAMS to use as ground truth.
    sql_file: str
        Path to the SQLite results database.
    params : dict
        Additional parameters (e.g. features)

    Return
    ------
    results : pd.DataFrame
        DataFrame containing the evaluations for each file.
    """

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

    if save:
        conn = sqlite3.connect(sql_file)
        conn.text_factory = str     # Fixes the 8-bit encoding string problem
        c = conn.cursor()

    # Get files
    jam_files = glob.glob(os.path.join(in_path, msaf.Dataset.references_dir,
                                       ("%s_*" + msaf.Dataset.references_ext)
                                       % ds_name))
    est_files = glob.glob(os.path.join(in_path, msaf.Dataset.estimations_dir,
                                       ("%s_*" + msaf.Dataset.estimations_ext)
                                       % ds_name))

    logging.info("Evaluating %d tracks..." % len(jam_files))

    # All evaluations
    results = pd.DataFrame()

    # Evaluate in parallel
    evals = Parallel(n_jobs=n_jobs)(delayed(process_track)(
        est_file, jam_file, salamii, beatles, trim, annot_beats, boundaries_id,
        labels_id, annotator, **params)
        for est_file, jam_file in zip(est_files, jam_files))

    for e in evals:
        if e != []:
            results = results.append(e, ignore_index=True)

    # TODO: Save all results
    if save:
        save_results_ds(c, boundaries_id, results, annot_beats, trim,
                        ds_name="all")

    # Print results
    print_results(results)

    # Commit changes to database and close
    if save:
        conn.commit()
        conn.close()

    logging.info("%d tracks analyzed" % len(results))

    return results


def main():
    """Main function to parse the arguments and call the main process."""
    parser = argparse.ArgumentParser(description=
        "Evaluates the estimated results of the Segmentation dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("in_path",
                        action="store",
                        help="Input dataset")
    parser.add_argument("boundaries_id",
                        action="store",
                        help="Boundary algorithm identifier "
                        "(e.g. olda, siplca)")
    parser.add_argument("-la",
                        action="store",
                        help="Label algorithm identifier "
                        "(e.g. cnmf, siplca)",
                        dest="labels_id",
                        default=None)
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
    parser.add_argument("-s",
                        action="store_true",
                        dest="save",
                        help="Whether to save the results in the SQL or not",
                        default=False)
    parser.add_argument("-j",
                        action="store",
                        dest="n_jobs",
                        default=4,
                        type=int,
                        help="The number of processes to run in parallel")
    args = parser.parse_args()
    start_time = time.time()

    # Setup the logger
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s',
                        level=logging.INFO)

    # Run the algorithm
    process(args.in_path, args.boundaries_id, args.labels_id, args.ds_name,
            args.annot_beats, trim=args.trim, save=args.save,
            feature=args.feature, n_jobs=args.n_jobs)

    # Done!
    logging.info("Done! Took %.2f seconds." % (time.time() - start_time))

if __name__ == '__main__':
    main()
