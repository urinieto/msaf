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
import pylab as plt
import pandas as pd

import mir_eval
import jams2

import msaf_io as MSAF


feat_dict = {
    'serra' :   'mix',
    'levy'  :   'hpcp',
    'foote' :   'hpcp',
    'siplca':   '',
    'olda'  :   '',
    'kmeans':   'hpcp',
    'cnmf'  :   'hpcp',
    'cnmf2' :   'hpcp',
    'cnmf3' :   'hpcp'
}


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


def compute_results(ann_inter, est_inter, ann_labels, est_labels, trim, bins,
                    est_file, annot_bounds=False):
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

    if annot_bounds:
        est_inter = ann_inter

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


def compute_gt_results(est_file, trim, annot_beats, jam_file, alg_id,
                       annotator="GT", bins=251, annot_bounds=False, **params):
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
            ann_inter, ann_labels = jams2.converters.load_jams_range(
                jam_file, "sections", annotator=0,
                context=MSAF.prefix_dict[ds_prefix])
        else:
            ann_inter, ann_labels = jams2.converters.load_jams_range(
                jam_file, "sections", annotator_name=annotator,
                context="large_scale")
    except:
        logging.warning("No annotations for file: %s" % jam_file)
        return {}

    est_inter = MSAF.read_estimations(est_file, alg_id, annot_beats, **params)
    est_labels = MSAF.read_estimations(est_file, alg_id, annot_beats,
                                       bounds=False, **params)
    if est_inter == []:
        return {}

    # Compute the results and return
    return compute_results(ann_inter, est_inter, ann_labels, est_labels, trim,
                           bins, est_file, annot_bounds=annot_bounds)


def plot_boundaries(all_boundaries, est_file):
    """Plots all the boundaries.

    Parameters
    ----------
    all_boundaries: list
        A list of np.arrays containing the times of the boundaries, one array
        for each algorithm.
    est_file: str
        Path to the estimated file (JSON file)
    """
    N = len(all_boundaries)  # Number of lists of boundaries
    algo_ids = MSAF.get_algo_ids(est_file)
    algo_ids = ["GT"] + algo_ids
    figsize = (5, 2.2)
    plt.figure(1, figsize=figsize, dpi=120, facecolor='w', edgecolor='k')
    for i, boundaries in enumerate(all_boundaries):
        print boundaries
        for b in boundaries:
            plt.axvline(b, i / float(N), (i + 1) / float(N))
        plt.axhline(i / float(N), color="k", linewidth=1)

    #plt.title(os.path.basename(est_file))
    plt.title("Nelly Furtado - Promiscuous")
    #plt.title("Quartetto Italiano - String Quartet in F")
    plt.yticks(np.arange(0, 1, 1 / float(N)) + 1 / (float(N) * 2))
    plt.gcf().subplots_adjust(bottom=0.22)
    plt.gca().set_yticklabels(algo_ids)
    #plt.gca().invert_yaxis()
    plt.xlabel("Time (seconds)")
    plt.show()


def get_all_est_boundaries(est_file, annot_beats):
    """Gets all the estimated boundaries for all the algorithms.

    Parameters
    ----------
    est_file: str
        Path to the estimated file (JSON file)
    annot_beats: bool
        Whether to use the annotated beats or not.

    Returns
    -------
    all_boundaries: list
        A list of np.arrays containing the times of the boundaries, one array
        for each algorithm
    """
    all_boundaries = []

    # Get GT boundaries
    jam_file = os.path.dirname(est_file) + "/../annotations/" + \
        os.path.basename(est_file).replace("json", "jams")
    ds_prefix = os.path.basename(est_file).split("_")[0]
    ann_inter, ann_labels = jams2.converters.load_jams_range(jam_file,
                        "sections", context=MSAF.prefix_dict[ds_prefix])
    ann_times = intervals_to_times(ann_inter)
    all_boundaries.append(ann_times)

    # Estimations
    for algo_id in MSAF.get_algo_ids(est_file):
        est_inters = MSAF.read_estimations(est_file, algo_id,
                        annot_beats, feature=feat_dict[algo_id])
        boundaries = intervals_to_times(est_inters)
        all_boundaries.append(boundaries)
    return all_boundaries


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


def process(in_path, alg_id, ds_name="*", annot_beats=False,
            annot_bounds=False, trim=False, save=False, annotator="GT",
            sql_file="results/results.sqlite", **params):
    """Main process.

    Parameters
    ----------
    in_path : str
        Path to the dataset root folder.
    alg_id : str
        Algorithm identifier (e.g. siplca, cnmf)
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
    jam_files = glob.glob(os.path.join(in_path, "annotations",
                                       "%s_*.jams" % ds_name))
    est_files = glob.glob(os.path.join(in_path, "estimations",
                                       "%s_*.json" % ds_name))

    logging.info("Evaluating %d tracks..." % len(jam_files))

    # All evaluations
    results = pd.DataFrame()

    for est_file, jam_file in zip(est_files, jam_files):
        # Sanity check
        assert os.path.basename(est_file)[:-4] == \
            os.path.basename(jam_file)[:-4]

        if salamii:
            num = int(os.path.basename(est_file).split("_")[1].split(".")[0])
            if num < 956 or num > 1498:
                continue

        if beatles:
            jam = jams2.load(jam_file)
            if jam.metadata.artist != "The Beatles":
                continue

        one_res = compute_gt_results(est_file, trim, annot_beats,
                                     jam_file, alg_id, annotator,
                                     annot_bounds=annot_bounds, **params)
        if one_res == []:
            continue

        results = results.append(one_res, ignore_index=True)

    # Save all results
    if save:
        save_results_ds(c, alg_id, results, annot_beats, trim, ds_name="all")

    # Print results
    print_results(results)

    # Commit changes to database and close
    if save:
        conn.commit()
        conn.close()

    logging.info("%d tracks analized" % len(results))

    return results


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
    parser.add_argument("-bo",
                        action="store_true",
                        dest="annot_bounds",
                        help="Use annotated bounds",
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
                        help="Whether to sasve the results in the SQL or not",
                        default=False)
    args = parser.parse_args()
    start_time = time.time()

    # Setup the logger
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s',
                        level=logging.INFO)

    # Run the algorithm
    process(args.in_path, args.alg_id, args.ds_name, args.annot_beats,
            trim=args.trim, save=args.save, feature=args.feature,
            annot_bounds=args.annot_bounds)

    # Done!
    logging.info("Done! Took %.2f seconds." % (time.time() - start_time))

if __name__ == '__main__':
    main()
