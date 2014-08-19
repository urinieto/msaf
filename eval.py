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
import os
import pandas as pd
import sys
import time

# Local stuff
import msaf
import msaf.input_output as io
import msaf.algorithms as algorithms
from msaf import jams2
from msaf import utils


def print_results(results):
    """Print all the results.

    Parameters
    ----------
    results: pd.DataFrame
        Dataframe with all the results
    """
    res = results.mean()
    logging.info("Results:\n%s" % res)


def compute_results(ann_inter, est_inter, ann_labels, est_labels, bins,
                    est_file):
    """Compute the results using all the available evaluations.

    Return
    ------
    results : dict
        Contains the results of all the evaluations for the given file.
        Keys are the following:
            track_name  : Name of the track
            ds_name :   Name of the data set
            HitRate_3F  :   F-measure of hit rate at 3 seconds
            HitRate_3P  :   Precision of hit rate at 3 seconds
            HitRate_3R  :   Recall of hit rate at 3 seconds
            HitRate_0.5F  :   F-measure of hit rate at 0.5 seconds
            HitRate_0.5P  :   Precision of hit rate at 0.5 seconds
            HitRate_0.5R  :   Recall of hit rate at 0.5 seconds
            HitRate_t3F  :   F-measure of hit rate at 3 seconds (trimmed)
            HitRate_t3P  :   Precision of hit rate at 3 seconds (trimmed)
            HitRate_t3F  :   Recall of hit rate at 3 seconds (trimmed)
            HitRate_t0.5F  :   F-measure of hit rate at 0.5 seconds (trimmed)
            HitRate_t0.5P  :   Precision of hit rate at 0.5 seconds (trimmed)
            HitRate_t0.5R  :   Recall of hit rate at 0.5 seconds (trimmed)
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
    res["HitRate_3P"], res["HitRate_3R"], res["HitRate_3F"] = \
        mir_eval.boundary.detection(ann_inter, est_inter, window=3, trim=False)
    res["HitRate_0.5P"], res["HitRate_0.5R"], res["HitRate_0.5F"] = \
        mir_eval.boundary.detection(ann_inter, est_inter, window=.5, trim=False)
    res["HitRate_t3P"], res["HitRate_t3R"], res["HitRate_t3F"] = \
        mir_eval.boundary.detection(ann_inter, est_inter, window=3, trim=True)
    res["HitRate_t0.5P"], res["HitRate_t0.5R"], res["HitRate_t0.5F"] = \
        mir_eval.boundary.detection(ann_inter, est_inter, window=.5, trim=True)

    # Information gain
    res["D"] = compute_information_gain(ann_inter, est_inter, est_file,
                                        bins=bins)

    # Median Deviations
    res["DevA2E"], res["DevE2A"] = mir_eval.boundary.deviation(
        ann_inter, est_inter, trim=False)
    res["DevtA2E"], res["DevtE2A"] = mir_eval.boundary.deviation(
        ann_inter, est_inter, trim=True)

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


def compute_gt_results(est_file, jam_file, boundaries_id, labels_id, config,
                       bins=251):
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
        ref_inter, ref_labels = jams2.converters.load_jams_range(
            jam_file, "sections", annotator=0,
            context=msaf.prefix_dict[ds_prefix])
    except:
        logging.warning("No annotations for file: %s" % jam_file)
        return {}

    # Read estimations with correct configuration
    est_inter, est_labels = io.read_estimations(est_file, boundaries_id,
                                                labels_id, **config)

    if est_inter == [] or len(est_inter) == 0:
        logging.warning("No estimations for file: %s" % est_file)
        return {}

    # Compute the results and return
    return compute_results(ref_inter, est_inter, ref_labels, est_labels,
                           bins, est_file)


def compute_information_gain(ann_inter, est_inter, est_file, bins):
    """Computes the information gain of the est_file from the annotated
    intervals and the estimated intervals."""
    ann_times = utils.intervals_to_times(ann_inter)
    est_times = utils.intervals_to_times(est_inter)
    try:
        D = mir_eval.beat.information_gain(ann_times, est_times, bins=bins)
    except:
        logging.warning("Couldn't compute the Information Gain for file "
                        "%s" % est_file)
        D = 0
    return D


def process_track(est_file, jam_file, salamii, beatles, boundaries_id,
                  labels_id, config):
    """Processes a single track."""

    # Sanity check
    assert os.path.basename(est_file)[:-4] == \
        os.path.basename(jam_file)[:-4], "File names are different %s --- %s" \
        % (os.path.basename(est_file)[:-4], os.path.basename(jam_file)[:-4])

    # Salami Internet hack
    if salamii:
        num = int(os.path.basename(est_file).split("_")[1].split(".")[0])
        if num < 956 or num > 1498:
            return []

    if beatles:
        jam = jams2.load(jam_file)
        if jam.metadata.artist != "The Beatles":
            return []

    try:
        one_res = compute_gt_results(est_file, jam_file, boundaries_id,
                                     labels_id, config)
    except:
        logging.warning("Could not compute evaluations for %s. Error: %s" %
                        (est_file, sys.exc_info()[1]))
        one_res = []

    return one_res


def get_results_file_name(boundaries_id, labels_id, config, ds_name):
    """Based on the config and the dataset, get the file name to store the
    results."""
    if ds_name == "*":
        ds_name = "All"
    file_name = os.path.join(msaf.results_dir, "results_%s" % ds_name)
    file_name += "_boundsE%s_labelsE%s" % (boundaries_id, labels_id)
    sorted_keys = sorted(config.keys(),
                         cmp=lambda x, y: cmp(x.lower(), y.lower()))
    for key in sorted_keys:
        file_name += "_%sE%s" % (key, str(config[key]))
    return file_name + msaf.results_ext


def process(in_path, boundaries_id, labels_id=None, ds_name="*",
            annot_beats=False, framesync=False, feature="hpcp", save=False,
            n_jobs=4):
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
    save: boolean
        Whether to save the results into the SQLite database.
    params : dict
        Additional parameters (e.g. features)

    Return
    ------
    results : pd.DataFrame
        DataFrame containing the evaluations for each file.
    """

    # Set up configuration based on algorithms parameters
    config = io.get_configuration(feature, annot_beats, framesync,
                                  boundaries_id, labels_id, algorithms)

    # Get out file in case we want to save results
    out_file = get_results_file_name(boundaries_id, labels_id, config, ds_name)

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
        est_file, jam_file, salamii, beatles, boundaries_id, labels_id, config)
        for est_file, jam_file in zip(est_files, jam_files)[:])

    # Aggregat evaluations in pandas format
    for e in evals:
        if e != []:
            results = results.append(e, ignore_index=True)
    logging.info("%d tracks analyzed" % len(results))

    # Print results
    print_results(results)

    # Save all results
    if save:
        logging.info("Writing average results in %s" % out_file)
        results.mean().to_csv(out_file)

    return results


def main():
    """Main function to parse the arguments and call the main process."""
    parser = argparse.ArgumentParser(description=
        "Evaluates the estimated results of the Segmentation dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("in_path",
                        action="store",
                        help="Input dataset")
    parser.add_argument("-bid",
                        action="store",
                        help="Boundary algorithm identifier",
                        dest="boundaries_id",
                        default="gt",
                        choices=["gt"] +
                        io.get_all_boundary_algorithms(algorithms))
    parser.add_argument("-lid",
                        action="store",
                        help="Label algorithm identifier",
                        dest="labels_id",
                        default=None,
                        choices= io.get_all_label_algorithms(algorithms))
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
    parser.add_argument("-fs",
                        action="store_true",
                        dest="framesync",
                        default=False,
                        help="Whether to use framesync features or not")
    parser.add_argument("-f",
                        action="store",
                        dest="feature",
                        default="hpcp",
                        type=str,
                        help="Type of features",
                        choices=["hpcp", "tonnetz", "mfcc"])
    parser.add_argument("-s",
                        action="store_true",
                        dest="save",
                        help="Whether to save the results or not",
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
            args.annot_beats, save=args.save, feature=args.feature,
            n_jobs=args.n_jobs, framesync=args.framesync)

    # Done!
    logging.info("Done! Took %.2f seconds." % (time.time() - start_time))

if __name__ == '__main__':
    main()
