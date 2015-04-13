"""
Evaluates the estimated results of the Segmentation dataset against the
ground truth (human annotated data).
"""

from joblib import Parallel, delayed
import logging
import mir_eval
import numpy as np
import os
import pandas as pd
import six
import sys

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
        mir_eval.segment.detection(ann_inter, est_inter, window=3, trim=False)
    res["HitRate_0.5P"], res["HitRate_0.5R"], res["HitRate_0.5F"] = \
        mir_eval.segment.detection(ann_inter, est_inter, window=.5, trim=False)
    res["HitRate_t3P"], res["HitRate_t3R"], res["HitRate_t3F"] = \
        mir_eval.segment.detection(ann_inter, est_inter, window=3, trim=True)
    res["HitRate_t0.5P"], res["HitRate_t0.5R"], res["HitRate_t0.5F"] = \
        mir_eval.segment.detection(ann_inter, est_inter, window=.5, trim=True)

    # Information gain
    res["D"] = compute_information_gain(ann_inter, est_inter, est_file,
                                        bins=bins)

    # Median Deviations
    res["DevR2E"], res["DevE2R"] = mir_eval.segment.deviation(
        ann_inter, est_inter, trim=False)
    res["DevtR2E"], res["DevtE2R"] = mir_eval.segment.deviation(
        ann_inter, est_inter, trim=True)

    ### Labels ###
    if est_labels is not None and len(est_labels) != 0:
        try:
            # Align labels with intervals
            ann_labels = list(ann_labels)
            est_labels = list(est_labels)
            ann_inter, ann_labels = mir_eval.util.adjust_intervals(ann_inter,
                                                                ann_labels)
            est_inter, est_labels = mir_eval.util.adjust_intervals(
                est_inter, est_labels, t_min=0, t_max=ann_inter.max())

            # Pair-wise frame clustering
            res["PWP"], res["PWR"], res["PWF"] = mir_eval.segment.pairwise(
                ann_inter, ann_labels, est_inter, est_labels)

            # Normalized Conditional Entropies
            res["So"], res["Su"], res["Sf"] = mir_eval.segment.nce(
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


def compute_gt_results(est_file, ref_file, boundaries_id, labels_id, config,
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

    # Get context
    if ds_prefix in msaf.prefix_dict.keys():
        context = msaf.prefix_dict[ds_prefix]
    else:
        context = "function"

    try:
        # TODO: Read hierarchical annotations
        if config["hier"]:
            ref_times, ref_labels, ref_levels = \
                msaf.io.read_hier_references(ref_file, annotation_id=0)
        else:
            ref_inter, ref_labels = jams2.converters.load_jams_range(
                ref_file, "sections", annotator=0, context=context)
    except:
        logging.warning("No references for file: %s" % ref_file)
        return {}

    # Read estimations with correct configuration
    est_inter, est_labels = io.read_estimations(est_file, boundaries_id,
                                                labels_id, **config)

    if len(est_inter) == 0:
        logging.warning("No estimations for file: %s" % est_file)
        return {}

    # Compute the results and return
    if config["hier"]:
        # Hierarchical
        assert len(est_inter) == len(est_labels), "Same number of levels " \
            "are required in the boundaries and labels for the hierarchical " \
            "evaluation."
        est_times = []
        est_labels = []
        for inter in est_inter:
            est_times.append(msaf.utils.intervals_to_times(inter))
            # Add fake labels (hierarchical eval does not use labels --yet--)
            est_labels.append(np.ones(len(est_times[-1]) - 1) * -1)

        # Align the times
        utils.align_end_hierarchies(est_times, ref_times)

        # Build trees
        ref_tree = mir_eval.segment.tree.SegmentTree(ref_times, ref_labels,
                                                     ref_levels)
        est_tree = mir_eval.segment.tree.SegmentTree(est_times, est_labels)

        # Compute evaluations
        res = {}
        res["t_under10"], res["t_over10"], res["t_measure10"] = \
            mir_eval.segment.hmeasure(ref_tree, est_tree, window=10)
        res["t_under30"], res["t_over30"], res["t_measure30"] = \
            mir_eval.segment.hmeasure(ref_tree, est_tree, window=30)
        return res
    else:
        # Flat
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


def process_track(file_struct, boundaries_id, labels_id, config):
    """Processes a single track.

    Parameters
    ----------
    file_struct : object (FileStruct) or str
        File struct or full path of the audio file to be evaluated.
    boundaries_id : str
        Identifier of the boundaries algorithm.
    labels_id : str
        Identifier of the labels algorithm.
    config : dict
        Configuration of the algorithms to be evaluated.

    Returns
    -------
    one_res : dict
        Dictionary of the results (see function compute_results).
    """
    # Convert to file_struct if string is passed
    if isinstance(file_struct, six.string_types):
        file_struct = io.FileStruct(file_struct)

    est_file = file_struct.est_file
    ref_file = file_struct.ref_file

    # Sanity check
    assert os.path.basename(est_file)[:-4] == \
        os.path.basename(ref_file)[:-4], "File names are different %s --- %s" \
        % (os.path.basename(est_file)[:-4], os.path.basename(ref_file)[:-4])

    try:
        one_res = compute_gt_results(est_file, ref_file, boundaries_id,
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
    utils.ensure_dir(msaf.results_dir)
    file_name = os.path.join(msaf.results_dir, "results_%s" % ds_name)
    file_name += "_boundsE%s_labelsE%s" % (boundaries_id, labels_id)
    sorted_keys = sorted(config.keys(),
                         cmp=lambda x, y: cmp(x.lower(), y.lower()))
    for key in sorted_keys:
        file_name += "_%sE%s" % (key, str(config[key]).replace("/", "_"))
    return file_name + msaf.results_ext


def process(in_path, boundaries_id, labels_id=None, ds_name="*",
            annot_beats=False, framesync=False, feature="hpcp", hier=False,
            save=False, n_jobs=4, config=None):
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
    framesync: str
        Whether to use framesync features or not (default: False -> beatsync)
    feature: str
        String representing the feature to be used (e.g. hpcp, mfcc, tonnetz)
    hier : bool
        Whether to compute a hierarchical or flat segmentation.
    save: boolean
        Whether to save the results into the SQLite database.
    n_jobs: int
        Number of processes to run in parallel. Only available in collection
        mode.
    config: dict
        Dictionary containing custom configuration parameters for the
        algorithms.  If None, the default parameters are used.

    Return
    ------
    results : pd.DataFrame
        DataFrame containing the evaluations for each file.
    """

    # Set up configuration based on algorithms parameters
    if config is None:
        config = io.get_configuration(feature, annot_beats, framesync,
                                      boundaries_id, labels_id)

    # Hierarchical segmentation
    config["hier"] = hier

    # Sanity check for hierarchical evaluation
    if hier:
        try:
            from mir_eval.segment import tree
        except:
            logging.error("An experimental mir_eval version is needed to "
                          "evaluate hierarchical segments. Please, download it"
                          " from: https://github.com/urinieto/mir_eval")
            return []

    # Get out file in case we want to save results
    out_file = get_results_file_name(boundaries_id, labels_id, config, ds_name)

    # All evaluations
    results = pd.DataFrame()

    if os.path.isfile(in_path):
        # Single File mode
        evals = [process_track(in_path, boundaries_id, labels_id, config)]
    else:
        # Collection mode
        # If out_file already exists, do not compute new results
        if os.path.exists(out_file):
            logging.info("Results already exists, reading from file %s" %
                         out_file)
            results = pd.read_csv(out_file)
            print_results(results)
            return results

        # Get files
        file_structs = io.get_dataset_files(in_path, ds_name)

        logging.info("Evaluating %d tracks..." % len(file_structs))

        # Evaluate in parallel
        evals = Parallel(n_jobs=n_jobs)(delayed(process_track)(
            file_struct, boundaries_id, labels_id, config)
            for file_struct in file_structs[:])

    # Aggregate evaluations in pandas format
    for e in evals:
        if e != []:
            results = results.append(e, ignore_index=True)
    logging.info("%d tracks analyzed" % len(results))

    # Print results
    print_results(results)

    # Save all results
    if save:
        logging.info("Writing results in %s" % out_file)
        results.to_csv(out_file)

    return results
