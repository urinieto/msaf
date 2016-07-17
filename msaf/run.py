"""
This module contains multiple functions in order to run MSAF algorithms.
"""
import librosa
import logging
import numpy as np
import os

from joblib import Parallel, delayed

import msaf
from msaf import input_output as io
from msaf import utils
from msaf import plotting
from msaf.features import Features
import msaf.algorithms as algorithms


def get_boundaries_module(boundaries_id):
    """Obtains the boundaries module given a boundary algorithm identificator.

    Parameters
    ----------
    boundaries_id: str
        Boundary algorithm identificator (e.g., foote, sf).

    Returns
    -------
    module: object
        Object containing the selected boundary module.
        None for "ground truth".
    """
    if boundaries_id == "gt":
        return None
    try:
        module = eval(algorithms.__name__ + "." + boundaries_id)
    except AttributeError:
        raise RuntimeError("Algorithm %s can not be found in msaf!" %
                           boundaries_id)
    if not module.is_boundary_type:
        raise RuntimeError("Algorithm %s can not identify boundaries!" %
                           boundaries_id)
    return module


def get_labels_module(labels_id):
    """Obtains the label module given a label algorithm identificator.

    Parameters
    ----------
    labels_id: str
        Label algorithm identificator (e.g., fmc2d, cnmf).

    Returns
    -------
    module: object
        Object containing the selected label module.
        None for not computing the labeling part of music segmentation.
    """
    if labels_id is None:
        return None
    try:
        module = eval(algorithms.__name__ + "." + labels_id)
    except AttributeError:
        raise RuntimeError("Algorithm %s can not be found in msaf!" %
                           labels_id)
    if not module.is_label_type:
        raise RuntimeError("Algorithm %s can not label segments!" %
                           labels_id)
    return module


def run_hierarchical(audio_file, bounds_module, labels_module, frame_times,
                     config, annotator_id=0):
    """Runs hierarchical algorithms with the specified identifiers on the
    audio_file. See run_algorithm for more information.
    """
    # Get features to make code nicer
    features = config["features"].features

    if bounds_module is None:
        raise RuntimeError("A boundary algorithm is needed when using "
                           "hierarchical segmentation.")
    if labels_module is not None and \
            bounds_module.__name__ != labels_module.__name__:
        raise RuntimeError("The same algorithm for boundaries and labels is "
                           "needed when using hierarchical segmentation.")
    S = bounds_module.Segmenter(audio_file, **config)
    est_idxs, est_labels = S.processHierarchical()

    # Make sure the first and last boundaries are included for each
    # level in the hierarchy
    est_times = []
    cleaned_est_labels = []
    for level in range(len(est_idxs)):
        est_level_times, est_level_labels = \
            utils.process_segmentation_level(
                est_idxs[level], est_labels[level], features.shape[0],
                frame_times, config["features"].dur)
        est_times.append(est_level_times)
        cleaned_est_labels.append(est_level_labels)
    est_labels = cleaned_est_labels

    return est_times, est_labels


def run_flat(file_struct, bounds_module, labels_module, frame_times, config,
             annotator_id):
    """Runs the flat algorithms with the specified identifiers on the
    audio_file. See run_algorithm for more information.
    """
    # Get features to make code nicer
    features = config["features"].features

    # Segment using the specified boundaries and labels
    # Case when boundaries and labels algorithms are the same
    if bounds_module is not None and labels_module is not None and \
            bounds_module.__name__ == labels_module.__name__:
        S = bounds_module.Segmenter(file_struct, **config)
        est_idxs, est_labels = S.processFlat()
    # Different boundary and label algorithms
    else:
        # Identify segment boundaries
        if bounds_module is not None:
            S = bounds_module.Segmenter(file_struct, in_labels=[], **config)
            est_idxs, est_labels = S.processFlat()
        else:
            try:
                est_times, est_labels = io.read_references(
                    file_struct.audio_file, annotator_id=annotator_id)
                est_idxs = io.align_times(est_times, frame_times[:-1])
                if est_idxs[0] != 0:
                    est_idxs = np.concatenate(([0], est_idxs))
                if est_idxs[-1] != features.shape[0] - 1:
                    est_idxs = np.concatenate((
                        est_idxs, [features.shape[0] - 1]))
            except:
                logging.warning("No references found for file: %s" %
                                file_struct.audio_file)
                return [], []

        # Label segments
        if labels_module is not None:
            if len(est_idxs) == 2:
                est_labels = np.array([0])
            else:
                S = labels_module.Segmenter(file_struct,
                                            in_bound_idxs=est_idxs,
                                            **config)
                est_labels = S.processFlat()[1]

    # Make sure the first and last boundaries are included
    est_times, est_labels = utils.process_segmentation_level(
        est_idxs, est_labels, features.shape[0], frame_times,
        config["features"].dur)

    return est_times, est_labels


def run_algorithms(file_struct, boundaries_id, labels_id, config,
                   annotator_id=0):
    """Runs the algorithms with the specified identifiers on the audio_file.

    Parameters
    ----------
    file_struct: `msaf.io.FileStruct`
        Object with the file paths.
    boundaries_id: str
        Identifier of the boundaries algorithm to use ("gt" for ground truth).
    labels_id: str
        Identifier of the labels algorithm to use (None for not labeling).
    config: dict
        Dictionary containing the custom parameters of the algorithms to use.
    annotator_id: int
        Annotator identificator in the ground truth.

    Returns
    -------
    est_times: np.array or list
        List of estimated times for the segment boundaries.
        If `list`, it will be a list of np.arrays, sorted by segmentation
        layer.
    est_labels: np.array or list
        List of all the labels associated segments.
        If `list`, it will be a list of np.arrays, sorted by segmentation
        layer.
    """
    # Check that there are enough audio frames
    if config["features"].features.shape[0] <= msaf.config.minimum_frames:
        logging.warning("Audio file too short, or too many few beats "
                        "estimated. Returning empty estimations.")
        return np.asarray([0, config["features"].dur]), \
            np.asarray([0], dtype=int)

    # Get the corresponding modules
    bounds_module = get_boundaries_module(boundaries_id)
    labels_module = get_labels_module(labels_id)

    # Get the correct frame times
    frame_times = config["features"].frame_times

    # Segment audio based on type of segmentation
    run_fun = run_hierarchical if config["hier"] else run_flat
    est_times, est_labels = run_fun(file_struct, bounds_module, labels_module,
                                    frame_times, config, annotator_id)

    return est_times, est_labels


def process_track(file_struct, boundaries_id, labels_id, config,
                  annotator_id=0):
    """Prepares the parameters, runs the algorithms, and saves results.

    Parameters
    ----------
    file_struct: `msaf.io.FileStruct`
        FileStruct containing the paths of the input files (audio file,
        features file, reference file, output estimation file).
    boundaries_id: str
        Identifier of the boundaries algorithm to use ("gt" for ground truth).
    labels_id: str
        Identifier of the labels algorithm to use (None for not labeling).
    config: dict
        Dictionary containing the custom parameters of the algorithms to use.
    annotator_id: int
        Annotator identificator in the ground truth.

    Returns
    -------
    est_times: np.array
        List of estimated times for the segment boundaries.
    est_labels: np.array
        List of all the labels associated segments.
    """
    logging.info("Segmenting %s" % file_struct.audio_file)

    # Get features
    config["features"] = Features.select_features(
        config["feature"], file_struct, config["annot_beats"],
        config["framesync"])

    # Get estimations
    est_times, est_labels = run_algorithms(file_struct,
                                           boundaries_id, labels_id, config,
                                           annotator_id=annotator_id)

    # Save
    logging.info("Writing results in: %s" % file_struct.est_file)
    io.save_estimations(file_struct, est_times, est_labels,
                        boundaries_id, labels_id, **config)

    return est_times, est_labels


def process(in_path, annot_beats=False, feature="pcp", framesync=False,
            boundaries_id=msaf.config.default_bound_id,
            labels_id=msaf.config.default_label_id, hier=False,
            sonify_bounds=False, plot=False, n_jobs=4, annotator_id=0,
            config=None, out_bounds="out_bounds.wav", out_sr=22050):
    """Main process to segment a file or a collection of files.

    Parameters
    ----------
    in_path: str
        Input path. If a directory, MSAF will function in collection mode.
        If audio file, MSAF will be in single file mode.
    annot_beats: bool
        Whether to use annotated beats or not. Only available in collection
        mode.
    feature: str
        String representing the feature to be used (e.g. pcp, mfcc, tonnetz)
    framesync: str
        Whether to use framesync features or not (default: False -> beatsync)
    boundaries_id: str
        Identifier of the boundaries algorithm (use "gt" for groundtruth)
    labels_id: str
        Identifier of the labels algorithm (use None to not compute labels)
    hier : bool
        Whether to compute a hierarchical or flat segmentation.
    sonify_bounds: bool
        Whether to write an output audio file with the annotated boundaries
        or not (only available in Single File Mode).
    plot: bool
        Whether to plot the boundaries and labels against the ground truth.
    n_jobs: int
        Number of processes to run in parallel. Only available in collection
        mode.
    annotator_id: int
        Annotator identificator in the ground truth.
    config: dict
        Dictionary containing custom configuration parameters for the
        algorithms.  If None, the default parameters are used.
    out_bounds: str
        Path to the output for the sonified boundaries (only in single file
        mode, when sonify_bounds is True.
    out_sr : int
        Sampling rate for the sonified bounds.

    Returns
    -------
    results : list
        List containing tuples of (est_times, est_labels) of estimated
        boundary times and estimated labels.
        If labels_id is None, est_labels will be a list of -1.
    """
    # Seed random to reproduce results
    np.random.seed(123)

    # Make sure that the features used are correct
    assert feature in msaf.features_registry.keys()

    # Set up configuration based on algorithms parameters
    if config is None:
        config = io.get_configuration(feature, annot_beats, framesync,
                                      boundaries_id, labels_id)
        config["features"] = None

    # Save multi-segment (hierarchical) configuration
    config["hier"] = hier
    if not os.path.exists(in_path):
        raise RuntimeError("File or directory does not exists, %s" %
                           in_path)
    if os.path.isfile(in_path):
        # Single file mode
        # Get (if they exitst) or compute features
        file_struct = msaf.io.FileStruct(in_path)

        # Use temporary file in single mode
        file_struct.features_file = msaf.config.features_tmp_file

        # Get features
        config["features"] = Features.select_features(
            feature, file_struct, annot_beats, framesync)

        # And run the algorithms
        est_times, est_labels = run_algorithms(file_struct, boundaries_id,
                                               labels_id, config,
                                               annotator_id=annotator_id)

        if sonify_bounds:
            logging.info("Sonifying boundaries in %s..." % out_bounds)
            audio_hq, sr = librosa.load(in_path, sr=out_sr)
            utils.sonify_clicks(audio_hq, est_times, out_bounds, out_sr)

        if plot:
            plotting.plot_one_track(file_struct, est_times, est_labels,
                                    boundaries_id, labels_id, ds_name)

        # Save estimations
        msaf.utils.ensure_dir(os.path.dirname(file_struct.est_file))
        io.save_estimations(file_struct, est_times, est_labels,
                            boundaries_id, labels_id, **config)

        return est_times, est_labels
    else:
        # Collection mode
        file_structs = io.get_dataset_files(in_path, ds_name)

        # Call in parallel
        return Parallel(n_jobs=n_jobs)(delayed(process_track)(
            file_struct, boundaries_id, labels_id, config,
            annotator_id=annotator_id) for file_struct in file_structs[:])
