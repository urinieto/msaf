"""
These set of functions help the algorithms of MSAF to read and write files
of the Segmentation Dataset.
"""
import datetime
import glob
import jams
import json
import logging
import numpy as np
import os
import re
import six

# Local stuff
import msaf
from msaf.exceptions import NoEstimationsError
from msaf import utils

# Put dataset config in a global var
ds_config = msaf.config.dataset


class FileStruct:
    def __init__(self, audio_file):
        """Creates the entire file structure given the audio file."""
        self.ds_path = os.path.dirname(os.path.dirname(audio_file))
        self.audio_file = audio_file
        self.est_file = self._get_dataset_file(ds_config.estimations_dir,
                                               ds_config.estimations_ext)
        self.features_file = self._get_dataset_file(ds_config.features_dir,
                                                    ds_config.features_ext)
        self.ref_file = self._get_dataset_file(ds_config.references_dir,
                                               ds_config.references_ext)

    def _get_dataset_file(self, dir, ext):
        """Gets the desired dataset file."""
        audio_file_ext = "." + self.audio_file.split(".")[-1]
        base_file = os.path.basename(self.audio_file).replace(
            audio_file_ext, ext)
        return os.path.join(self.ds_path, dir, base_file)

    def __repr__(self):
        """Prints the file structure."""
        return "FileStruct(\n\tds_path=%s,\n\taudio_file=%s,\n\test_file=%s," \
            "\n\tfeatures_file=%s,\n\tref_file=%s\n)" % (
                self.ds_path, self.audio_file, self.est_file,
                self.features_file, self.ref_file)


def read_estimations(est_file, boundaries_id, labels_id=None, **params):
    """Reads the estimations (boundaries and/or labels) from a jams file
    containing the estimations of an algorithm.

    Parameters
    ----------
    est_file : str
        Path to the estimated file (JAMS file).
    boundaries_id : str
        Identifier of the algorithm used to compute the boundaries.
    labels_id : str
        Identifier of the algorithm used to compute the labels.
    params : dict
        Additional search parameters. E.g. {"feature" : "pcp"}.

    Returns
    -------
    boundaries : np.array((N,2))
        Array containing the estimated boundaries in intervals.
    labels : np.array(N)
        Array containing the estimated labels.
        Empty array if labels_id is None.
    """
    # Open file and read jams
    jam = jams.load(est_file)

    # Find correct estimation
    est = find_estimation(jam, boundaries_id, labels_id, params)
    if est is None:
        raise NoEstimationsError("No estimations for file: %s" % est_file)

    # Get data values
    all_boundaries, all_labels = est.data.to_interval_values()
    if params["hier"]:
        hier_bounds = []
        hier_labels = []
        curr_bounds = []
        curr_labels = []
        curr_level = all_labels[0]["level"]
        for bounds, value in zip(all_boundaries, all_labels):
            if curr_level != value["level"]:
                hier_bounds.append(np.asarray(curr_bounds))
                hier_labels.append(np.asarray(curr_labels))
                curr_bounds = []
                curr_labels = []
            curr_bounds.append(bounds)
            curr_labels.append(value["label"])
            curr_level = value["level"]
        hier_bounds.append(np.asarray(curr_bounds))
        hier_labels.append(np.asarray(curr_labels))
        all_boundaries = hier_bounds
        all_labels = hier_labels

    return all_boundaries, all_labels


def read_references(audio_path, annotator_id=0):
    """Reads the boundary times and the labels.

    Parameters
    ----------
    audio_path : str
        Path to the audio file

    Returns
    -------
    ref_times : list
        List of boundary times
    ref_labels : list
        List of labels

    Raises
    ------
    IOError: if `audio_path` doesn't exist.
    """
    # Dataset path
    ds_path = os.path.dirname(os.path.dirname(audio_path))

    # Read references
    jam_path = os.path.join(ds_path, ds_config.references_dir,
                            os.path.basename(audio_path)[:-4] +
                            ds_config.references_ext)

    jam = jams.load(jam_path, validate=False)
    ann = jam.search(namespace='segment_.*')[annotator_id]
    ref_inters, ref_labels = ann.data.to_interval_values()

    # Intervals to times
    ref_times = utils.intervals_to_times(ref_inters)

    return ref_times, ref_labels


def align_times(times, frames):
    """Aligns the times to the closest frame times (e.g. beats).

    Parameters
    ----------
    times: np.ndarray
        Times in seconds to be aligned.
    frames: np.ndarray
        Frame times in seconds.

    Returns
    -------
    aligned_times: np.ndarray
        Aligned times.
    """
    dist = np.minimum.outer(times, frames)
    bound_frames = np.argmax(np.maximum(0, dist), axis=1)
    aligned_times = np.unique(bound_frames)
    return aligned_times


def find_estimation(jam, boundaries_id, labels_id, params):
    """Finds the correct estimation from all the estimations contained in a
    JAMS file given the specified arguments.

    Parameters
    ----------
    jam : jams.JAMS
        JAMS object.
    boundaries_id : str
        Identifier of the algorithm used to compute the boundaries.
    labels_id : str
        Identifier of the algorithm used to compute the labels.
    params : dict
        Additional search parameters. E.g. {"feature" : "pcp"}.

    Returns
    -------
    ann : jams.Annotation
        Found estimation.
        `None` if it couldn't be found.
    """
    # Use handy JAMS search interface
    namespace = "multi_segment" if params["hier"] else "segment_open"
    # TODO: This is a workaround to issue in JAMS. Should be
    # resolved in JAMS 0.2.3, but for now, this works too.
    ann = jam.search(namespace=namespace).\
        search(**{"Sandbox.boundaries_id": boundaries_id}).\
        search(**{"Sandbox.labels_id": lambda x:
                  isinstance(x, six.string_types) and
                  re.match(labels_id, x) is not None})
    for key, val in zip(params.keys(), params.values()):
        if isinstance(val, six.string_types):
            ann = ann.search(**{"Sandbox.%s" % key: val})
        else:
            ann = ann.search(**{"Sandbox.%s" % key: lambda x: x == val})

    # Check estimations found
    if len(ann) > 1:
        logging.warning("More than one estimation with same parameters.")

    if len(ann) > 0:
        ann = ann[0]

    # If we couldn't find anything, let's return None
    if not ann:
        ann = None

    return ann


def save_estimations(file_struct, times, labels, boundaries_id, labels_id,
                     **params):
    """Saves the segment estimations in a JAMS file.

    Parameters
    ----------
    file_struct : FileStruct
        Object with the different file paths of the current file.
    times : np.array or list
        Estimated boundary times.
        If `list`, estimated hierarchical boundaries.
    labels : np.array(N, 2)
        Estimated labels (None in case we are only storing boundary
        evaluations).
    boundaries_id : str
        Boundary algorithm identifier.
    labels_id : str
        Labels algorithm identifier.
    params : dict
        Dictionary with additional parameters for both algorithms.
    """
    # Remove features if they exist
    params.pop("features", None)

    # Get duration
    dur = get_duration(file_struct.features_file)

    # Convert to intervals and sanity check
    if 'numpy' in str(type(times)):
        # Flat check
        inters = utils.times_to_intervals(times)
        assert len(inters) == len(labels), "Number of boundary intervals " \
            "(%d) and labels (%d) do not match" % (len(inters), len(labels))
        # Put into lists to simplify the writing process later
        inters = [inters]
        labels = [labels]
    else:
        # Hierarchical check
        inters = []
        for level in range(len(times)):
            est_inters = utils.times_to_intervals(times[level])
            inters.append(est_inters)
            assert len(inters[level]) == len(labels[level]), \
                "Number of boundary intervals (%d) and labels (%d) do not " \
                "match in level %d" % (len(inters[level]), len(labels[level]), level)

    # Create new estimation
    namespace = "multi_segment" if params["hier"] else "segment_open"
    ann = jams.Annotation(namespace=namespace)

    # Find estimation in file
    if os.path.isfile(file_struct.est_file):
        jam = jams.load(file_struct.est_file, validate=False)
        curr_ann = find_estimation(jam, boundaries_id, labels_id, params)
        if curr_ann is not None:
            curr_ann.data = ann.data  # cleanup all data
            ann = curr_ann  # This will overwrite the existing estimation
        else:
            jam.annotations.append(ann)
    else:
        # Create new JAMS if it doesn't exist
        jam = jams.JAMS()
        jam.file_metadata.duration = dur
        jam.annotations.append(ann)

    # Save metadata and parameters
    ann.annotation_metadata.version = msaf.__version__
    ann.annotation_metadata.data_source = "MSAF"
    sandbox = {}
    sandbox["boundaries_id"] = boundaries_id
    sandbox["labels_id"] = labels_id
    sandbox["timestamp"] = \
        datetime.datetime.today().strftime("%Y/%m/%d %H:%M:%S")
    for key in params:
        sandbox[key] = params[key]
    ann.sandbox = sandbox

    # Save actual data
    for i, (level_inters, level_labels) in enumerate(zip(inters, labels)):
        for bound_inter, label in zip(level_inters, level_labels):
            dur = float(bound_inter[1]) - float(bound_inter[0])
            label = chr(int(label) + 65)
            if params["hier"]:
                value = {"label": label, "level": i}
            else:
                value = label
            ann.append(time=bound_inter[0], duration=dur,
                       value=value)

    # Write results
    jam.save(file_struct.est_file)


def get_all_boundary_algorithms():
    """Gets all the possible boundary algorithms in MSAF.

    Returns
    -------
    algo_ids : list
        List of all the IDs of boundary algorithms (strings).
    """
    algo_ids = []
    for name in msaf.algorithms.__all__:
        module = eval(msaf.algorithms.__name__ + "." + name)
        if module.is_boundary_type:
            algo_ids.append(module.algo_id)
    return algo_ids


def get_all_label_algorithms():
    """Gets all the possible label (structural grouping) algorithms in MSAF.

    Returns
    -------
    algo_ids : list
        List of all the IDs of label algorithms (strings).
    """
    algo_ids = []
    for name in msaf.algorithms.__all__:
        module = eval(msaf.algorithms.__name__ + "." + name)
        if module.is_label_type:
            algo_ids.append(module.algo_id)
    return algo_ids


def get_configuration(feature, annot_beats, framesync, boundaries_id,
                      labels_id):
    """Gets the configuration dictionary from the current parameters of the
    algorithms to be evaluated."""
    config = {}
    config["annot_beats"] = annot_beats
    config["feature"] = feature
    config["framesync"] = framesync
    bound_config = {}
    if boundaries_id != "gt":
        bound_config = \
            eval(msaf.algorithms.__name__ + "." + boundaries_id).config
        config.update(bound_config)
    if labels_id is not None:
        label_config = \
            eval(msaf.algorithms.__name__ + "." + labels_id).config

        # Make sure we don't have parameter name duplicates
        if labels_id != boundaries_id:
            overlap = set(bound_config.keys()). \
                intersection(set(label_config.keys()))
            assert len(overlap) == 0, \
                "Parameter %s must not exist both in %s and %s algorithms" % \
                (overlap, boundaries_id, labels_id)
        config.update(label_config)
    return config


def get_dataset_files(in_path):
    """Gets the files of the given dataset."""
    # Get audio files
    audio_files = []
    for ext in ds_config.audio_exts:
        audio_files += glob.glob(
            os.path.join(in_path, ds_config.audio_dir, "*" + ext))

    # Make sure directories exist
    utils.ensure_dir(os.path.join(in_path, ds_config.features_dir))
    utils.ensure_dir(os.path.join(in_path, ds_config.estimations_dir))
    utils.ensure_dir(os.path.join(in_path, ds_config.references_dir))

    # Get the file structs
    file_structs = []
    for audio_file in audio_files:
        file_structs.append(FileStruct(audio_file))

    # Sort by audio file name
    file_structs = sorted(file_structs,
                          key=lambda file_struct: file_struct.audio_file)

    return file_structs


def read_hier_references(jams_file, annotation_id=0, exclude_levels=[]):
    """Reads hierarchical references from a jams file.

    Parameters
    ----------
    jams_file : str
        Path to the jams file.
    annotation_id : int > 0
        Identifier of the annotator to read from.
    exclude_levels: list
        List of levels to exclude. Empty list to include all levels.

    Returns
    -------
    hier_bounds : list
        List of the segment boundary times in seconds for each level.
    hier_labels : list
        List of the segment labels for each level.
    hier_levels : list
        List of strings for the level identifiers.
    """
    hier_bounds = []
    hier_labels = []
    hier_levels = []
    jam = jams.load(jams_file)
    namespaces = ["segment_salami_upper", "segment_salami_function",
                  "segment_open", "segment_tut", "segment_salami_lower"]

    # Remove levels if needed
    for exclude in exclude_levels:
        if exclude in namespaces:
            namespaces.remove(exclude)

    # Build hierarchy references
    for ns in namespaces:
        ann = jam.search(namespace=ns)
        if not ann:
            continue
        ref_inters, ref_labels = ann[annotation_id].data.to_interval_values()
        hier_bounds.append(utils.intervals_to_times(ref_inters))
        hier_labels.append(ref_labels)
        hier_levels.append(ns)

    return hier_bounds, hier_labels, hier_levels


def get_duration(features_file):
    """Reads the duration of a given features file.

    Parameters
    ----------
    features_file: str
        Path to the JSON file containing the features.

    Returns
    -------
    dur: float
        Duration of the analyzed file.
    """
    with open(features_file) as f:
        feats = json.load(f)
    return float(feats["globals"]["dur"])


def write_mirex(times, labels, out_file):
    """Writes results to file using the standard MIREX format.

    Parameters
    ----------
    times: np.array
        Times in seconds of the boundaries.
    labels: np.array
        Labels associated to the segments defined by the boundaries.
    out_file: str
        Output file path to save the results.
    """
    inters = msaf.utils.times_to_intervals(times)
    assert len(inters) == len(labels)
    out_str = ""
    for inter, label in zip(inters, labels):
        out_str += "%.3f\t%.3f\t%s\n" % (inter[0], inter[1], label)
    with open(out_file, "w") as f:
        f.write(out_str[:-1])
