"""
These set of functions help the algorithms of MSAF to read and write files
of the Segmentation Dataset.
"""
from collections import Counter
import datetime
import glob
import jams
import json
import logging
import numpy as np
import os
import six

# Local stuff
import msaf
from msaf import utils


class FileStruct:
    def __init__(self, audio_file):
        """Creates the entire file structure given the audio file."""
        self.ds_path = os.path.dirname(os.path.dirname(audio_file))
        self.audio_file = audio_file
        self.est_file = self._get_dataset_file(msaf.Dataset.estimations_dir,
                                               msaf.Dataset.estimations_ext)
        self.features_file = self._get_dataset_file(msaf.Dataset.features_dir,
                                                    msaf.Dataset.features_ext)
        self.ref_file = self._get_dataset_file(msaf.Dataset.references_dir,
                                               msaf.Dataset.references_ext)

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
        Additional search parameters. E.g. {"feature" : "hpcp"}.

    Returns
    -------
    boundaries : np.array((N,2))
        Array containing the estimated boundaries in intervals.
    labels : np.array(N)
        Array containing the estimated labels.
        Empty array if labels_id is None.
    """
    # Open file and read jams
    try:
        jam = jams.load(est_file)
    except FileNotFoundError:
        logging.error("JAMS file doesn't exist %s" % est_file)
        return np.array([]), np.array([])
    except Exception as e:
        logging.error("Could not open JAMS file %s. Exception: %s" % (est_file,
                                                                      e))
        return np.array([]), np.array([])

    # Find correct estimation
    est = find_estimation(jam, boundaries_id, labels_id, params)
    if est is None:
        logging.error("Could not find estimation in %s" % est_file)
        return np.array([]), np.array([])

    # TODO: Hierarchical
    # Retrieve unique levels of segmentation
    #levels = []
    #for range in est.data:
        #levels.append(range.label.context)
    #levels = list(set(levels))

    ## Retrieve data
    #all_boundaries = []
    #all_labels = []
    #for level in levels:
        #boundaries = []
        #labels = []
        #for range in est.data:
            #if level == range.label.context:
                #boundaries.append([range.start.value, range.end.value])
                #if labels_id is not None:
                    #labels.append(range.label.value)
        #all_boundaries.append(np.asarray(boundaries))
        #all_labels.append(np.asarray(labels, dtype=int))

    ## If there is only one level, return np.arrays instead of lists
    #if len(levels) == 1:
        #all_boundaries = all_boundaries[0]
        #all_labels = all_labels[0]

    all_boundaries, all_labels = est.data.to_interval_values()

    return all_boundaries, all_labels


def get_algo_ids(est_file):
    """Gets the algorithm ids that are contained in the est_file."""
    with open(est_file, "r") as f:
        est_data = json.load(f)
        algo_ids = est_data["boundaries"].keys()
    return algo_ids


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
    """
    # Dataset path
    ds_path = os.path.dirname(os.path.dirname(audio_path))

    # Read references
    jam_path = os.path.join(ds_path, msaf.Dataset.references_dir,
                            os.path.basename(audio_path)[:-4] +
                            msaf.Dataset.references_ext)

    try:
        jam = jams.load(jam_path)
        ann = jam.search(namespace='segment_.*')[annotator_id]
        ref_inters, ref_labels = ann.data.to_interval_values()
    except:
        # TODO: better exception handling
        logging.warning("Reference not found in %s" % jam_path)
        return []

    # Intervals to times
    ref_times = utils.intervals_to_times(ref_inters)

    return ref_times, ref_labels


def read_ref_labels(audio_path):
    """Reads the annotated labels from the given audio path."""
    ref_times, ref_labels = read_references(audio_path)
    return ref_labels


def read_ref_int_labels(audio_path):
    """Reads the annotated labels using unique integers as identifiers
    instead of strings."""
    ref_labels = read_ref_labels(audio_path)
    labels = []
    label_dict = {}
    k = 1
    for ref_label in ref_labels:
        if ref_label in label_dict.keys():
            labels.append(label_dict[ref_label])
        else:
            label_dict[ref_label] = k
            labels.append(k)
            k += 1
    return labels


def align_times(times, frames):
    """Aligns the times to the closes frame times (e.g. beats)."""
    dist = np.minimum.outer(times, frames)
    bound_frames = np.argmax(np.maximum(0, dist), axis=1)
    return np.unique(bound_frames)


def read_ref_bound_frames(audio_path, beats):
    """Reads the corresponding references file to retrieve the boundaries
        in frames."""
    ref_times, ref_labels = read_references(audio_path)
    # align with beats
    bound_frames = align_times(ref_times, beats)
    return bound_frames


def get_features(audio_path, annot_beats=False, framesync=False,
                 pre_features=None):
    """
    Gets the features of an audio file given the audio_path.

    Parameters
    ----------
    audio_path: str
        Path to the audio file.
    annot_beats: bool
        Whether to use annotated beats or not.
    framesync: bool
        Whether to use framesync features or not.
    pre_features: dict
        Pre computed features as a dictionary.
        `None` for reading them form the json file.

    Return
    ------
    C: np.array((N, 12))
        (Beat-sync) Chromagram
    M: np.array((N, 13))
        (Beat-sync) MFCC
    T: np.array((N, 6))
        (Beat-sync) Tonnetz
    cqt: np.array((N, msaf.Anal.cqt_bins))
        (Beat-sync) Constant-Q transform
    beats: np.array(T)
        Beats in seconds
    dur: float
        Song duration
    analysis : dict
        Parameters of analysis of track (e.g. sampling rate)
    """
    if pre_features is None:
        # Dataset path
        ds_path = os.path.dirname(os.path.dirname(audio_path))

        # Read Estimations
        features_path = os.path.join(ds_path, msaf.Dataset.features_dir,
            os.path.basename(audio_path)[:-4] + msaf.Dataset.features_ext)
        with open(features_path, "r") as f:
            feats = json.load(f)

        # Beat Synchronous Feats
        if framesync:
            feat_str = "framesync"
            beats = None
        else:
            if annot_beats:
                # Read references
                try:
                    annotation_path = os.path.join(
                        ds_path, msaf.Dataset.references_dir,
                        os.path.basename(audio_path)[:-4] +
                        msaf.Dataset.references_ext)
                    # TODO: Correct exception handling
                    jam = jams.load(annotation_path)
                except:
                    raise RuntimeError("No references found in file %s" %
                                    annotation_path)

                feat_str = "ann_beatsync"
                beats = []
                beat_data = jam.beats[0].data
                if beat_data == []:
                    raise ValueError
                for data in beat_data:
                    beats.append(data.time.value)
                beats = np.unique(beats)
            else:
                feat_str = "est_beatsync"
                beats = np.asarray(feats["beats"]["times"])
        C = np.asarray(feats[feat_str]["hpcp"])
        M = np.asarray(feats[feat_str]["mfcc"])
        T = np.asarray(feats[feat_str]["tonnetz"])
        cqt = np.asarray(feats[feat_str]["cqt"])
        analysis = feats["analysis"]
        dur = analysis["dur"]

        # Frame times might be shorter than the actual number of features.
        if framesync:
            frame_times = utils.get_time_frames(dur, analysis)
            C = C[:len(frame_times)]
            M = M[:len(frame_times)]
            T = T[:len(frame_times)]

    else:
        feat_prefix = ""
        if not framesync:
            feat_prefix = "bs_"
        C = pre_features["%shpcp" % feat_prefix]
        M = pre_features["%smfcc" % feat_prefix]
        T = pre_features["%stonnetz" % feat_prefix]
        cqt = pre_features["%scqt" % feat_prefix]
        beats = pre_features["beats"]
        dur = pre_features["anal"]["dur"]
        analysis = pre_features["anal"]

    return C, M, T, cqt, beats, dur, analysis


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
        Additional search parameters. E.g. {"feature" : "hpcp"}.

    Returns
    -------
    ann : jams.Annotation
        Found estimation.
        `None` if it couldn't be found.
    """
    # Use handy JAMS search interface
    ann = jam.search(namespace="segment_open").\
        search(**{"Sandbox.boundaries_id": boundaries_id})
    if labels_id is not None:
        ann = ann.search(**{"Sandbox.labels_id": labels_id})
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
        inters = utils.times_to_intervals(times)
        assert len(inters) == len(labels), "Number of boundary intervals " \
            "(%d) and labels (%d) do not match" % (len(inters), len(labels))
        # Put into lists to simplify the writing process later
        inters = [inters]
        labels = [labels]
    else:
        inters = []
        for level in range(len(times)):
            est_inters = utils.times_to_intervals(times[level])
            inters.append(est_inters)
            assert len(inters[level]) == len(labels[level]), \
                "Number of boundary intervals (%d) and labels (%d) do not " \
                "match" % (len(inters[level]), len(labels[level]))

    # Create new estimation
    ann = jams.Annotation(namespace="segment_open")

    # Find estimation in file
    if os.path.isfile(file_struct.est_file):
        jam = jams.load(file_struct.est_file)
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
        if level_labels is None:
            label = np.ones(len(inters)) * -1
        for bound_inter, label in zip(level_inters, level_labels):
            dur = float(bound_inter[1]) - float(bound_inter[0])
            ann.append(time=bound_inter[0],
                       duration=dur,
                       value=str(int(label)))
            # TODO: Hierarchical Level
            #segment.label.context = "level_%d" % i

    # Write results
    jam.save(file_struct.est_file)


def get_all_est_boundaries(est_file, annot_beats, algo_ids=None,
                           annotator_id=0):
    """Gets all the estimated boundaries for all the algorithms.

    Parameters
    ----------
    est_file: str
        Path to the estimated JAMS file.
    annot_beats: bool
        Whether to use the annotated beats or not.
    algo_ids : list
        List of algorithm ids to to read boundaries from.
        If None, all algorithm ids are read.

    Returns
    -------
    all_boundaries: list
        A list of np.arrays containing the times of the boundaries, one array
        for each algorithm
    """
    all_boundaries = []

    # Get GT boundaries
    jam_file = os.path.join(os.path.dirname(est_file), "..",
                            msaf.Dataset.references_dir,
                            os.path.basename(est_file))
    jam = jams.load(jam_file)
    ann = jam.search(namespace='segment_.*')[annotator_id]
    ann_inter, ann_labels = ann.data.to_interval_values()
    ann_times = utils.intervals_to_times(ann_inter)
    all_boundaries.append(ann_times)

    # Estimations
    if algo_ids is None:
        algo_ids = get_algo_ids(est_file)
    for algo_id in algo_ids:
        est_inters, est_labels = read_estimations(
            est_file, algo_id, annot_beats, feature=msaf.feat_dict[algo_id])
        if len(est_inters) == 0:
            logging.warning("no estimations for algorithm: %s" % algo_id)
            continue
        boundaries = utils.intervals_to_times(est_inters)
        all_boundaries.append(boundaries)
    return all_boundaries


def get_all_est_labels(est_file, annot_beats, algo_ids=None, annotator_id=0):
    """Gets all the estimated boundaries for all the algorithms.

    Parameters
    ----------
    est_file: str
        Path to the estimated file (JSON file)
    annot_beats: bool
        Whether to use the annotated beats or not.
    algo_ids : list
        List of algorithm ids to to read boundaries from.
        If None, all algorithm ids are read.
    annotator_id : int
        Identifier of the annotator.

    Returns
    -------
    gt_times:  np.array
        Ground Truth boundaries in times.
    all_labels: list
        A list of np.arrays containing the labels corresponding to the ground
        truth boundaries.
    """
    all_labels = []

    # Get GT boundaries and labels
    jam_file = os.path.join(os.path.dirname(est_file), "..",
                            msaf.Dataset.references_dir,
                            os.path.basename(est_file))
    jam = jams.load(jam_file)
    ann = jam.search(namespace='segment_.*')[annotator_id]
    ann_inter, ann_labels = ann.data.to_interval_values()
    gt_times = utils.intervals_to_times(ann_inter)
    all_labels.append(ann_labels)

    # Estimations
    if algo_ids is None:
        algo_ids = get_algo_ids(est_file)
    for algo_id in algo_ids:
        est_inters, est_labels = read_estimations(
            est_file, algo_id, annot_beats, annot_bounds=True, bounds=False,
            feature=msaf.feat_dict[algo_id])
        if len(est_labels) == 0:
            logging.warning("no estimations for algorithm: %s" % algo_id)
            continue
        all_labels.append(est_labels)
    return gt_times, all_labels


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
    if boundaries_id != "gt":
        bound_config = \
            eval(msaf.algorithms.__name__ + "." + boundaries_id).config
        config.update(bound_config)
    if labels_id is not None:
        label_config = \
            eval(msaf.algorithms.__name__ + "." + labels_id).config
        config.update(label_config)
    return config


def filter_by_artist(file_structs, artist_name="The Beatles"):
    """Filters data set files by artist name."""
    new_file_structs = []
    for file_struct in file_structs:
        jam = jams.load(file_struct.ref_file)
        if jam.file_metadata.artist == artist_name:
            new_file_structs.append(file_struct)
    return new_file_structs


def get_SALAMI_internet(file_structs):
    """Gets the SALAMI Internet subset from SALAMI (bit of a hack...)"""
    new_file_structs = []
    for file_struct in file_structs:
        num = int(os.path.basename(file_struct.est_file).split("_")[1].
                  split(".")[0])
        if num >= 956 and num <= 1498:
            new_file_structs.append(file_struct)
    return new_file_structs


def get_dataset_files(in_path, ds_name="*"):
    """Gets the files of the dataset with a prefix of ds_name."""

    # All datasets
    ds_dict = {
        "Beatles"   : "Isophonics",
        "Cerulean"  : "Cerulean",
        "Epiphyte"  : "Epiphyte",
        "Isophonics": "Isophonics",
        "SALAMI"    : "SALAMI",
        "SALAMI-i"  : "SALAMI",
        "*"         : "*"
    }

    try:
        prefix = ds_dict[ds_name]
    except KeyError:
        raise RuntimeError("Dataset %s is not valid. Valid datasets are: %s" %
                           (ds_name, ds_dict.keys()))

    # Get audio files
    audio_files = []
    for ext in msaf.Dataset.audio_exts:
        audio_files += glob.glob(os.path.join(in_path, msaf.Dataset.audio_dir,
                                              ("%s_*" + ext) % prefix))

    # Check for datasets with different prefix
    if len(audio_files) == 0:
        for ext in msaf.Dataset.audio_exts:
            audio_files += glob.glob(os.path.join(in_path,
                                                  msaf.Dataset.audio_dir,
                                                  "*" + ext))

    # Make sure directories exist
    utils.ensure_dir(os.path.join(in_path, msaf.Dataset.features_dir))
    utils.ensure_dir(os.path.join(in_path, msaf.Dataset.estimations_dir))
    utils.ensure_dir(os.path.join(in_path, msaf.Dataset.references_dir))

    # Get the file structs
    file_structs = []
    for audio_file in audio_files:
        file_structs.append(FileStruct(audio_file))

    # Filter by the beatles
    if ds_name == "Beatles":
        file_structs = filter_by_artist(file_structs, "The Beatles")

    # Salami Internet hack
    if ds_name == "SALAMI-i":
        file_structs = get_SALAMI_internet(file_structs)

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
    # TODO: Hier
    def get_levels():
        """Obtains the set of unique levels contained in the jams
            sorted by the number of segments they contain.

        Returns
        -------
        levels : np.array
            Level identifiers for the entire hierarchy.
        """
        levels = []
        jam = jams2.load(jams_file)
        annotation = jam.sections[annotation_id]
        for segment in annotation.data:
            if segment.label.context not in exclude_levels:
                levels.append(segment.label.context)
        c = Counter(levels)     # Count frequency
        # Sort
        return np.asarray(list(dict(c).keys()))[np.argsort(list(c.values()))]

    def get_segments_in_level(level):
        """Gets the segments of a specific level.

        Paramters
        ---------
        level : str
            Indentifier of the level within the jams file.

        Returns
        -------
        times : np.array
            Boundary times in seconds for the given level.
        labels : np.array
            Labels for the given level.
        """
        intervals, labels = jams2.converters.load_jams_range(jams_file,
                "sections", annotator=annotation_id, context=level)
        times = utils.intervals_to_times(intervals)
        return np.array(times), np.array(labels)

    # Get the levels of the annotations in the jams file
    hier_levels = get_levels()

    # Get the boundaries and labels for each level
    hier_bounds = []
    hier_labels = []
    for level in hier_levels:
        bound_times, labels = get_segments_in_level(level)
        hier_bounds.append(bound_times)
        hier_labels.append(labels)

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
    return float(feats["analysis"]["dur"])
