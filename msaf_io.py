"""
These set of functions help the algorithms of MSAF to read and write files
of the Segmentation Dataset.
"""

__author__ = "Oriol Nieto"
__copyright__ = "Copyright 2014, Music and Audio Research Lab (MARL)"
__license__ = "GPL"
__version__ = "1.0"
__email__ = "oriol@nyu.edu"

import datetime
import json
import os
import logging

import numpy as np

# Local stuff
import jams2


prefix_dict = {
    "Cerulean"      : "large_scale",
    "Epiphyte"      : "function",
    "Isophonics"    : "function",
    "SALAMI"        : "large_scale"
}


def read_estimations(est_file, alg_id, annot_beats, annot_bounds=False,
                     bounds=True, **params):
    """Reads the estimations (either boundaries or labels) from an estimated
    file.

    Parameters
    ----------
    est_file: str
        Path to the estimated file (JSON file).
    alg_id: str
        Algorithm ID from which to retrieve the boundaries. E.g. serra
    annot_beats: bool
        Whether to retrieve the boundaries using annotated beats or not.
    annot_bounds: bool
        Whether to retrieve the boundaries using annotated bounds or not.
    bounds : bool
        Whether to extract boundaries or labels
    params: dict
        Additional search parameters. E.g. {"features" : "hpcp"}

    Returns
    -------
    estimations = np.array(N,2) or np.array(N)
        Set of read estimations:
            If boundaries: interval format (start, end).
            If labels: flattened format.
    """
    est_data = json.load(open(est_file, "r"))

    if bounds:
        est_type = "boundaries"
    else:
        est_type = "labels"

    if est_type not in est_data.keys() or alg_id not in est_data[est_type]:
        if bounds:
            logging.error("Estimation not found for algorithm %s in %s" %
                          (est_file, alg_id))
        return []

    estimations = []
    for alg in est_data[est_type][alg_id]:
        if alg["annot_beats"] == annot_beats:
            # Match non-empty parameters
            found = True
            for key in params:
                if params[key] != "" and alg[key] != params[key]:
                    found = False

            if not bounds:
                if "annot_bounds" in alg.keys():
                    if alg["annot_bounds"] != annot_bounds:
                        found = False

            if found:
                estimations = np.array(alg["data"])
                if bounds:
                    # Sort the boundaries by time
                    estimations = np.sort(estimations)
                break

    # Convert to interval if needed
    if bounds:
        estimations = np.asarray(zip(estimations[:-1], estimations[1:]))

    return estimations


def get_algo_ids(est_file):
    """Gets the algorithm ids that are contained in the est_file."""
    with open(est_file, "r") as f:
        est_data = json.load(f)
        algo_ids = est_data["boundaries"].keys()
    return algo_ids


def read_annotations(audio_path):
    """Reads the boundary frames and the labels."""
    # Dataset path
    ds_path = os.path.dirname(os.path.dirname(audio_path))

    # Read annotations
    jam_path = os.path.join(ds_path, "annotations",
                            os.path.basename(audio_path)[:-4] + ".jams")
    ds_prefix = os.path.basename(audio_path).split("_")[0]
    ann_inters, ann_labels = jams2.converters.load_jams_range(
        jam_path, "sections", context=prefix_dict[ds_prefix])
    try:
        ann_inters, ann_labels = jams2.converters.load_jams_range(
            jam_path, "sections", context=prefix_dict[ds_prefix])
    except:
        logging.warning("Annotation not found in %s" % jam_path)
        return []

    # Intervals to times
    ann_times = np.concatenate((ann_inters.flatten()[::2],
                                [ann_inters[-1, -1]]))

    return ann_times, ann_labels


def read_annot_labels(audio_path):
    """Reads the annotated labels from the given audio path."""
    ann_times, ann_labels = read_annotations(audio_path)
    return ann_labels


def read_annot_int_labels(audio_path):
    """Reads the annotated labels using unique integers as identifiers
    instead of strings."""
    ann_labels = read_annot_labels(audio_path)
    labels = []
    label_dict = {}
    k = 1
    for ann_label in ann_labels:
        if ann_label in label_dict.keys():
            labels.append(label_dict[ann_label])
        else:
            label_dict[ann_label] = k
            labels.append(k)
            k += 1
    return labels


def read_annot_bound_frames(audio_path, beats):
    """Reads the corresponding annotations file to retrieve the boundaries
        in frames."""

    ann_times, ann_labels = read_annotations(audio_path)

    # align with beats
    dist = np.minimum.outer(ann_times, beats)
    bound_frames = np.argmax(np.maximum(0, dist), axis=1)

    return bound_frames


def get_features(audio_path, annot_beats=False):
    """
    Gets the features of an audio file given the audio_path.

    Parameters
    ----------
    audio_path: str
        Path to the audio file.

    annot_beats: bool
        Whether to use annotated beats or not.

    Return
    ------
    C: np.array((T, 12))
        Beat-sync Chromagram

    M: np.array((T, 13))
        Beat-sync MFCC

    beats: np.array(T)
        Beats in seconds

    dur: float
        Song duration

    """
    # Dataset path
    ds_path = os.path.dirname(os.path.dirname(audio_path))

    # Read annotations
    annotation_path = os.path.join(ds_path, "annotations",
        os.path.basename(audio_path)[:-4] + ".jams")
    jam = jams2.load(annotation_path)

    # Read Estimations
    features_path = os.path.join(ds_path, "features",
        os.path.basename(audio_path)[:-4] + ".json")
    f = open(features_path, "r")
    feats = json.load(f)

    # Beat Synchronous Chroma
    if annot_beats:
        C = np.asarray(feats["ann_beatsync"]["hpcp"])
        M = np.asarray(feats["ann_beatsync"]["mfcc"])
        beats = []
        beat_data = jam.beats[0].data
        if beat_data == []:
            raise ValueError
        for data in beat_data:
            beats.append(data.time.value)
        beats = np.unique(beats)
    else:
        C = np.asarray(feats["est_beatsync"]["hpcp"])
        M = np.asarray(feats["est_beatsync"]["mfcc"])
        beats = np.asarray(feats["beats"]["ticks"])[0]

    # Duration
    dur = jam.metadata.duration

    f.close()

    return C, M, beats, dur


def create_estimation(times, annot_beats, annot_bounds, **params):
    """Creates a new estimation (dictionary)."""
    est = {}
    est["annot_beats"] = annot_beats
    est["annot_bounds"] = annot_bounds
    est["timestamp"] = datetime.datetime.today().strftime("%Y/%m/%d %H:%M:%S")
    est["data"] = list(times)
    for key in params:
        est[key] = params[key]
    return est


def save_estimations(out_file, estimations, annot_beats, alg_name,
                     bounds=True, annot_bounds=False, **params):
    """Saves the segment estimations (either boundaries or labels) in the
        out_file using a JSON format.
        If file exists, update with new annotation.

    Parameters
    ----------
    out_file : str
        Path to the output JSON file.
    estimations : list
        Set of boundary times or labels to be stored.
    annot_beats : boolean
        Whether the estimations were obtained using annotated beats.
    alg_name : str
        Algorithm identifier.
    bounds : boolean
        Whether the estimations represent boundaries or lables.
    annot_bounds : boolean
        Whether the labels were obtained using the annotated boundaries.
    params : dict
        Extra parameters, algorithm dependant.
    """
    if bounds:
        est_type = "boundaries"
    else:
        est_type = "labels"

    # Create new estimation
    new_est = create_estimation(estimations, annot_beats, annot_bounds,
                                **params)

    # Find correct place to store it within the estimation file
    if os.path.isfile(out_file):
        # Add estimation
        res = json.load(open(out_file, "r"))

        # Check if estimation already exists
        if est_type in res.keys():
            if alg_name in res[est_type]:
                found = False
                for i, est in enumerate(res[est_type][alg_name]):
                    if est["annot_beats"] == annot_beats:
                        found = True
                        for key in params:
                            if params[key] != est[key]:
                                found = False
                                break
                        if not found:
                            continue
                        # Check for annot_bounds only if saving labels
                        if not bounds:
                            if "annot_bounds" in est.keys():
                                if est["annot_bounds"] != annot_bounds:
                                    found = False
                                    continue
                        res[est_type][alg_name][i] = new_est
                        break
                if not found:
                    res[est_type][alg_name].append(new_est)
            else:
                res[est_type][alg_name] = []
                res[est_type][alg_name].append(new_est)
        else:
            # Esitmation doesn't exist for this type of feature
            res[est_type] = {}
            res[est_type][alg_name] = []
            res[est_type][alg_name].append(new_est)
    else:
        # Create new estimation file
        res = {}
        res[est_type] = {}
        res[est_type][alg_name] = []
        res[est_type][alg_name].append(new_est)

    # Save dictionary to disk
    with open(out_file, "w") as f:
        json.dump(res, f, indent=2)
