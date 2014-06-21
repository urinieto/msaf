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


def read_boundaries(est_file, alg_id, annot_beats, **params):
    """Reads the boundaries from an estimated file.

    Parameters
    ----------
    est_file: str
        Path to the estimated file (JSON file).
    alg_id: str
        Algorithm ID from which to retrieve the boundaries. E.g. serra
    annot_beats: bool
        Whether to retrieve the boundaries using annotated beats or not.
    params: dict
        Additional search parameters. E.g. {"features" : "hpcp"}

    Returns
    -------
    bounds = np.array(N,2)
        Set of read boundaries in an interval format (start, end).
    """
    est_data = json.load(open(est_file, "r"))

    if alg_id not in est_data["boundaries"]:
        logging.error("Estimation not found for algorithm %s in %s" %
                      (est_file, alg_id))
        return []

    bounds = []
    for alg in est_data["boundaries"][alg_id]:
        if alg["annot_beats"] == annot_beats:
            # Match non-empty parameters
            found = True
            for key in params:
                if params[key] != "" and alg[key] != params[key]:
                    found = False

            if found:
                # Sort the boundaries by time
                bounds = np.sort(np.array(alg["data"]))
                break

    # Convert to interval
    bounds = np.asarray(zip(bounds[:-1], bounds[1:]))
    return bounds


def get_algo_ids(est_file):
    """Gets the algorithm ids that are contained in the est_file."""
    with open(est_file, "r") as f:
        est_data = json.load(f)
        algo_ids = est_data["boundaries"].keys()
    return algo_ids


def read_annot_bound_frames(audio_path, beats):
    """Reads the corresponding annotations file to retrieve the boundaries
        in frames."""

    # Dataset path
    ds_path = os.path.dirname(os.path.dirname(audio_path))

    # Read annotations
    jam_path = os.path.join(ds_path, "annotations",
                            os.path.basename(audio_path)[:-4] + ".jams")
    ds_prefix = os.path.basename(audio_path).split("_")[0]
    ann_times, ann_labels = jams2.converters.load_jams_range(
        jam_path, "sections", context=prefix_dict[ds_prefix])
    try:
        ann_times, ann_labels = jams2.converters.load_jams_range(
            jam_path, "sections", context=prefix_dict[ds_prefix])
    except:
        logging.warning("Annotation not found in %s" % jam_path)
        return []

    # align with beats
    ann_times = np.concatenate((ann_times.flatten()[::2], [ann_times[-1, -1]]))
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


def create_estimation(times, annot_beats, **params):
    """Creates a new estimation (dictionary)."""
    est = {}
    est["annot_beats"] = annot_beats
    est["timestamp"] = datetime.datetime.today().strftime("%Y/%m/%d %H:%M:%S")
    est["data"] = list(times)
    for key in params:
        est[key] = params[key]
    return est


def save_estimations(out_file, estimations, annot_beats, alg_name,
                     bounds=True, **params):
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
    params : dict
        Extra parameters, algorithm dependant.
    """
    if bounds:
        est_type = "boundaries"
    else:
        est_type = "labels"
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
                        res[est_type][alg_name][i] = \
                            create_estimation(estimations, 
                                              annot_beats, **params)
                        break
                if not found:
                    res[est_type][alg_name].append(create_estimation(
                        estimations, annot_beats, **params))
            else:
                res[est_type][alg_name] = []
                res[est_type][alg_name].append(create_estimation(estimations,
                                                    annot_beats, **params))
        else:
            res[est_type] = {}
            res[est_type][alg_name] = []
            res[est_type][alg_name].append(create_estimation(estimations,
                                                annot_beats, **params))
    else:
        # Create new estimation
        res = {}
        res[est_type] = {}
        res[est_type][alg_name] = []
        res[est_type][alg_name].append(
            create_estimation(estimations, annot_beats, **params))

    # Save dictionary to disk
    with open(out_file, "w") as f:
        json.dump(res, f, indent=2)
