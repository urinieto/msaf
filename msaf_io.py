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

import numpy as np

# Local stuff
import jams


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
        os.path.basename(audio_path)[:-4]+".jams")
    jam = jams.load(annotation_path)

    # Read Estimations
    features_path = os.path.join(ds_path, "features", 
        os.path.basename(audio_path)[:-4]+".json")
    f = open(features_path, "r")
    feats = json.load(f)

    # Beat Synchronous Chroma
    if annot_beats:
        C = np.asarray(feats["ann_beatsync"]["hpcp"])
        M = np.asarray(feats["ann_beatsync"]["mfcc"])
        beats = []
        beat_data = jam.beats[0].data
        if beat_data == []: raise ValueError
        for data in beat_data:
            if data.label.value != -1:
                beats.append(data.time.value)
        beats = np.asarray(beats)
    else:
        C = np.asarray(feats["est_beatsync"]["hpcp"])
        M = np.asarray(feats["est_beatsync"]["mfcc"])
        beats = np.asarray(feats["beats"]["ticks"])

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


def save_boundaries(out_file, times, annot_beats, alg_name, **params):
    """Saves the segment boundary times in the out_file using a JSON format. 
        if file exists, update with new annotation."""
    if os.path.isfile(out_file):
        # Add estimation
        f = open(out_file, "r")
        res = json.load(f)

        # Check if estimation already exists
        if "boundaries" in res.keys():
            if alg_name in res["boundaries"]:
                found = False
                for i, est in enumerate(res["boundaries"][alg_name]):
                    if est["annot_beats"] == annot_beats:
                        found = True
                        res["boundaries"][alg_name][i] = \
                            create_estimation(times, annot_beats, **params)
                        break
                if not found:
                    res["boundaries"][alg_name].append(create_estimation(times, 
                                                annot_beats, **params))
            else:
                res["boundaries"][alg_name] = []
                res["boundaries"][alg_name].append(create_estimation(times,
                                                    annot_beats, **params))
        else:
            res["boundaries"] = {}
            res["boundaries"][alg_name] = []
            res["boundaries"][alg_name].append(create_estimation(times,
                                                annot_beats, **params))
        f.close()
    else:
        # Create new estimation
        res = {}
        res["boundaries"] = {}
        res["boundaries"][alg_name] = []
        res["boundaries"][alg_name].append(create_estimation(times,
                                            annot_beats, **params))

    # Save dictionary to disk
    f = open(out_file, "w")
    json.dump(res, f, indent=2)
    f.close()