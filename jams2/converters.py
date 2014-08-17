"""
Converters for JAMS.
"""

import logging
import numpy as np

# Local stuff
from pyjams import Jams


def get_annotator_idx(jam, feature_name, annotator_name, filename):
    """Gets the annotator inddex of the annotation annotated by
    annotator_name."""
    annotator_idx = -1
    for i, annotator in enumerate(jam[feature_name]):
        if annotator.annotation_metadata.annotator.name == annotator_name:
            annotator_idx = i
            break
    if annotator_idx == -1:
        logging.warning("Annotator %s not found in %s" % (annotator_name,
                                                          filename))
    return annotator_idx


def load_jams_range(filename, feature_name, annotator=0, annotator_name=None,
                    converter=None, context='large_scale', confidence=False):
    """Import specific data from a JAMS annotation file. It imports range
        data, i.e., data that spans within two time points and it has a label
        associated with it.

        :parameters:
        - filename : str
        Path to the annotation file.

        - feature_name: str
        The key to the JAMS range feature to be extracted
        (e.g. "sections", "chords")

        - annotator: int
        The idx of the annotator from which to extract the annotations.

        - annotator_name: str
        The name of the annotator from which to extract the annotations. If not
        None, this parameter overwrites the "annotator".

        - converter : function
        Function to convert time-stamp data into numerics. Defaults to float().

        - context : str
        Context of the labels to be extracted (e.g. "large_scale", "function").

        :returns:
        - event_times : np.ndarray
        array of event times (float).

        - event_labels : list of str
        list of corresponding event labels.
    """

    if converter is None:
        converter = float

    jam = Jams.load(filename)
    if annotator_name is not None:
        annotator = get_annotator_idx(jam, feature_name, annotator_name,
                                      filename)

    try:
        jam = Jams.load(filename)
    except:
        print "Error: could not open %s (JAMS module not installed?)" % \
            filename
        return [], []

    times   = []
    labels  = []
    conf    = []
    if len(jam[feature_name]) == 0:
        print "Warning: %s empty in file %s" % (feature_name, filename)
        return []

    for data in jam[feature_name][annotator].data:
        if data.label.context == context:
            times.append([converter(data.start.value),
                          converter(data.end.value)])
            conf.append([converter(data.start.confidence),
                         converter(data.end.confidence)])
            labels.append(data.label.value)

    times = np.asarray(times)

    if confidence:
        return times, labels, conf
    else:
        return times, labels
