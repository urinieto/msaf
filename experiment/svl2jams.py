#!/usr/bin/env python
"""
Converts an svl file into a JAMS annotation
"""

__author__      = "Oriol Nieto"
__copyright__   = "Copyright 2014, Music and Audio Research Lab (MARL)"
__license__     = "GPL"
__version__     = "1.0"
__email__       = "oriol@nyu.edu"

import argparse
import logging
import os
import time
import json
import xml.etree.ElementTree as ET

import jams2

annotators = {
    "Colin" : {
        "name"  : "Colin",
        "email" : "colin.z.hua@gmail.com"
    },
    "Eleni" : {
        "name"  : "Eleni",
        "email" : "evm241@nyu.edu"
    },
    "Evan" : {
        "name"  : "Evan",
        "email" : "esj254@nyu.edu"
    },
    "John" : {
        "name"  : "John",
        "email" : "johnturner@me.com"
    },
    "Shuli" : {
        "name"  : "Shuli",
        "email" : "luiseslt@gmail.com"
    }
}


def create_annotation(root, annotator_id, jam_file, context):
    """Creates an annotation from the given root of an XML svl file."""

    # Load jam file
    jam = jams2.load(jam_file)

    # If annotation exists, replace it
    annot = None
    for section in jam.sections:
        if section.annotation_metadata.annotator.name == annotator_id:
            annot = section
            # If this context already exists, do nothing
            for data in annot.data:
                if data.label.context == context:
                    return
            break

    # Create Annotation if needed
    if annot is None:
        annot = jam.sections.create_annotation()

    # Create Metadata
    annot.annotation_metadata.annotator = annotators[annotator_id]
    # TODO: More metadata

    # Get sampling rate from XML root
    sr = float(root.iter("model").next().attrib["sampleRate"])

    # Create datapoints from the XML root
    points = root.iter("point")
    point = points.next()
    start = float(point.attrib["frame"]) / sr
    label = point.attrib["label"]
    for point in points:
        section = annot.create_datapoint()
        section.start.value = start
        section.end.value = float(point.attrib["frame"]) / sr
        # Make sure upper and lower case are consistent
        if context == "small_scale":
            section.label.value = label.lower()
        elif context == "large_scale":
            section.label.value = label.upper()
        section.label.context = context
        start = float(point.attrib["frame"]) / sr
        label = point.attrib["label"]

    # Save file
    with open(jam_file, "w") as f:
        json.dump(jam, f, indent=2)


def process(in_file, out_file="output.jams"):
    """Main process to convert an svl file to JAMS."""
    # Make sure that the jams exist (we simply have metadata there)
    assert(os.path.isfile(out_file))

    # Parse svl file (XML)
    tree = ET.parse(in_file)
    root = tree.getroot()

    # Retrieve context from in_file name
    contexts = {
        "s" : "small_scale",
        "l" : "large_scale"
    }
    context_id = in_file[:-4].split("_")[-1].lower()

    # Retrieve annotator id from in_file name
    annotator_id = os.path.basename(os.path.dirname(in_file))

    # Create Annotation
    create_annotation(root, annotator_id, out_file, contexts[context_id])


def main():
    """Main function to convert the annotation."""
    parser = argparse.ArgumentParser(description=
        "Converts a Sonic Visualizer annotation into a JAMS.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("in_path",
                        action="store",
                        help="Input svl file.")
    parser.add_argument("-o",
                        action="store",
                        dest="out_file",
                        help="Output file",
                        default="output.jams")
    args = parser.parse_args()
    start_time = time.time()

    # Setup the logger
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s',
                        level=logging.INFO)

    # Run the algorithm
    process(args.in_path, out_file=args.out_file)

    # Done!
    logging.info("Done! Took %.2f seconds." % (time.time() - start_time))

if __name__ == '__main__':
    main()
