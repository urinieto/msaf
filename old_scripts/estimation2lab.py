#!/usr/bin/env python
"""
This script converts the estimations to lab files in order to, e.g. analize them
in Sonic Visualizer
"""

__author__ = "Oriol Nieto"
__copyright__ = "Copyright 2014, Music and Audio Research Lab (MARL)"
__license__ = "GPL"
__version__ = "1.0"
__email__ = "oriol@nyu.edu"

import argparse
import logging
import os
import time
import utils

# Local stuff
import input_output as io


def est2lab(est_file, lab_file, annot_beats=False, bound_name="olda",
            label_name="olda"):
    """Estimation file (JSON) to lab file."""

    bounds = io.read_estimations(est_file, bound_name, annot_beats)
    bounds = zip(bounds, bounds[1:])  # Lab format

    # TODO: Labels
    labels = ["TODO"]*(len(bounds))

    # Create output lab string
    out_str = ""
    for bound, label in zip(bounds, labels):
        out_str += str(bound[0]) + "\t" + str(bound[1]) + "\t" + label + "\n"

    # Write lab file
    with open(lab_file, "w") as f:
        f.write(out_str)


def process(in_path, out_path, **args):
    """Main process."""

    # If in_path it's a file, we only compute one file
    if os.path.isfile(in_path):
        est2lab(in_path, out_path, **args)

    elif os.path.isdir(in_path):

        # Check that in_path exists
        utils.ensure_dir(out_path)

        # TODO


def main():
    """Main function to parse the arguments and call the main process."""
    parser = argparse.ArgumentParser(description=
        "Extracts a set of features from the Segmentation dataset or a given "
        "audio file and saves them into the 'features' folder of the dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("in_path",
                        action="store",
                        help="Input estimation dir or json file")
    parser.add_argument("bound_name",
                        action="store",
                        help="The name of the boundaries algorithm")
    parser.add_argument("label_name",
                        action="store",
                        help="The name of the labeling algorithm")
    parser.add_argument("-o",
                        dest="out_path",
                        action="store",
                        default="labs",
                        help="Output dir or lab file")
    parser.add_argument("-b",
                        action="store_true",
                        dest="annot_beats",
                        help="Estimated beats",
                        default=False)
    args = parser.parse_args()
    start_time = time.time()

    # Setup the logger
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s',
        level=logging.INFO)

    # Run the algorithm
    process(args.in_path, args.out_path, annot_beats=args.annot_beats,
            bound_name=args.bound_name, label_name=args.label_name)

    # Done!
    logging.info("Done! Took %.2f seconds." % (time.time() - start_time))

if __name__ == '__main__':
    main()
