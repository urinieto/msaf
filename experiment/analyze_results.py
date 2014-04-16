#!/usr/bin/env python
"""
Analyzes the experiment results:

    - Read the jams annotations and compare them using various metrics.
    - Plot the correlation between MPG and MMA.
"""

__author__      = "Oriol Nieto"
__copyright__   = "Copyright 2014, Music and Audio Research Lab (MARL)"
__license__     = "GPL"
__version__     = "1.0"
__email__       = "oriol@nyu.edu"

import argparse
import glob
import logging
import os
import sys
import time
import numpy as np
from collections import OrderedDict

sys.path.append("../../")
import mir_eval


def process(annot_dir, trim=False):
    """Main process to parse all the results from the results_dir
        to out_dir."""

    # Now compute all the metrics
    annotators = OrderedDict()
    annotators["GT"] = {
        "name"  : "GT",
        "email" : "TODO"
    }
    annotators["Colin"] = {
        "name"  : "Colin",
        "email" : "colin.z.hua@gmail.com"
    }
    annotators["Eleni"] = {
        "name"  : "Eleni",
        "email" : "evm241@nyu.edu"
    }
    annotators["Evan"] = {
        "name"  : "Evan",
        "email" : "esj254@nyu.edu"
    }
    annotators["John"] = {
        "name"  : "John",
        "email" : "johnturner@me.com"
    }
    jams_files = glob.glob(os.path.join(annot_dir, "*.jams"))
    context = "large_scale"
    for i in xrange(1, len(annotators.keys())):
        FPR = np.empty((0, 6))
        for jam_file in jams_files:
            ann_times, ann_labels = mir_eval.io.load_jams_range(jam_file,
                            "sections", annotator=0, context=context)
            try:
                est_times, est_labels = mir_eval.io.load_jams_range(jam_file,
                            "sections", annotator=i, context=context)
            except:
                logging.warning("Couldn't read annotator %d in JAMS %s" %
                                (i, jam_file))
                continue
            if len(ann_times) == 0:
                ann_times, ann_labels = mir_eval.io.load_jams_range(jam_file,
                            "sections", annotator=0, context="function")
            if len(est_times) == 0:
                logging.warning("No annotation in file %s for annotator %s." %
                                (jam_file, annotators.keys()[i]))
                continue

            ann_times = np.asarray(ann_times)
            est_times = np.asarray(est_times)
            #print ann_times.shape, jam_file
            P3, R3, F3 = mir_eval.segment.boundary_detection(
                ann_times, est_times, window=3, trim=trim)
            P05, R05, F05 = mir_eval.segment.boundary_detection(
                ann_times, est_times, window=0.5, trim=trim)

            FPR = np.vstack((FPR, [F3, P3, R3, F05, P05, R05]))

        FPR = np.mean(FPR, axis=0)
        print i, annotators.keys()
        logging.info("Results for %s:\n\tF3: %.4f, P3: %.4f, R3: %.4f\n"
                     "\tF05: %.4f, P05: %.4f, R05: %.4f" % (
                         annotators.keys()[i], FPR[0], FPR[1], FPR[2], FPR[3],
                         FPR[4], FPR[5]))


def main():
    """Main function to analyze the experiment results."""
    parser = argparse.ArgumentParser(description=
        "Analyzes the experiment results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("annot_dir",
                        action="store",
                        help="Results directory")
    parser.add_argument("-t",
                        action="store_true",
                        dest="trim",
                        default=False,
                        help="Whether trim the boundaries or not.")
    args = parser.parse_args()
    start_time = time.time()

    # Setup the logger
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s',
                        level=logging.INFO)

    # Run the algorithm
    process(args.annot_dir, trim=args.trim)

    # Done!
    logging.info("Done! Took %.2f seconds." % (time.time() - start_time))

if __name__ == '__main__':
    main()
