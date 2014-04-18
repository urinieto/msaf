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
import pylab as plt
from collections import OrderedDict
from collections import Counter
from operator import itemgetter
import pickle
import csv

sys.path.append("../../")
import mir_eval


def get_track_ids(annot_dir):
    """Obtains the track ids of all the files contained in the experiment.

    Parameters
    ----------
    annot_dir: str
        Path to the annotations directory where all the jams file reside.

    Retutns
    -------
    track_ids: list
        List containing all the files in the experiment.
    """
    return glob.glob(os.path.join(annot_dir, "*.jams"))


def analyze_answers(results_dir="results/"):
    """Analyzes the answers of the experiment, contained in csv files."""
    # Get all the folders with the annotators results
    annotator_dirs = glob.glob(os.path.join(results_dir, "*"))
    answers = []
    for annot_dir in annotator_dirs:
        csv_file = os.path.join(annot_dir, "answers.csv")
        print csv_file
        if not os.path.isfile(csv_file):
            continue
        file_reader = csv.reader(open(csv_file, "rU"))
        for fields in file_reader:
            # Ignore the first line
            if fields[0] == "File Name":
                continue
            if fields[0] != "" and fields[1] != "":
                for word in fields[1].split(";"):
                    answers.append(word.strip(" ").lower())

    # Get histogram from the answers
    histo = Counter(answers)

    # Sort and print
    sorted_histo = sorted(histo.items(), key=itemgetter(1), reverse=True)
    for element in sorted_histo:
        logging.info(element)


def compute_mgp(jams_files, annotators, context, trim):
    """Computes the Mean Ground-truth Performance of the experiment
        results."""
    mgp_results = np.empty((0, 6))
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

            FPR = np.vstack((FPR, (F3, P3, R3, F05, P05, R05)))

        if i == 1:
            mgp_results = np.vstack((mgp_results, FPR))
        else:
            if np.asarray([mgp_results, FPR]).ndim != 3:
                logging.warning("Ndim is not valid %d" %
                                np.asarray([mgp_results, FPR]).ndim)
                print len(mgp_results)
                print len(FPR)
                continue
            mgp_results = np.mean([mgp_results, FPR], axis=0)

        FPR = np.mean(FPR, axis=0)
        logging.info("Results for %s:\n\tF3: %.4f, P3: %.4f, R3: %.4f\n"
                     "\tF05: %.4f, P05: %.4f, R05: %.4f" % (
                         annotators.keys()[i], FPR[0], FPR[1], FPR[2], FPR[3],
                         FPR[4], FPR[5]))
    return mgp_results


def analyze_boundaries(annot_dir, trim, annotators):
    """Analyzes the annotated boundaries.

    Parameters
    ----------
    annot_dir: str
        Path to the annotations directory where all the jams file reside.
    trim: boolean
        Whether to trim the first and last boundaries.
    annotators: dict
        Dictionary containing the names and e-mail addresses of the 5
        different annotators.
    """
    # Compute the MGP human results
    jams_files = glob.glob(os.path.join(annot_dir, "*.jams"))
    context = "large_scale"
    dtype = [('F3', float), ('P3', float), ('R3', float), ('F05', float),
             ('P05', float), ('R05', float), ('track_id', '<U400')]
    mgp_results = compute_mgp(jams_files, annotators, context, trim)
    mgp_results = mgp_results.tolist()
    track_ids = get_track_ids(annot_dir)
    for i in xrange(len(track_ids)):
        mgp_results[i].append(track_ids[i])
        mgp_results[i] = tuple(mgp_results[i])
    mgp_results = np.asarray(mgp_results, dtype=dtype)
    mgp_results = np.sort(mgp_results, order='F3')

    # Get MGP machine results
    mgp_results_machine = pickle.load(open(
        "../notes/mgp_experiment_machine.pk", "r"))

    # Plot results
    mgp_results_machine = np.sort(mgp_results_machine, order="track_id")
    mgp_results = np.sort(mgp_results, order="track_id")
    print "Humans", mgp_results
    #print "Machines", mgp_results_machine
    fig, ax = plt.subplots()
    ax.scatter(mgp_results_machine["F3"], mgp_results["F3"])
    ax.plot([0, 1], [0, 1])
    ax.set_ylabel("Human results")
    ax.set_xlabel("Machine results")
    ax.set_xticks(np.arange(0, 1.1, .1))
    ax.set_yticks(np.arange(0, 1.1, .1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.show()


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
    annotators["Shuli"] = {
        "name"  : "Shuli",
        "email" : "luiseslt@gmail.com"
    }

    # Analyze the answers
    analyze_answers()

    # Analyze the boundaries
    analyze_boundaries(annot_dir, trim, annotators)


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
