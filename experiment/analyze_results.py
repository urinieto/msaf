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

sys.path.append("..")
import msaf_io as MSAF
import eval as EV
import mir_eval
import jams2


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
    """Analyzes the answers of the experiment, contained in csv files.
    More specifically, it counts the number of words in the answers in order
    to know how often they appear. Then it sorts them based on their frequency
    and prints them.

    Parameters
    ----------
    results_dir: srt
        Path to the directory where the results are.
    """
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


def compute_mgp(jams_files, annotators, trim):
    """Computes the Mean Ground-truth Performance of the experiment
        results.

    For all the jams files of the experiment, it compares the results with
    the ground truth (also contained in the jam file), and computes the
    Mean Ground-truth Performance across them.

    It also reads the MGP of the machines, and plots them along with the
    humans results.

    Parameters
    ----------
    jams_files: list
        List containing all the file paths to the experiment files.
    annotators: dict
        Dictionary containing the names and e-mail addresses of the 5
        different annotators.
    trim: boolean
        Whether to trim the first and last boundaries.

    Returns
    -------
    mgp_results: np.array
        Array containing the results for the F3, P3, R3, F05, P05, R05 of the
        humans performance.
    """
    mgp_results = np.empty((0, 6))
    est_context = "large_scale"
    for i in xrange(1, len(annotators.keys())):
        FPR = np.empty((0, 6))
        for jam_file in jams_files:
            ds_name = os.path.basename(jam_file).split("_")[0]
            ann_context = MSAF.prefix_dict[ds_name]
            ann_times, ann_labels = jams2.converters.load_jams_range(jam_file,
                "sections", annotator_name=annotators.keys()[0],
                context=ann_context)
            try:
                est_times, est_labels = jams2.converters.load_jams_range(
                    jam_file, "sections", annotator_name=annotators.keys()[i],
                    context=est_context)
            except:
                logging.warning("Couldn't read annotator %d in JAMS %s" %
                                (i, jam_file))
                continue
            if len(ann_times) == 0:
                logging.warning("No GT annotations in file %s" % jam_file)
                continue
            if len(est_times) == 0:
                logging.warning("No annotation in file %s for annotator %s." %
                                (jam_file, annotators.keys()[i]))
                continue

            ann_times = np.asarray(ann_times)
            est_times = np.asarray(est_times)
            #print jam_file, ann_times, est_times, annotators.keys()[i]
            P3, R3, F3 = mir_eval.boundary.detection(
                ann_times, est_times, window=3, trim=trim)
            P05, R05, F05 = mir_eval.boundary.detection(
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
    dtype = [('F3', float), ('P3', float), ('R3', float), ('F05', float),
             ('P05', float), ('R05', float), ('track_id', '<U400')]
    mgp_results = compute_mgp(jams_files, annotators, trim)
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


def plot_ann_boundaries(jam_file, annotators, context="large_scale"):
    """Plots the different boundaries for a given track, contained in the
    jams file.

    Parameters
    ----------
    jam_file: str
        Path to the jam file that contains all the annotations.
    annotators: dict
        Dictionary containing the names and e-mail addresses of the 5
        different annotators.
    context: str
        The context of the level of annotation to plot.
    """
    all_boundaries = []
    annot_ids = []
    annot_name_ids_dict = {"GT":"GT", "Colin":"Ann1", "Eleni":"Ann2",
                           "Evan":"Ann3", "John":"Ann4", "Shuli":"Ann5"}
    for key in annotators.keys():
        if key == "GT":
            ds_name = os.path.basename(jam_file).split("_")[0]
            ann_context = MSAF.prefix_dict[ds_name]
            est_inters, est_labels = jams2.converters.load_jams_range(
                jam_file, "sections", annotator=0, context=ann_context)
        else:
            est_inters, est_labels = jams2.converters.load_jams_range(
                jam_file, "sections", annotator_name=key, context=context)
        est_times = EV.intervals_to_times(est_inters)
        all_boundaries.append(est_times)
        annot_ids.append(annot_name_ids_dict[key])

    N = len(all_boundaries)  # Number of lists of boundaries
    figsize = (5, 2.2)
    plt.figure(1, figsize=figsize, dpi=120, facecolor='w', edgecolor='k')
    for i, boundaries in enumerate(all_boundaries):
        print boundaries
        for b in boundaries:
            plt.axvline(b, i / float(N), (i + 1) / float(N))
        plt.axhline(i / float(N), color="k", linewidth=1)

    plt.title("Nelly Furtado - Promiscuous")
    #plt.title("Quartetto Italiano - String Quartet in F")
    plt.yticks(np.arange(0, 1, 1 / float(N)) + 1 / (float(N) * 2))
    plt.gcf().subplots_adjust(bottom=0.22)
    plt.gca().set_yticklabels(annot_ids)
    #plt.gca().invert_yaxis()
    plt.xlabel("Time (seconds)")
    plt.show()
    sys.exit()


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

    # Plot
    jam_file = "/Users/uri/datasets/SubSegments/annotations/Epiphyte_0220_promiscuous.jams"
    #jam_file = "/Users/uri/datasets/SubSegments/annotations/SALAMI_68.jams"
    plot_ann_boundaries(jam_file, annotators, "large_scale")

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
