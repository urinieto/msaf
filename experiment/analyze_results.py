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
import itertools
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
import matplotlib.patches as mpatches

sys.path.append("..")
import msaf_io as MSAF
import eval as EV
import jams2

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


def analyze_tags(results_dir="results/"):
    """Analyzes the tags (answers) of the experiment, contained in csv files.
    More specifically, it counts the number of tags in the answers in order
    to know how often they appear. Then it sorts them based on their frequency
    and prints them.

    Parameters
    ----------
    results_dir: srt
        Path to the directory where the results are.
    """
    # Get all the folders with the annotators results
    annotator_dirs = glob.glob(os.path.join(results_dir, "*"))
    tags = []
    for annot_dir in annotator_dirs:
        csv_file = os.path.join(annot_dir, "tags.csv")
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
                    tags.append(word.strip(" ").lower())

    # Get histogram from the tags
    histo = Counter(tags)

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
        Array containing the results as in the eval module of MSAF for the
        humans performance.
    """
    mgp_results = np.empty((0, 9))
    est_context = "large_scale"
    bins = 250
    for i in xrange(1, len(annotators.keys())):
        FPR = np.empty((0, 9))
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
            results = EV.compute_results(ann_times, est_times, trim, bins,
                                         jam_file)
            FPR = np.vstack((FPR, tuple(results)))

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
                         annotators.keys()[i], FPR[2], FPR[0], FPR[1], FPR[5],
                         FPR[3], FPR[4]))
    return mgp_results


def compute_mma_results(jam_file, annotators, trim, bins=250, gt=False):
    """Compute the Mean Measure Agreement for all the algorithms of the given
    file jam_file.

    Parameters
    ----------
    jam_file: str
        Jam file containing all the annotations for a given track.
    annotators: dict
        Dictionary containing the names and e-mail addresses of the 5
        different annotators.
    trim: boolean
        Whether to trim the first and last boundary or not.
    bins: int
        The number of bins to compute for the Information Gain.

    Returns
    -------
    results_mma: np.array
        All the results for all the different comparisons between algorithms.
        In order to obtain the average, simply take the mean across axis=0.
    """
    context = "large_scale"
    results_mma = []
    if gt:
        keys = annotators.keys()
    else:
        keys = annotators.keys()[1:]
    for names in itertools.combinations(keys, 2):
        # Read estimated times from both algorithms
        if names[0] == "GT":
            ds_name = os.path.basename(jam_file).split("_")[0]
            ann_context = MSAF.prefix_dict[ds_name]
            est_inters1, est_labels1 = jams2.converters.load_jams_range(
                jam_file, "sections", annotator=0, context=ann_context)
        else:
            est_inters1, est_labels1 = jams2.converters.load_jams_range(
                jam_file, "sections", annotator_name=names[0], context=context)
        est_inters2, est_labels2 = jams2.converters.load_jams_range(jam_file,
                            "sections", annotator_name=names[1],
                            context=context)
        if est_inters1 == [] or est_inters2 == []:
            continue

        # Compute results
        #print est_inters1, est_inters2, jam_file, names
        #print names
        results = EV.compute_results(est_inters1, est_inters2, trim, bins,
                                     jam_file)
        results_mma.append(results)

    return np.asarray(results_mma)


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
    jams_files = glob.glob(os.path.join(annot_dir, "*.jams"))
    dtype = [('P3', float), ('R3', float), ('F3', float), ('P05', float),
             ('R05', float), ('F05', float), ('D', float), ('DevA2E', float),
             ('DevE2A', float), ('track_id', '<U400')]
    track_ids = get_track_ids(annot_dir)
    mma_humans_file = "mma_experiment_humans.pk"
    mgp_humans_file = "mgp_experiment_humans.pk"

    # Compute the MMA human results
    logging.info("Computing the MMA...")
    #mma_results = []
    #for jam_file in jams_files:
        #mma_file = compute_mma_results(jam_file, annotators, trim, 250)
        #mma_results.append(np.mean(mma_file, axis=0))
    #for i in xrange(len(track_ids)):
        #mma_results[i] = mma_results[i].tolist()
        #mma_results[i].append(track_ids[i])
        #mma_results[i] = tuple(mma_results[i])
    #mma_results = np.asarray(mma_results, dtype=dtype)
    #pickle.dump(mma_results, open(mma_humans_file, "w"))
    mma_results = pickle.load(open(mma_humans_file, "r"))

    # Compute the MGP human results (not really necessary)
    #logging.info("Computing the MGP...")
    #mgp_results = compute_mgp(jams_files, annotators, trim)
    #mgp_results = mgp_results.tolist()
    #for i in xrange(len(track_ids)):
        #mgp_results[i].append(track_ids[i])
        #mgp_results[i] = tuple(mgp_results[i])
    #mgp_results = np.asarray(mgp_results, dtype=dtype)
    #pickle.dump(mgp_results, open(mgp_humans_file, "w"))
    #mgp_results = pickle.load(open(mgp_humans_file, "r"))

    # Get MGP machine results
    mgp_results_machine = pickle.load(open(
        "../notes/mgp_experiment_machine.pk", "r"))

    mgp_results_machine = np.sort(mgp_results_machine, order="track_id")
    mma_results = np.sort(mma_results, order='track_id')

    for mgp_res, mma_res in zip(mgp_results_machine, mma_results):
        print mgp_res["F3"], mma_res["F3"], mgp_res["track_id"], \
            mma_res["track_id"]

    # Plot results
    figsize = (4, 4)
    plt.figure(1, figsize=figsize, dpi=120, facecolor='w', edgecolor='k')
    plt.scatter(mgp_results_machine["F3"], mma_results["F3"])
    plt.plot([0, 1], [0, 1])
    plt.gca().set_ylabel("Human MMA$_{F3}$ results")
    plt.gca().set_xlabel("Machine MGP$_{F3}$ results")
    plt.gca().set_xticks(np.arange(0, 1.1, .1))
    plt.gca().set_yticks(np.arange(0, 1.1, .1))
    plt.gca().set_xlim(0, 1)
    plt.gca().set_ylim(0, 1)
    plt.gcf().subplots_adjust(bottom=0.12, left=0.15)

    alpha = 0.3
    # Best-humans, worse-machines
    circle = mpatches.Circle([0.3, 0.79], 0.1, ec="none", color="g",
                             alpha=alpha)
    plt.gca().add_artist(circle)

    # Best-humans, worse-machines
    ellipse = mpatches.Ellipse([0.84, 0.84], 0.1, 0.26, ec="none", color="r",
                               alpha=alpha)
    plt.gca().add_artist(ellipse)

    # Worse-humans, worse-machines
    rectangle = mpatches.Rectangle([0.2, 0.3], 0.2, 0.2, ec="none", color="m",
                             alpha=alpha)
    plt.gca().add_artist(rectangle)

    plt.show()


def plot_ann_boundaries(jam_file, annotators, context="large_scale",
                        ax=None):
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
    annot_name_ids_dict = {"GT": "GT", "Colin": "Ann1", "Eleni": "Ann2",
                           "Evan": "Ann3", "John": "Ann4", "Shuli": "Ann5"}
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
    if ax is None:
        figsize = (5, 2.2)
        plt.figure(1, figsize=figsize, dpi=120, facecolor='w', edgecolor='k')
        my_plt = plt
    else:
        my_plt = ax
    for i, boundaries in enumerate(all_boundaries):
        print boundaries
        for b in boundaries:
            my_plt.axvline(b, i / float(N), (i + 1) / float(N))
        my_plt.axhline(i / float(N), color="k", linewidth=1)

    #my_plt.title("Nelly Furtado - Promiscuous")
    #my_plt.title("Quartetto Italiano - String Quartet in F")
    #my_plt.title("Prince & The Revolution - Purple Rain")
    #my_plt.title("Yes - Starship Trooper")
    if ax is None:
        my_plt = plt.gca()
    my_plt.set_yticks(np.arange(0, 1, 1 / float(N)) + 1 / (float(N) * 2))
    #my_plt.gcf().subplots_adjust(bottom=0.22)
    my_plt.set_yticklabels(annot_ids)
    #my_plt.gca().invert_yaxis()
    my_plt.set_xlabel("Time (seconds)")
    #my_plt.show()


def process(annot_dir, trim=False):
    """Main process to parse all the results from the results_dir
        to out_dir."""

    # Plot
    ann_dir = "/Users/uri/datasets/SubSegments/annotations/"
    #jam_file = "Epiphyte_0220_promiscuous.jams"
    #jam_file = "SALAMI_68.jams"
    #jam_file = "Isophonics_01 The Show Must Go On.jams"
    #jam_file = "Cerulean_Prince_&_The_Revolution-Purple_Rain.jams"
    #jam_file = "Cerulean_Yes-Starship_Trooper:"\
                #"_A._Life_Seeker,_B._Disillu.jams"
    #jam_file = "SALAMI_546.jams"
    #jam_file = "Epiphyte_0723_hotelroomservice.jams"
    #jam_file = "SALAMI_114.jams"
    jam_file = "Epiphyte_0780_letmebereal.jams"
    plot_ann_boundaries(ann_dir + jam_file, annotators, "large_scale")

    # Analyze the tags
    analyze_tags()

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
