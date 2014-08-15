"""This script contains methods to plot multiple aspects of the results
of MSAF.
"""

__author__      = "Oriol Nieto"
__copyright__   = "Copyright 2014, Music and Audio Research Lab (MARL)"
__license__     = "BSD"
__version__     = "1.0"
__email__       = "oriol@nyu.edu"

import numpy as np
import os
import pylab as plt

# Local stuff
import msaf_io as MSAF


def plot_boundaries(all_boundaries, est_file, algo_ids=None, title=None):
    """Plots all the boundaries.

    Parameters
    ----------
    all_boundaries: list
        A list of np.arrays containing the times of the boundaries, one array
        for each algorithm.
    est_file: str
        Path to the estimated file (JSON file)
    algo_ids : list
        List of algorithm ids to to read boundaries from.
        If None, all algorithm ids are read.
    title : str
        Title of the plot. If None, the name of the file is printed instead.
    """
    translate_ids = {
        "olda"  : "OLDA",
        "cnmf3" : "C-NMF",
        "foote" : "Foote",
        "levy"  : "CC",
        "serra" : "SF",
        "siplca": "SI-PLCA"
    }

    N = len(all_boundaries)  # Number of lists of boundaries
    if algo_ids is None:
        algo_ids = MSAF.get_algo_ids(est_file)

    # Translate ids
    for i, algo_id in enumerate(algo_ids):
        algo_ids[i] = translate_ids[algo_id]
    algo_ids = ["GT"] + algo_ids

    figsize = (6, 4)
    plt.figure(1, figsize=figsize, dpi=120, facecolor='w', edgecolor='k')
    for i, boundaries in enumerate(all_boundaries):
        color = "b"
        if i == 0:
            color = "g"
        for b in boundaries:
            plt.axvline(b, i / float(N), (i + 1) / float(N), color=color)
        plt.axhline(i / float(N), color="k", linewidth=1)

    if title is None:
        title = os.path.basename(est_file).split(".")[0]
    plt.title(title)
    #plt.title("Nelly Furtado - Promiscuous")
    #plt.title("Quartetto Italiano - String Quartet in F")
    plt.yticks(np.arange(0, 1, 1 / float(N)) + 1 / (float(N) * 2))
    plt.gcf().subplots_adjust(bottom=0.22)
    plt.gca().set_yticklabels(algo_ids)
    #plt.gca().invert_yaxis()
    plt.xlabel("Time (seconds)")
    plt.show()
