#!/usr/bin/env python
# coding: utf-8
"""
This script identifies the boundaries of a given track using a novel C-NMF
method (v2).
"""

__author__ = "Oriol Nieto"
__copyright__ = "Copyright 2014, Music and Audio Research Lab (MARL)"
__license__ = "GPL"
__version__ = "1.0"
__email__ = "oriol@nyu.edu"

import logging
import numpy as np
import time
import sys

import run_segmenter as RS
sys.path.append("../../")
import eval2 as EV


def process():
    ranks = np.arange(3, 6)
    hh = np.arange(8, 18)
    RR = np.arange(8, 18)
    in_path = "/Users/uri/datasets/Segments/"
    res_file = "results-salami.txt"
    ds_name = "SALAMI"
    feature = "hpcp"
    for rank in ranks:
        for h in hh:
            for R in RR:
                logging.info("Computing rank: %d, h: %d, R: %d" %
                            (rank, h, R))
                RS.process(in_path, feature=feature,
                           ds_name=ds_name, rank=rank, h=h, R=R)
                res = EV.process(in_path, "cnmf2", ds_name=ds_name)
                res = res.mean()
                str = "%d\t%d\t%d\t%.4f\t%4f\t%4f\t%.4f\t%4f\t%4f\n" % \
                    (rank, h, R, res["F3"], res["P3"], res["R3"],
                     res["F0.5"], res["P0.5"], res["R0.5"])
                with open(res_file, "a") as f:
                    f.write(str)


def main():
    """Main function to sweep parameters."""
    start_time = time.time()

    # Setup the logger
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s',
        level=logging.INFO)

    # Run the algorithm
    process()

    # Done!
    logging.info("Done! Took %.2f seconds." % (time.time() - start_time))


if __name__ == '__main__':
    main()
