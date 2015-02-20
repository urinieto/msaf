#!/usr/bin/env python
# coding: utf-8
"""
This script identifies the boundaries of a given track using a novel C-NMF
method (v3).
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

import run_boundaries as RB
sys.path.append("../../")
import eval as EV


def process():
    ranks = np.arange(2, 6)
    hh = np.arange(2, 18)
    RR = np.arange(2, 18)
    in_path = "/Users/uri/datasets/Segments/"
    res_file = "results-salami.txt"
    ds_name = "SALAMI"
    feature = "hpcp"
    for rank in ranks:
        for h in hh:
            for R in RR:
                logging.info("Computing rank: %d, h: %d, R: %d" %
                            (rank, h, R))
                RB.process(in_path, feature=feature,
                           ds_name=ds_name, rank=rank, h=h, R=R)
                res = EV.process(in_path, "cnmf3", ds_name=ds_name)
                res = res.mean(axis=0)
                with open(res_file, "a") as f:
                    str = "%d\t%d\t%d\t%.4f\t%4f\t%4f\t%.4f\t%4f\t%4f\n" % \
                        (rank, h, R, res[2], res[0], res[1], res[5], 
                         res[3], res[4])
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
