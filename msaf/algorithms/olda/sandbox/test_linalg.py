#!/usr/bin/env python
""" Testing linalg.eig function.
I have problems running the scipy version in OS X, while the version in numpy
works fine.
"""

import argparse
import numpy as np
import scipy.linalg

from joblib import Parallel, delayed


def test_linalg(X, fun):
    vals, vect = eval(fun).eig(X)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=
        "Testin linalg.eig.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-f",
                        action="store",
                        dest="file_name",
                        type=str,
                        help="Path to the npy file containing the matrix to"
                        "decompose",
                        default="breaking_eig.npy")
    parser.add_argument("-j",
                        action="store",
                        dest="n_jobs",
                        type=int,
                        help="Number of jobs (threads)",
                        default=4)
    args = parser.parse_args()

    X = np.load(args.file_name)

    # This should work:
    data = Parallel(n_jobs=args.n_jobs)(delayed(test_linalg)(X, "np.linalg")
            for i in xrange(10))

    # This breaks in main code
    data = Parallel(n_jobs=args.n_jobs)(delayed(test_linalg)(X, "scipy.linalg")
            for i in xrange(10))
