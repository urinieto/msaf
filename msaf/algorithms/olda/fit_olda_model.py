#!/usr/bin/env python

import os
import sys
import argparse
import numpy as np
import glob

import mir_eval
import jams
import cPickle as pickle

from joblib import Parallel, delayed

import OLDA
import segmenter

import msaf


def process_arguments():
    parser = argparse.ArgumentParser(
        description='OLDA fit for music segmentation')

    parser.add_argument("ds_path",
                        action="store",
                        help="Input dataset")

    parser.add_argument('input_file',
                        action='store',
                        help='path to training data (from make_*_train.py)')

    parser.add_argument('output_file',
                        action='store',
                        help='path to save model file')

    parser.add_argument('-j',
                        '--num-jobs',
                        dest='num_jobs',
                        action='store',
                        type=int,
                        required=False,
                        default='4',
                        help='Number of parallel jobs')
    parser.add_argument("-b",
                        action="store_true",
                        dest="annot_beats",
                        help="Use annotated beats",
                        default=False)
    return vars(parser.parse_args(sys.argv[1:]))


def load_data(input_file):

    with open(input_file, 'r') as f:
        #   X = features
        #   Y = segment boundaries (as beat numbers)
        #   B = beat timings
        #   T = true segment boundaries (seconds)
        #   F = filename

        X, Y, B, T = pickle.load(f)[:4]

    return X, Y, B, T


def score_model(model, x, b, t):

    # First, transform the data
    if model is not None:
        try:
            xt = model.dot(x)
        except:
            return 0.0
    else:
        xt = x

    # Then, run the segmenter
    kmin, kmax = segmenter.get_num_segs(b[-1])
    boundary_beats = segmenter.get_segments(xt, kmin=kmin, kmax=kmax)

    if len(boundary_beats) < 2 or len(t) < 2:
        return 0.0

    t = np.unique(t)
    boundary_times = mir_eval.util.adjust_events(b[boundary_beats], t_min=0.0,
                                                 t_max=t[-1])[0]

    truth_intervals = mir_eval.util.boundaries_to_intervals(t)
    pred_intervals = mir_eval.util.boundaries_to_intervals(boundary_times)
    score = mir_eval.segment.detection(truth_intervals, pred_intervals,
                                       trim=True)[-1]

    return score


def fit_model(X, Y, B, T, n_jobs, annot_beats, ds_path):

    SIGMA = 10 ** np.arange(-2, 18)

    best_score = -np.inf
    best_sigma = None
    model = None

    print len(X)

    for sig in SIGMA:
        O = OLDA.OLDA(sigma=sig)
        O.fit(X, Y)

        scores = []
        files = msaf.io.get_dataset_files(ds_path)
        for f, z in zip(files, zip(X, B, T)):
            f = f.ref_file
            beats = z[1]
            if annot_beats:
                jam = jams.load(f)
                ann = jam.search(namespace="beat")[0]
                beats = ann.data.to_interval_values()[0][:, 0]
            print "\t\tProcessing ", f
            scores.append(score_model(O.components_, z[0], beats, z[2]))

        mean_score = np.mean(scores)
        print 'Sigma=%.2e, score=%.3f' % (sig, mean_score)

        if mean_score > best_score:
            best_score = mean_score
            best_sigma = sig
            model = O.components_

    print 'Best sigma: %.2e' % best_sigma
    return model

if __name__ == '__main__':
    parameters = process_arguments()

    print "Loading data from %s ..." % parameters["input_file"]
    X, Y, B, T = load_data(parameters['input_file'])[:4]

    print "Fitting model..."
    model = fit_model(X, Y, B, T, parameters['num_jobs'],
                      parameters['annot_beats'], parameters['ds_path'])

    np.save(parameters['output_file'], model)
