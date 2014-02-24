#!/usr/bin/env python

import sys
import argparse
import numpy as np

import mir_eval
import cPickle as pickle

from joblib import Parallel, delayed

import RFDA
import segmenter

def process_arguments():
    parser = argparse.ArgumentParser(description='RFDA fit for music segmentation')

    parser.add_argument(    'input_file',
                            action  =   'store',
                            help    =   'path to training data (from make_*_train.py)')

    parser.add_argument(    'output_file',
                            action  =   'store',
                            help    =   'path to save model file')

    parser.add_argument(    '-j',
                            '--num-jobs',
                            dest    =   'num_jobs',
                            action  =   'store',
                            type    =   int,
                            required=   False,
                            default =   '4',
                            help    =   'Number of parallel jobs')

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
        xt = model.dot(x)
    else:
        xt = x

    # Then, run the segmenter
    kmin, kmax = segmenter.get_num_segs(b[-1])
    boundary_beats = segmenter.get_segments(xt, kmin=kmin, kmax=kmax)

    if len(boundary_beats) < 2 or len(t) < 2:
        return 0.0

    boundary_times = mir_eval.util.adjust_events(b[boundary_beats], t_min=0.0, t_max=t[-1])[0]

    truth_intervals = mir_eval.util.boundaries_to_intervals(t)[0]
    pred_intervals = mir_eval.util.boundaries_to_intervals(boundary_times)[0]
    score = mir_eval.segment.boundary_detection(truth_intervals, pred_intervals)[-1]

    return score

def fit_model(X, Y, B, T, n_jobs):

    SIGMA = 10**np.arange(0, 10)

    best_score  = -np.inf
    best_sigma  = None
    model       = None

    for sig in SIGMA:
        O = RFDA.RFDA(sigma=sig)
        O.fit(X, Y)

        scores = Parallel(n_jobs=n_jobs)( delayed(score_model)(O.components_, *z) for z in zip(X, B, T))

        mean_score = np.mean(scores)
        print 'Sigma=%.2e, score=%.3f' % (sig, mean_score)

        if mean_score > best_score:
            best_score  = mean_score
            best_sigma  = sig
            model       = O.components_

    print 'Best sigma: %.2e' % best_sigma
    return model

if __name__ == '__main__':
    parameters = process_arguments()

    X, Y, B, T = load_data(parameters['input_file'])[:4]

    model = fit_model(X, Y, B, T, parameters['num_jobs'])

    np.save(parameters['output_file'], model)
