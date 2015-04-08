#!/usr/bin/env python
# CREATED:2013-08-22 12:20:01 by Brian McFee <brm2132@columbia.edu>
'''Music segmentation using timbre, pitch, repetition and time.

If run as a program, usage is:

    ./segmenter.py AUDIO.mp3 OUTPUT.lab

'''


import sys
import os
import argparse

import cPickle as pickle
import numpy as np

import segmenter

def features(input_song):

    with open(input_song, 'r') as f:
        data = pickle.load(f)

    return data['features'], data['segment_times'], data['beats']

def get_num_segs(duration, MIN_SEG=10.0, MAX_SEG=45.0):

    kmin = max(1, np.floor(duration / MAX_SEG).astype(int))
    kmax = max(1, np.ceil(duration / MIN_SEG).astype(int))

    return kmin, kmax

def process_arguments():
    parser = argparse.ArgumentParser(description='Music segmentation with pre-computed features')

    parser.add_argument(    '-t',
                            '--transform',
                            dest    =   'transform',
                            required = False,
                            type    =   str,
                            help    =   'npy file containing the linear projection',
                            default =   None)

    parser.add_argument(    '-d',
                            '--dynamic',
                            dest    =   'dynamic',
                            required    =   False,
                            action      =   'store_true',
                            help        =   'dynamic segment numberings')

    parser.add_argument(    '-g',
                            '--gnostic',
                            dest    =   'gnostic',
                            action  =   'store_true',
                            required=   False,
                            help    =   'Operate with knowledge of k')

    parser.add_argument(    'input_song',
                            action  =   'store',
                            help    =   'path to input feature data (pickle file)')

    parser.add_argument(    'output_file',
                            action  =   'store',
                            help    =   'path to output segment file')

    return vars(parser.parse_args(sys.argv[1:]))

if __name__ == '__main__':

    parameters = process_arguments()

    # Load the features
    print '- ', os.path.basename(parameters['input_song'])

    X, Y, beats    = features(parameters['input_song'])
    # Load the transformation
    W           = segmenter.load_transform(parameters['transform'])
    print '\tapplying transformation...'
    X           = W.dot(X)

    # Find the segment boundaries
    print '\tpredicting segments...'
    if parameters['gnostic']:
        S           = segmenter.get_segments(X, kmin=len(Y)-1, kmax=len(Y))
    elif parameters['dynamic']:
        kmin, kmax  = get_num_segs(beats[-1])
        S           = segmenter.get_segments(X, kmin=kmin, kmax=kmax)
    else:
        S           = segmenter.get_segments(X)

    # Output lab file
    print '\tsaving output to ', parameters['output_file']
    segmenter.save_segments(parameters['output_file'], S, beats)

    pass
