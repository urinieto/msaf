#!/usr/bin/env python
# CREATED:2013-08-22 12:20:01 by Brian McFee <brm2132@columbia.edu>
'''Audio beat tracking using median-aggregation.

If run as a program, usage is:

    ./beat_tracker.py AUDIO.mp3 OUTPUT.csv

'''


import sys
import os
import argparse

import numpy as np

# Requires librosa-develop branch
import librosa

SR          = 22050
N_FFT       = 2048
HOP         = 64
N_MELS      = 128
FMAX        = 8000

# mfcc, chroma, repetitions for each, and 4 time features

def get_beats(filename):
    '''LibROSA beat tracking

    Arguments:
        filename -- str
        path to the input song

    Returns:
        - beat_times -- array
            mapping of beat index => timestamp
            includes start and end markers (0, duration)

    '''
    
    

    # Onset strength function for beat tracking
    def onset(S):
        odf = np.median(np.maximum(0.0, np.diff(S, axis=1)), axis=0)
        odf = odf - odf.min()
        odf = odf / odf.max()
        return odf
    
    print '\t[1/2] loading audio'
    # Load the waveform
    y, sr = librosa.load(filename, sr=SR)

    
    # Generate a mel-spectrogram
    S = librosa.feature.melspectrogram(y, sr,   n_fft=N_FFT, 
                                                hop_length=HOP, 
                                                n_mels=N_MELS, 
                                                fmax=FMAX).astype(np.float32)

    # Normalize by peak energy
    S = S / S.max()

    # Put on a log scale
    S = librosa.logamplitude(S)
    
    print '\t[2/2] detecting beats'
    # Get the beats
    bpm, beats = librosa.beat.beat_track(onsets=onset(S), 
                                            sr=SR, 
                                            hop_length=HOP, 
                                            n_fft=N_FFT)

    beat_times = librosa.frames_to_time(beats, sr=SR, hop_length=HOP)
    return beats, beat_times

def process_arguments():
    parser = argparse.ArgumentParser(description='Beat tracking')

    parser.add_argument(    'input_song',
                            action  =   'store',
                            help    =   'path to input audio data')

    parser.add_argument(    'output_file',
                            action  =   'store',
                            help    =   'path to output beat file')

    return vars(parser.parse_args(sys.argv[1:]))


if __name__ == '__main__':

    parameters = process_arguments()

    # Load the features
    print '- ', os.path.basename(parameters['input_song'])

    beats, beat_times   = get_beats(parameters['input_song'])

    # Output lab file
    print '\tsaving output to ', parameters['output_file']
    librosa.output.frames_csv(parameters['output_file'], beats, sr=SR, hop_length=HOP)

    pass
