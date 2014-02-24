#!/usr/bin/env python
# CREATED:2013-08-22 12:20:01 by Brian McFee <brm2132@columbia.edu>
'''Music segmentation using timbre, pitch, repetition and time.

If run as a program, usage is:

    ./segmenter.py AUDIO.mp3 OUTPUT.lab

'''


import sys
import os
import argparse

import numpy as np
import scipy.signal
import scipy.linalg

# Requires librosa-develop 0.3 branch
import librosa

SR          = 22050
N_FFT       = 2048
HOP_LENGTH  = 512
HOP_BEATS   = 64
N_MELS      = 128
FMAX        = 8000

REP_WIDTH   = 3
REP_FILTER  = 7

N_MFCC      = 32
N_CHROMA    = 12
N_REP       = 32

NOTE_MIN    = librosa.midi_to_hz(24) # 32Hz
NOTE_NUM    = 84
NOTE_RES    = 2                     # CQT filter resolution

# mfcc, chroma, repetitions for each, and 4 time features
__DIMENSION = N_MFCC + N_CHROMA + 2 * N_REP + 4

def features(filename):
    '''Feature-extraction for audio segmentation
    Arguments:
        filename -- str
        path to the input song

    Returns:
        - X -- ndarray
            
            beat-synchronous feature matrix:
            MFCC (mean-aggregated)
            Chroma (median-aggregated)
            Latent timbre repetition
            Latent chroma repetition
            Time index
            Beat index

        - beat_times -- array
            mapping of beat index => timestamp
            includes start and end markers (0, duration)

    '''
    
    

    def compress_data(X, k):
        e_vals, e_vecs = scipy.linalg.eig(X.dot(X.T))
        
        e_vals = np.maximum(0.0, np.real(e_vals))
        e_vecs = np.real(e_vecs)
        
        idx = np.argsort(e_vals)[::-1]
        
        e_vals = e_vals[idx]
        e_vecs = e_vecs[:, idx]
        
        # Truncate to k dimensions
        if k < len(e_vals):
            e_vals = e_vals[:k]
            e_vecs = e_vecs[:, :k]
        
        # Normalize by the leading singular value of X
        Z = np.sqrt(e_vals.max())
        
        if Z > 0:
            e_vecs = e_vecs / Z
        
        return e_vecs.T.dot(X)

    # Harmonic waveform
    def harmonify(y):
        D = librosa.stft(y)
        return librosa.istft(librosa.decompose.hpss(D)[0])

    # HPSS waveforms
    def hpss_wav(y):
        H, P = librosa.decompose.hpss(librosa.stft(y))

        return librosa.istft(H), librosa.istft(P)

    # Beats and tempo
    def get_beats(y):
        odf = librosa.onset.onset_strength(y=y, 
                                            sr=sr, 
                                            n_fft=N_FFT, 
                                            hop_length=HOP_BEATS, 
                                            n_mels=N_MELS, 
                                            fmax=FMAX, 
                                            aggregate=np.median)

        bpm, beats = librosa.beat.beat_track(onsets=odf, sr=sr, hop_length=HOP_BEATS)
        
        return bpm, beats

    # MFCC features
    def get_mfcc(y):
        # Generate a mel-spectrogram
        S = librosa.feature.melspectrogram(y, sr,   n_fft=N_FFT, 
                                                    hop_length=HOP_LENGTH, 
                                                    n_mels=N_MELS, 
                                                    fmax=FMAX).astype(np.float32)
    
        # Put on a log scale
        S = librosa.logamplitude(S, ref_power=S.max())

        return librosa.feature.mfcc(S=S, n_mfcc=N_MFCC)

    # Chroma features
    def chroma(y):
        # Build the wrapper
        CQT      = np.abs(librosa.cqt(y,    sr=SR, 
                                            resolution=NOTE_RES,
                                            hop_length=HOP_LENGTH,
                                            fmin=NOTE_MIN,
                                            n_bins=NOTE_NUM))

        C_to_Chr = librosa.filters.cq_to_chroma(CQT.shape[0], n_chroma=N_CHROMA) 

        return librosa.logamplitude(librosa.util.normalize(C_to_Chr.dot(CQT)))

    # Latent factor repetition features
    def repetition(X, metric='seuclidean'):
        R = librosa.segment.recurrence_matrix(X, 
                                            k=2 * int(np.ceil(np.sqrt(X.shape[1]))), 
                                            width=REP_WIDTH, 
                                            metric=metric,
                                            sym=False).astype(np.float32)

        P = scipy.signal.medfilt2d(librosa.segment.structure_feature(R), [1, REP_FILTER])
        
        # Discard empty rows.  
        # This should give an equivalent SVD, but resolves some numerical instabilities.
        P = P[P.any(axis=1)]

        return compress_data(P, N_REP)


    print '\t[1/6] loading audio'
    # Load the waveform
    y, sr = librosa.load(filename, sr=SR)

    # Compute duration
    duration = float(len(y)) / sr

    print '\t[2/6] Separating harmonic and percussive signals'
    # Separate signals
    y_harm, y_perc = hpss_wav(y)

    
    
    print '\t[3/6] detecting beats'
    # Get the beats
    bpm, beats = get_beats(y_perc)

    # augment the beat boundaries with the starting point
    beats = np.unique(np.concatenate([ [0], beats]))

    B = librosa.frames_to_time(beats, sr=SR, hop_length=HOP_BEATS)

    beat_frames = np.unique(librosa.time_to_frames(B, sr=SR, hop_length=HOP_LENGTH))

    # Stash beat times aligned to the longer hop lengths
    B = librosa.frames_to_time(beat_frames, sr=SR, hop_length=HOP_LENGTH)

    print '\t[4/6] generating MFCC'
    # Get the MFCCs
    M = get_mfcc(y)

    # Beat-synchronize the features
    M = librosa.feature.sync(M, beat_frames, aggregate=np.mean)
    
    print '\t[5/6] generating chroma'
    # Get the chroma from the harmonic component
    C = chroma(y_harm)

    # Beat-synchronize the features
    C = librosa.feature.sync(C, beat_frames, aggregate=np.median)
    
    # Time-stamp features
    N = np.arange(float(len(beat_frames)))
    
    # Beat-synchronous repetition features
    print '\t[6/6] generating structure features'
    R_timbre = repetition(librosa.segment.stack_memory(M))
    R_chroma = repetition(librosa.segment.stack_memory(C))
    
    # Stack it all up
    X = np.vstack([M, C, R_timbre, R_chroma, B, B / duration, N, N / len(beats)])

    # Add on the end-of-track timestamp
    B = np.concatenate([B, [duration]])

    return X, B

def gaussian_cost(X):
    '''Return the average log-likelihood of data under a standard normal
    '''
    
    d, n = X.shape
    
    if n < 2:
        return 0
    
    sigma = np.var(X, axis=1, ddof=1)
    
    cost =  -0.5 * d * n * np.log(2. * np.pi) - 0.5 * (n - 1.) * np.sum(sigma) 
    return cost
    
def clustering_cost(X, boundaries):
    
    # Boundaries include beginning and end frames, so k is one less
    k = len(boundaries) - 1
    
    d, n = map(float, X.shape)
    
    # Compute the average log-likelihood of each cluster
    cost = [gaussian_cost(X[:, start:end]) for (start, end) in zip(boundaries[:-1], 
                                                                    boundaries[1:])]
    
    cost = - 2 * np.sum(cost) / n + 2 * ( d * k )

    return cost

def get_k_segments(X, k):
    
    # Step 1: run ward
    boundaries = librosa.segment.agglomerative(X, k)
    
    boundaries = np.unique(np.concatenate(([0], boundaries, [X.shape[1]])))
    
    # Step 2: compute cost
    cost = clustering_cost(X, boundaries)
        
    return boundaries, cost

def get_segments(X, kmin=8, kmax=32):
    
    cost_min = np.inf
    S_best = []
    for k in range(kmax, kmin, -1):
        S, cost = get_k_segments(X, k)
        if cost < cost_min:
            cost_min = cost
            S_best = S
        else:
            break
            
    return S_best

def save_segments(outfile, S, beats):

    times = beats[S]
    with open(outfile, 'w') as f:
        for idx, (start, end) in enumerate(zip(times[:-1], times[1:]), 1):
            f.write('%.3f\t%.3f\tSeg#%03d\n' % (start, end, idx))
    
    pass

def process_arguments():
    parser = argparse.ArgumentParser(description='Music segmentation')

    parser.add_argument(    '-t',
                            '--transform',
                            dest    =   'transform',
                            required = False,
                            type    =   str,
                            help    =   'npy file containing the linear projection',
                            default =   None)

    parser.add_argument(    'input_song',
                            action  =   'store',
                            help    =   'path to input audio data')

    parser.add_argument(    'output_file',
                            action  =   'store',
                            help    =   'path to output segment file')

    return vars(parser.parse_args(sys.argv[1:]))


def load_transform(transform_file):

    if transform_file is None:
        W = np.eye(__DIMENSION)
    else:
        W = np.load(transform_file)

    return W

def get_num_segs(duration, MIN_SEG=10.0, MAX_SEG=45.0):
    kmin = max(1, np.floor(duration / MAX_SEG).astype(int))
    kmax = max(2, np.ceil(duration / MIN_SEG).astype(int))

    return kmin, kmax

if __name__ == '__main__':

    parameters = process_arguments()

    # Load the features
    print '- ', os.path.basename(parameters['input_song'])

    X, beats    = features(parameters['input_song'])

    # Load the transformation
    W           = load_transform(parameters['transform'])
    print '\tapplying transformation...'
    X           = W.dot(X)

    # Find the segment boundaries
    print '\tpredicting segments...'
    kmin, kmax  = get_num_segs(beats[-1])
    S           = get_segments(X, kmin=kmin, kmax=kmax)

    # Output lab file
    print '\tsaving output to ', parameters['output_file']
    save_segments(parameters['output_file'], S, beats)

    pass
