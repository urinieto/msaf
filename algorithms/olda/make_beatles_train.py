#!/usr/bin/env python

import numpy as np
import glob
import os
import sys

from joblib import Parallel, delayed
import cPickle as pickle

# Needs the most recent mir_eval and librosa development branch
import mir_eval
import librosa

from segmenter import features

def get_all_files(basedir, ext='.wav'):
    for root, dirs, files in os.walk(basedir, followlinks=True):
        files = glob.glob(os.path.join(root, '*'+ext))
        for f in files:
            yield os.path.abspath(f)
    

def align_segmentation(filename, beat_times):
    '''Load a ground-truth segmentation, and align times to the nearest detected beats
    
    Arguments:
        filename -- str
        beat_times -- array

    Returns:
        segment_beats -- array
            beat-aligned segment boundaries

        segment_times -- array
            true segment times

        segment_labels -- array
            list of segment labels

    '''
    
    # These labels have both begin and end times
    segment_times, segment_labels = mir_eval.io.load_annotation(filename)

    # Compute beat intervals
    beat_intervals = np.asarray(zip(beat_times[:-1], beat_times[1:]))

    # Map beats to segments
    beat_segment_ids  = librosa.util.match_intervals(beat_intervals, segment_times)

    # Now, find the first beat mapping to each segment
    segment_beats = []
    segment_times_out = []
    segment_labels_out = []
    for i in range(segment_times.shape[0]):
        hits = np.argwhere(beat_segment_ids == i)
        if len(hits) > 0:
            segment_beats.extend( hits[0] )
            segment_times_out.append(segment_times[i,:])
            segment_labels_out.append(segment_labels[i])

    # Pull out the segment-start times
    segment_beats = list(segment_beats)
    segment_times_out = np.asarray(segment_times_out)[:, 0].squeeze()

    if segment_times_out.ndim == 0:
        segment_times_out = segment_times_out[np.newaxis]

    return segment_beats, segment_times_out, segment_labels_out

# <codecell>

def import_data(audio, label, rootpath, output_path):
        data_file = '%s/features/BEATLES/%s.pickle' % (output_path, os.path.splitext(os.path.basename(audio))[0])

        if os.path.exists(data_file):
            with open(data_file, 'r') as f:
                Data = pickle.load(f)
                print audio, 'cached!'
        else:
            try:
                X, B     = features(audio)
                Y, T, L  = align_segmentation(label, B)
                
                Data = {'features':         X, 
                        'beats':            B, 
                        'filename':         audio, 
                        'segment_times':    T,
                        'segment_labels':   L,
                        'segments':         Y}

                print audio, 'processed!'
        
                with open(data_file, 'w') as f:
                    pickle.dump( Data, f )
            except Exception as e:
                print audio, 'failed!'
                print e
                Data = None

        return Data

# <codecell>

def make_dataset(n=None, n_jobs=16, rootpath='beatles/', output_path='data/'):
    
    F_audio     = sorted([_ for _ in get_all_files(os.path.join(rootpath, 'audio'), '.wav')])
    F_labels    = sorted([_ for _ in get_all_files(os.path.join(rootpath, 'seglab'), '.lab')])

    assert(len(F_audio) == len(F_labels))
    if n is None:
        n = len(F_audio)

    data = Parallel(n_jobs=n_jobs)(delayed(import_data)(audio, label, rootpath, output_path) for (audio, label) in zip(F_audio[:n], F_labels[:n]))
    
    X, Y, B, T, F, L = [], [], [], [], [], []
    for d in data:
        if d is None:
            continue
        X.append(d['features'])
        Y.append(d['segments'])
        B.append(d['beats'])
        T.append(d['segment_times'])
        F.append(d['filename'])
        L.append(d['segment_labels'])
    
    return X, Y, B, T, F, L


if __name__ == '__main__':
    beatles_path = sys.argv[1]
    output_path = sys.argv[2]
    X, Y, B, T, F, L = make_dataset(rootpath=beatles_path, output_path=output_path)
    with open('%s/beatles_data.pickle' % output_path, 'w') as f:
        pickle.dump( (X, Y, B, T, F, L), f)
