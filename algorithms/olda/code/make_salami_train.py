#!/usr/bin/env python

import numpy as np
import glob
import os
import sys

from joblib import Parallel, delayed
import cPickle as pickle

import mir_eval
import librosa

from segmenter import features

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
    
    segment_times, segment_labels = mir_eval.io.load_events(filename)

    # Map to intervals, clip the last label marker
    segment_intervals = np.asarray(zip(segment_times[:-1], segment_times[1:]))
    segment_labels    = segment_labels[:-1]

    # Map beats to intervals
    beat_intervals    = np.asarray(zip(beat_times[:-1], beat_times[1:]))

    # Map beats to segments
    beat_segment_ids  = librosa.util.match_intervals(beat_intervals, segment_intervals)

    segment_beats = []
    segment_times_out = []
    segment_labels_out = []

    for i in range(segment_times.shape[0]):
        hits = np.argwhere(beat_segment_ids == i)
        if len(hits) > 0:
            segment_beats.extend(hits[0])
            segment_times_out.append(segment_intervals[i,:])
            segment_labels_out.append(segment_labels[i])

    # Pull out the segment start times
    segment_beats = list(segment_beats)
    segment_times_out = np.asarray(segment_times_out)[:, 0].squeeze().reshape((-1, 1))

    if segment_times_out.ndim == 0:
        segment_times_out = segment_times_out[np.newaxis]

    return segment_beats, segment_times_out, segment_labels_out

# <codecell>

def get_annotation(song, rootpath):
    song_num = os.path.splitext(os.path.split(song)[-1])[0]
    return '%s/data/%s/parsed/textfile1_functions.txt' % (rootpath, song_num)

# <codecell>

def import_data(song, rootpath, output_path):
        data_file = '%s/features/SALAMI/%s.pickle' % (output_path, os.path.splitext(os.path.basename(song))[0])

        if os.path.exists(data_file):
            with open(data_file, 'r') as f:
                Data = pickle.load(f)
                print song, 'cached!'
        else:
            try:
                X, B     = features(song)
                Y, T, L  = align_segmentation(get_annotation(song, rootpath), B)
                
                Data = {'features': X, 
                        'beats': B, 
                        'filename': song, 
                        'segment_times': T,
                        'segment_labels': L,
                        'segments': Y}
                print song, 'processed!'
        
                with open(data_file, 'w') as f:
                    pickle.dump( Data, f )
            except Exception as e:
                print song, 'failed!'
                print e
                Data = None

        return Data

# <codecell>

def make_dataset(n=None, n_jobs=16, rootpath='SALAMI/', output_path='data/'):
    
#    EXTS = ['mp3', 'wav', 'ogg', 'flac', 'm4a', 'aac']
    EXTS = ['wav']
    files = []
    for e in EXTS:
        files.extend(filter(lambda x: os.path.exists(get_annotation(x, rootpath)), glob.iglob('%s/audio/*.%s' % (rootpath, e))))
    files = sorted(files)
    if n is None:
        n = len(files)

    data = Parallel(n_jobs=n_jobs)(delayed(import_data)(song, rootpath, output_path) for song in files[:n])
    
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
    salami_path = sys.argv[1]
    output_path = sys.argv[2]
    X, Y, B, T, F, L = make_dataset(rootpath=salami_path, output_path=output_path)
    with open('%s/salami_data.pickle' % output_path, 'w') as f:
        pickle.dump( (X, Y, B, T, F, L), f)
