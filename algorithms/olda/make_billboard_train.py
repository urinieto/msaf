#!/usr/bin/env python

import numpy as np
import glob
import os
import sys

from joblib import Parallel, delayed
import cPickle as pickle

import mir_eval

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

    segment_beats = []
    for t in segment_times:
        # Find the closest beat
        segment_beats.append( np.argmin((beat_times - t)**2))
        
    return segment_beats, segment_times, segment_labels

# <codecell>

def get_annotation(song, rootpath):
    song_num = os.path.splitext(os.path.split(song)[-1])[0]
    return '%s/structure_new/%s.lab' % (rootpath, song_num)

# <codecell>

def import_data(song, rootpath, output_path):
        data_file = '%s/features/BILLBOARD/%s.pickle' % (output_path, os.path.splitext(os.path.basename(song))[0])

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

def make_dataset(n=None, n_jobs=16, rootpath='BILLBOARD/', output_path='data/'):
    
    EXTS = ['m4a']
    files = []
    for e in EXTS:
        files.extend(filter(lambda x: os.path.exists(get_annotation(x, rootpath)), glob.iglob('%s/m4a/*.%s' % (rootpath, e))))
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
    billboard_path = sys.argv[1]
    output_path = sys.argv[2]
    X, Y, B, T, F, L = make_dataset(rootpath=billboard_path, output_path=output_path)
    with open('%s/billboard_data.pickle' % output_path, 'w') as f:
        pickle.dump( (X, Y, B, T, F, L), f)
