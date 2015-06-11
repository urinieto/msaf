"""
Set of util functions for the section similarity project.
"""

import copy
import numpy as np
import json
import scipy.fftpack
import pylab as plt

def resample_mx(X, incolpos, outcolpos):
    """
    Y = resample_mx(X, incolpos, outcolpos)
    X is taken as a set of columns, each starting at 'time'
    colpos, and continuing until the start of the next column.
    Y is a similar matrix, with time boundaries defined by
    outcolpos.  Each column of Y is a duration-weighted average of
    the overlapping columns of X.
    2010-04-14 Dan Ellis dpwe@ee.columbia.edu  based on samplemx/beatavg
    -> python: TBM, 2011-11-05, TESTED
    """
    noutcols = len(outcolpos)
    Y = np.zeros((X.shape[0], noutcols))
    # assign 'end times' to final columns
    if outcolpos.max() > incolpos.max():
        incolpos = np.concatenate([incolpos,[outcolpos.max()]])
        X = np.concatenate([X, X[:,-1].reshape(X.shape[0],1)], axis=1)
    outcolpos = np.concatenate([outcolpos, [outcolpos[-1]]])
    # durations (default weights) of input columns)
    incoldurs = np.concatenate([np.diff(incolpos), [1]])

    for c in range(noutcols):
        firstincol = np.where(incolpos <= outcolpos[c])[0][-1]
        firstincolnext = np.where(incolpos < outcolpos[c+1])[0][-1]
        lastincol = max(firstincol,firstincolnext)
        # default weights
        wts = copy.deepcopy(incoldurs[firstincol:lastincol+1])
        # now fix up by partial overlap at ends
        if len(wts) > 1:
            wts[0] = wts[0] - (outcolpos[c] - incolpos[firstincol])
            wts[-1] = wts[-1] - (incolpos[lastincol+1] - outcolpos[c+1])
        wts = wts * 1. / float(sum(wts))
        Y[:,c] = np.dot(X[:,firstincol:lastincol+1], wts)
    # done
    return Y

def magnitude(X):
    """Magnitude of a complex matrix."""
    r = np.real(X)
    i = np.imag(X)
    return np.sqrt(r * r + i * i);

def json_to_bounds(segments_json):
    """Extracts the boundaries from a json file and puts them into
        an np array."""
    f = open(segments_json)
    segments = json.load(f)["segments"]
    bounds = []
    for segment in segments:
        bounds.append(segment["start"])
    bounds.append(bounds[-1] + segments[-1]["duration"]) # Add last boundary
    f.close()
    return np.asarray(bounds)

def json_bounds_to_bounds(bounds_json):
    """Extracts the boundaries from a bounds json file and puts them into
        an np array."""
    f = open(bounds_json)
    segments = json.load(f)["bounds"]
    bounds = []
    for segment in segments:
        bounds.append(segment["start"])
    f.close()
    return np.asarray(bounds)

def json_to_labels(segments_json):
    """Extracts the labels from a json file and puts them into
        an np array."""
    f = open(segments_json)
    segments = json.load(f)["segments"]
    labels = []
    str_labels = []
    for segment in segments:
        if not segment["label"] in str_labels:
            str_labels.append(segment["label"])
            labels.append(len(str_labels)-1)
        else:
            label_idx = np.where(np.asarray(str_labels) == segment["label"])[0][0]
            labels.append(label_idx)
    f.close()
    return np.asarray(labels)

def json_to_beats(beats_json_file):
    """Extracts the beats from the beats_json_file and puts them into
        an np array."""
    f = open(beats_json_file, "r")
    beats_json = json.load(f)
    beats = []
    for beat in beats_json["beats"]:
        beats.append(beat["start"])
    f.close()
    return np.asarray(beats)

def analyze_results(file):
    f = open(file, "r")
    lines = f.readlines()
    F = []
    for line in lines:
        F.append(float(line.split("\t")[0]))
    f.close()
    #print np.mean(F)

def compute_ffmc2d(X):
    """Computes the 2D-Fourier Magnitude Coefficients."""
    # 2d-fft
    fft2 = scipy.fftpack.fft2(X)

    # Magnitude
    fft2m = magnitude(fft2)

    # FFTshift and flatten
    fftshift = scipy.fftpack.fftshift(fft2m).flatten()

    #cmap = plt.cm.get_cmap('hot')
    #plt.imshow(np.log1p(scipy.fftpack.fftshift(fft2m)).T, interpolation="nearest",
    #    aspect="auto", cmap=cmap)
    #plt.show()

    # Take out redundant components
    return fftshift[:fftshift.shape[0] // 2 + 1]
