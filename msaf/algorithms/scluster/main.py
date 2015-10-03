# CREATED:2013-08-22 12:20:01 by Brian McFee <brm2132@columbia.edu>
'''Music segmentation using timbre, pitch, repetition and time.
'''


import sys
import os
import argparse
import logging
import string

import numpy as np
import scipy.spatial
import scipy.signal
import scipy.linalg

import sklearn.cluster

import msaf

# Requires librosa-develop 0.3 branch
import librosa

# Suppress neighbor links within REP_WIDTH beats of the current one
REP_WIDTH=3

# Only consider repetitions of at least (FILTER_WIDTH-1)/2
FILTER_WIDTH=1 + 2 * 8

# How much mass should we put along the +- diagonals?  We don't want this to influence nodes with high degree
# If we set the kernel weights appropriately, most edges should have weight >= exp(-0.5)
# Let's set the ridge flow to a small constant
RIDGE_FLOW = np.exp(-2.0)

# How much state to use?
N_STEPS = 2

# Local model
N_MELS = 128
N_MFCC = 13

# Which similarity metric to use?
METRIC='sqeuclidean'

# Sample rate for signal analysis
SR=22050

# Hop length for signal analysis
HOP_LENGTH=512

# Maximum number of structural components to consider
MAX_REP=10

# Minimum and maximum average segment duration
MIN_SEG=10.0
MAX_SEG=30.0

# Minimum tempo threshold; if we dip below this, double the bpm estimator and resample
MIN_TEMPO=70.0

# Minimum duration (in beats) of a "non-repeat" section
MIN_NON_REPEATING = (FILTER_WIDTH - 1) / 2

SEGMENT_NAMES = list(string.ascii_uppercase)
for x in string.ascii_uppercase:
    SEGMENT_NAMES.extend(['%s%s' % (x, y) for y in string.ascii_lowercase])

def hp_sep(y):
    D_h, D_p = librosa.decompose.hpss(librosa.stft(y))
    return librosa.istft(D_h), librosa.istft(D_p)

def get_beats(y, sr, hop_length):

    odf = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length, aggregate=np.median, n_mels=128)
    bpm, beats = librosa.beat.beat_track(onset_envelope=odf, sr=sr, hop_length=hop_length, start_bpm=120)

    if bpm < MIN_TEMPO:
        bpm, beats = librosa.beat.beat_track(onset_envelope=odf, sr=sr, hop_length=hop_length, bpm=2*bpm)

    return bpm, beats

def features(filename):
    #print '\t[1/5] loading audio'
    y, sr = librosa.load(filename, sr=SR)

    #print '\t[2/5] Separating harmonic and percussive signals'
    y_perc, y_harm = hp_sep(y)

    #print '\t[3/5] detecting beats'
    bpm, beats = get_beats(y=y_perc, sr=sr, hop_length=HOP_LENGTH)

    #print '\t[4/5] generating CQT'
    M1 = np.abs(librosa.cqt(y=y_harm,
                            sr=sr,
                            hop_length=HOP_LENGTH,
                            bins_per_octave=12,
                            fmin=librosa.midi_to_hz(24),
                            n_bins=72))

    M1 = librosa.logamplitude(M1**2.0, ref_power=np.max)

    #print '\t[5/5] generating MFCC'
    S = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=HOP_LENGTH, n_mels=N_MELS)
    M2 = librosa.feature.mfcc(S=librosa.logamplitude(S), n_mfcc=N_MFCC)

    n = min(M1.shape[1], M2.shape[1])

    beats = beats[beats < n]

    beats = np.unique(np.concatenate([[0], beats]))

    times = librosa.frames_to_time(beats, sr=sr, hop_length=HOP_LENGTH)

    times = np.concatenate([times, [float(len(y)) / sr]])
    M1 = librosa.feature.sync(M1, beats, aggregate=np.median)
    M2 = librosa.feature.sync(M2, beats, aggregate=np.mean)
    return (M1, M2), times

def save_segments(outfile, boundaries, beats, labels=None):

    if labels is None:
        labels = [('Seg#%03d' % idx) for idx in range(1, len(boundaries))]

    times = beats[boundaries]
    with open(outfile, 'w') as f:
        for idx, (start, end, lab) in enumerate(list(zip(times[:-1], times[1:], labels)), 1):
            f.write('%.3f\t%.3f\t%s\n' % (start, end, lab))

    pass

def get_num_segs(duration):
    kmin = max(2, np.floor(duration / MAX_SEG).astype(int))
    kmax = max(3, np.ceil(duration / MIN_SEG).astype(int))

    return kmin, kmax

def clean_reps(S):
    # Median filter with reflected padding
    Sf = np.pad(S, [(0, 0), (FILTER_WIDTH, FILTER_WIDTH)], mode='reflect')
    Sf = scipy.signal.medfilt2d(Sf, kernel_size=(1, FILTER_WIDTH))
    Sf = Sf[:, FILTER_WIDTH:-FILTER_WIDTH]
    return Sf

def expand_transitionals(R, local=True):
    '''Sometimes, a frame does not repeat.
    Sequences of non-repeating frames are bad news, so we'll link them up together as a transitional clique.

    input:

      - filtered repetition matrix R
    '''

    n = len(R)
    R_out = R.copy()

    degree = np.sum(R, axis=0)

    start = None

    all_idx = []

    for i in range(n):
        if start is not None:
            # If we're starting a new repeating section,
            # or we're at the end
            if (i == n - 1) or (degree[i] > 0):

                # Fill in R_out[start:i, start:i]
                idx = slice(start, i)

                if i - start >= MIN_NON_REPEATING:
                    if local:
                        # Add links for all unique pairs
                        R_out[np.ix_(idx, idx)] = 1
                        R_out[idx, idx] = 0
                    else:
                        all_idx.extend(range(start, i))

                # reset the counter
                start = None

        elif degree[i] == 0:
            start = i

    if not local and all_idx:
        # Add links for all unique pairs
        R_out[np.ix_(all_idx, all_idx)] = 1
        R_out[all_idx, all_idx] = 0

    return R_out

def rw_laplacian(A):
    Dinv = np.sum(A, axis=1)**-1.0
    Dinv[~np.isfinite(Dinv)] = 1.0
    L = np.eye(A.shape[0]) - (Dinv * A.T).T
    return L

def sym_laplacian(A):
    Dinv = np.sum(A, axis=1)**-1.0

    Dinv[~np.isfinite(Dinv)] = 1.0

    Dinv = np.diag(Dinv**0.5)

    L = np.eye(len(A)) - Dinv.dot(A.dot(Dinv))

    return L

def ridge(A):

    n = len(A)

    ridge_val = RIDGE_FLOW * np.ones(n-1)

    A_out = A.copy()
    A_out[range(n-1), range(1,n)] = ridge_val
    A_out[range(1,n), range(n-1)] = ridge_val

    return A_out

def min_ridge(A, R):
    R = R.astype(np.bool)

    n = len(A)
    D = RIDGE_FLOW * np.ones(n)
    for i in range(n):
        idx = R[i]
        if idx.any():
            D[i] = np.mean(A[i, idx])

    ridge_val = np.minimum(D[:-1], D[1:])

    A_out = A.copy()
    A_out[range(n-1), range(1,n)] = ridge_val
    A_out[range(1,n), range(n-1)] = ridge_val

    return A_out

def local_ridge(A_rep, A_loc):

    n = len(A_rep)

    ridge_val = np.diag(A_loc, k=1)

    A_out = A_rep.copy()
    A_out[range(n-1), range(1,n)] = ridge_val
    A_out[range(1,n), range(n-1)] = ridge_val

    return A_out

def weighted_ridge(A_rep, A_loc):
    ''' Find mu such that mu * deg(A_rep, i) ~= (1-mu) * deg(A_loc, i)

    Goal: avg flow should be balanced between the repeater graph and the sequence graph

    '''
    d1 = np.sum(A_rep, axis=1)
    d2 = np.sum(A_loc, axis=1)

    ds = d1 + d2
    mu = d2.dot(ds) / np.dot(ds, ds)
    return mu * A_rep + (1 - mu) * A_loc

def factorize(L, k=20):
    e_vals, e_vecs = scipy.linalg.eig(L)
    e_vals = e_vals.real
    e_vecs = e_vecs.real
    idx    = np.argsort(e_vals)

    e_vals = e_vals[idx]
    e_vecs = e_vecs[:, idx]

    if len(e_vals) < k + 1:
        k = -1

    return e_vecs[:, :k].T, e_vals[k] - e_vals[k-1]

def label_rep_sections(X, boundaries, n_types):
    # Classify each segment centroid
    Xs = librosa.feature.sync(X, boundaries)

    C = sklearn.cluster.KMeans(n_clusters=n_types, tol=1e-8)

    labels = C.fit_predict(Xs.T)
    intervals = list(zip(boundaries[:-1], boundaries[1:]))

    return  intervals, labels[:len(intervals)]

def cond_entropy(y_old, y_new):
    ''' Compute the conditional entropy of y_old given y_new'''

    # P[i,j] = |y_old[i] = y_new[j]|
    P = sklearn.metrics.cluster.contingency_matrix(y_old, y_new)

    # Normalize to form the joint distribution
    P = P.astype(float) / len(y_old)

    # Marginalize
    P_new = P.sum(axis=0)

    h_old_given_new = scipy.stats.entropy(P, base=2.0)

    return P_new.dot(h_old_given_new)

def time_clusterer(Lf, k_min, k_max, times):

    best_boundaries = np.asarray([0, Lf.shape[1]])
    best_n_types    = 1
    Y_best          = Lf[:1].T

    times = np.asarray(times)

    for n_types in range(2, Lf.shape[0]):
        # Build the affinity matrix on the first n_types-1 repetition features
        Y = librosa.util.normalize(Lf[:n_types].T, norm=2, axis=1)
        # Try to label the data with n_types
        C = sklearn.cluster.KMeans(n_clusters=n_types, tol=1e-10, n_init=100)
        labels = C.fit_predict(Y)

        boundaries = 1 + np.asarray(np.where(labels[:-1] != labels[1:])).reshape((-1,))

        boundaries = np.unique(np.concatenate([[0], boundaries, [len(labels)]]))

        segment_deltas = np.diff(times[boundaries])

        # Easier to compute this before filling it out
        feasible = (np.mean(segment_deltas) >= MIN_SEG)# and (np.mean(segment_deltas) <= MAX_SEG)

        # Edge-case: always take at least 2 segment types
        if feasible:
            best_boundaries = boundaries
            best_n_types    = n_types
            Y_best          = Y

    intervals, labels = label_rep_sections(Y_best.T, best_boundaries, best_n_types)

    return best_boundaries, labels

def fixed_partition(Lf, n_types):

    # Build the affinity matrix on the first n_types-1 repetition features
    Y = librosa.util.normalize(Lf[:n_types].T, norm=2, axis=1)

    # Try to label the data with n_types
    C = sklearn.cluster.KMeans(n_clusters=n_types, tol=1e-10, n_init=100)
    labels = C.fit_predict(Y)

    boundaries = 1 + np.asarray(np.where(labels[:-1] != labels[1:])).reshape((-1,))

    boundaries = np.unique(np.concatenate([[0], boundaries, [Lf.shape[1] - 1]]))

    intervals, labels = label_rep_sections(Y.T, boundaries, n_types)

    return boundaries, labels

def median_partition(Lf, k_min, k_max, beats):
    best_score      = -np.inf
    best_boundaries = np.asarray([0, Lf.shape[1]-1])
    best_n_types    = 1
    Y_best          = Lf[:1].T

    label_dict = {}

    # The trivial solution
    label_dict[1]   = np.zeros(Lf.shape[1])

    for n_types in range(2, 1+len(Lf)):
        Y = librosa.util.normalize(Lf[:n_types].T, norm=2, axis=1)

        # Try to label the data with n_types
        C = sklearn.cluster.KMeans(n_clusters=n_types, n_init=100)
        labels = C.fit_predict(Y)
        label_dict[n_types] = labels

        # Find the label change-points
        boundaries = 1 + np.asarray(np.where(labels[:-1] != labels[1:])).reshape((-1,))

        boundaries = np.unique(np.concatenate([[0], boundaries, [Lf.shape[1] - 1]]))

        # boundaries now include start and end markers; n-1 is the number of segments
        feasible = (len(boundaries) > k_min)
        durations = np.diff([beats[x] for x in boundaries])
        med_diff = np.median(durations)

        score = -np.mean(np.abs(np.log(durations) - np.log(med_diff)))

        if score > best_score and feasible:
            best_boundaries = boundaries
            best_n_types    = n_types
            best_score      = score
            Y_best          = Y

    # Did we fail to find anything with enough boundaries?
    # Take the last one then
    if best_boundaries is None:
        best_boundaries = boundaries
        best_n_types    = n_types
        Y_best          = librosa.util.normalize(Lf[:best_n_types].T, norm=2, axis=1)

    intervals, best_labels = label_rep_sections(Y_best.T, best_boundaries, best_n_types)

    return best_boundaries, best_labels

def segment_speed(Y):
    return np.mean(np.sum(np.abs(np.diff(Y, axis=1))**2, axis=0))

def label_entropy(labels):

    values = np.unique(labels)
    hits = np.zeros(len(values))

    for v in values:
        hits[v] = np.sum(labels == v)

    hits = hits / hits.sum()

    return scipy.stats.entropy(hits)


def label_clusterer(Lf, k_min, k_max):
    best_score      = -np.inf
    best_boundaries = np.asarray([0, Lf.shape[1]-1])
    best_n_types    = 1
    Y_best          = Lf[:1].T

    label_dict = {}

    # The trivial solution
    label_dict[1]   = np.zeros(Lf.shape[1]-1)

    for n_types in range(2, 1+len(Lf)):
        Y = librosa.util.normalize(Lf[:n_types].T, norm=2, axis=1)

        # Try to label the data with n_types
        C = sklearn.cluster.KMeans(n_clusters=n_types, n_init=100)
        labels = C.fit_predict(Y)
        label_dict[n_types] = labels

        # Find the label change-points
        boundaries = 1 + np.asarray(np.where(labels[:-1] != labels[1:])).reshape((-1,))

        #boundaries = np.unique(np.concatenate([[0], boundaries, [len(labels)]]))
        boundaries = np.unique(np.concatenate([[0], boundaries, [Lf.shape[1]-1]]))

        # boundaries now include start and end markers; n-1 is the number of segments
        feasible = (len(boundaries) > k_min)

        score = label_entropy(labels) / np.log(n_types)

        if score > best_score and feasible:
            best_boundaries = boundaries
            best_n_types    = n_types
            best_score      = score
            Y_best          = Y

    # Did we fail to find anything with enough boundaries?
    # Take the last one then
    if best_boundaries is None:
        best_boundaries = boundaries
        best_n_types    = n_types
        Y_best          = librosa.util.normalize(Lf[:best_n_types].T, norm=2, axis=1)

    intervals, best_labels = label_rep_sections(Y_best.T, best_boundaries, best_n_types)

    return np.array(best_boundaries), best_labels


def estimate_bandwidth(D, k):
    D_sort = np.sort(D, axis=1)

    if 1 + k >= len(D):
        k = len(D) - 2

    sigma = np.mean(D_sort[:, 1+k])
    return sigma

def self_similarity(X, k):
    D = scipy.spatial.distance.cdist(X.T, X.T, metric=METRIC)
    sigma = estimate_bandwidth(D, k)
    A = np.exp(-0.5 * (D / sigma))
    return A

def local_similarity(X):

    d, n = X.shape

    dists = np.sum(np.diff(X, axis=1)**2, axis=0)
    # dists[i] = ||X[i] - X[i-1]||

    sigma = np.mean(dists)

    rbf = np.exp(-0.5 * (dists / sigma))

    A = np.diag(rbf, k=1) + np.diag(rbf, k=-1)
    return A


def do_segmentation(X, beats, parameters, bound_idxs):

    # If number of frames is too small, assign empty labels and quit:
    if X[0].shape[1] <= REP_WIDTH:
        if bound_idxs is not None:
            return  np.array(bound_idxs), [0] * len(bound_idxs)
        else:
            return np.array([0, X[0].shape[1] -1]), [0]

    X_rep, X_loc = X
    # Find the segment boundaries
    #print '\tpredicting segments...'
    k_min, k_max  = get_num_segs(beats[-1])
    #k_min, k_max  = 8, 32

    L = np.nan
    #while np.any(np.isnan(L)):
    # Get the raw recurrence plot
    Xpad = np.pad(X_rep, [(0,0), (N_STEPS, 0)], mode='edge')
    Xs = librosa.feature.stack_memory(Xpad, n_steps=N_STEPS)[:, N_STEPS:]

    k_link = 1 + int(np.ceil(2 * np.log2(X_rep.shape[1])))
    R = librosa.segment.recurrence_matrix(Xs,
                                        k=k_link,
                                        width=REP_WIDTH,
                                        metric=METRIC,
                                        sym=True).astype(np.float32)
    # Generate the repetition kernel
    A_rep = self_similarity(Xs, k=k_link)

    # And the local path kernel
    A_loc = self_similarity(X_loc, k=k_link)

    # Mask the self-similarity matrix by recurrence
    S = librosa.segment.structure_feature(R)

    Sf = clean_reps(S)

    # De-skew
    Rf = librosa.segment.structure_feature(Sf, inverse=True)

    # Symmetrize by force
    Rf = np.maximum(Rf, Rf.T)

    # Suppress the diagonal
    Rf[np.diag_indices_from(Rf)] = 0

    # We can jump to a random neighbor, or +- 1 step in time
    # Call it the infinite jukebox matrix
    T = weighted_ridge(Rf * A_rep,
                       (np.eye(len(A_loc),k=1) + np.eye(len(A_loc),k=-1)) * A_loc)
    # Get the graph laplacian
    try:
        L = sym_laplacian(T)
        #import pylab as plt
        #plt.imshow(L, interpolation="nearest")
        #plt.show()

        # Get the bottom k eigenvectors of L
        # TODO: Sometimes nans in L
        Lf = factorize(L, k=1 + MAX_REP)[0]
    except:
        logging.warning("Warning, nan numbers in scluster, returning only"
                        " first and last boundary")
        if parameters['hier']:
            return [np.array([0, X[0].shape[1]-1])], [[0]]
        if bound_idxs is not None:
            return bound_idxs, [0] * (len(bound_idxs) - 1)
        else:
            return np.array([0, X[0].shape[1]-1]), [0]


    if parameters['num_types']:
        boundaries, labels = fixed_partition(Lf, parameters['num_types'])
    elif parameters['median']:
        boundaries, labels = median_partition(Lf, k_min, k_max, beats)
    elif parameters['hier']:
        boundaries = []
        labels = []
        for layer in range(parameters["start_layer"],
                           parameters["start_layer"] + parameters["num_layers"]):
            layer_bounds, layer_labels = fixed_partition(Lf, layer)
            boundaries.append(layer_bounds)
            labels.append(layer_labels)
    else:
        boundaries, labels = label_clusterer(Lf, k_min, k_max)

    # Synchronize with previously found boundaries
    if bound_idxs is not None and not parameters["hier"]:
        labels = msaf.utils.synchronize_labels(bound_idxs, boundaries, labels,
                                               X[0].shape[1])
        boundaries = bound_idxs

    return boundaries, labels
