
import sklearn
import numpy as np
import vmo
import vmo.analysis as van
import scipy.linalg
import scipy.ndimage
import librosa


def vmo_routine(feature):
    ideal_t = vmo.find_threshold(feature, dim=feature.shape[1])
    oracle = vmo.build_oracle(feature, flag='a', threshold=ideal_t[0][1], dim=feature.shape[1])

    return oracle


def connectivity_from_vmo(oracle, config):

    median_filter_width = config['median_filter_width']

    connectivity = van.create_selfsim(oracle, method=config['connectivity'])
    obs_len = oracle.n_states-1

    df = librosa.segment.timelag_filter(scipy.ndimage.median_filter)
    connectivity = df(connectivity, size=(1, median_filter_width))

    connectivity[range(1, obs_len), range(obs_len - 1)] = 1.0
    connectivity[range(obs_len - 1), range(1, obs_len)] = 1.0
    connectivity[np.diag_indices(obs_len)] = 0

    return connectivity


def eigen_decomposition(mat, k=6):  # Changed from 11 to 8 then to 6(7/22)
    vals, vecs = scipy.linalg.eig(mat)
    vals = vals.real
    vecs = vecs.real
    idx = np.argsort(vals)

    vals = vals[idx]
    vecs = vecs[:, idx]

    if len(vals) < k + 1:
        k = -1
    vecs = scipy.ndimage.median_filter(vecs, size=(5,1))
    return vecs[:, :k]


def cluster(evecs, Cnorm, k, in_bound_idxs=None):
    X = evecs[:, :k] / (Cnorm[:, k - 1:k] + 1e-5)
    KM = sklearn.cluster.KMeans(n_clusters=k, n_init=50, max_iter=500)
    seg_ids = KM.fit_predict(X)

    ###############################################################
    # Locate segment boundaries from the label sequence
    if in_bound_idxs is None:
        bound_beats = 1 + np.flatnonzero(seg_ids[:-1] != seg_ids[1:])

        # Count beats 0 as a boundary
        bound_idxs = librosa.util.fix_frames(bound_beats, x_min=0)

    else:
        bound_idxs = in_bound_idxs
        if len(bound_idxs) <= k:
            k = len(bound_idxs)

    X_sync = librosa.util.utils.sync(X.T, bound_idxs, aggregate=np.mean)
    c = sklearn.cluster.KMeans(n_clusters=k, n_init=50, max_iter=500)
    bound_segs = c.fit_predict(X_sync.T)

    # Compute the segment label for each boundary
    # bound_segs = list(seg_ids[bound_idxs])

    # Tack on the end-time
    bound_idxs = list(np.append(bound_idxs, len(Cnorm) - 1))

    return bound_idxs, bound_segs


def scluster_segment(feature, config, in_bound_idxs=None):
    v_oracle = vmo_routine(feature)
    connectivity_mat = connectivity_from_vmo(v_oracle, config)
    embedding = eigen_decomposition(connectivity_mat, k=config["hier_num_layers"])

    Cnorm = np.cumsum(embedding ** 2, axis=1) ** 0.5

    if config["hier"]:
        est_idxs = []
        est_labels = []
        for k in range(1, config["hier_num_layers"] + 1):
            est_idx, est_label = cluster(embedding, Cnorm, k)
            est_idxs.append(est_idx)
            est_labels.append(np.asarray(est_label, dtype=np.int))

    else:
        est_idxs, est_labels = cluster(embedding, Cnorm, config["vmo_k"], in_bound_idxs)
        est_labels = np.asarray(est_labels, dtype=np.int)

    return est_idxs, est_labels, Cnorm
