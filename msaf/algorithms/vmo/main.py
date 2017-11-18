

from vmo.analysis.segmentation import eigen_decomposition
import vmo
import vmo.analysis as van
from ..scluster.main2 import *
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
        est_idxs, est_labels = cluster(embedding, Cnorm, config["k"], in_bound_idxs)
        est_labels = np.asarray(est_labels, dtype=np.int)

    return est_idxs, est_labels, Cnorm
