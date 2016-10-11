"""Configuration for the C-NMF algorithm."""

import numpy as np

config = {
    "h": 20,
    "R": 16,
    "rank": 3,
    "R_labels": 16,
    "rank_labels": 4,
    "niters": 500,
    "norm_feats": np.inf  # min_max, log, np.inf,
                          # -np.inf, float >= 0, None
}

algo_id = "cnmf"
is_boundary_type = True
is_label_type = True
