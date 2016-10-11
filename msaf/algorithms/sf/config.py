"""Config for the Structural Features algorithm."""

import numpy as np

# Serra params
config = {
    "M_gaussian": 27,
    "m_embedded": 3,
    "k_nearest": 0.04,
    "Mp_adaptive": 28,
    "offset_thres": 0.05,
    "bound_norm_feats": np.inf  # min_max, log, np.inf,
                                # -np.inf, float >= 0, None

    # For framesync features
    # "M_gaussian"    : 100,
    # "m_embedded"    : 3,
    # "k_nearest"     : 0.06,
    # "Mp_adaptive"   : 100,
    # "offset_thres"  : 0.01
}

algo_id = "sf"
is_boundary_type = True
is_label_type = False
