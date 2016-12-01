"""Config for the 2D-FMC."""

import numpy as np

# 2D-FMC Params
config = {
    "dirichlet": False,
    "xmeans": False,
    "k": 4,
    "2dfmc_offset": 4,  # Number of frames to ignore in the beginning and end of each segment
    "label_norm_feats": np.inf,  # "min_max", "log", np.inf,
                                 # -np.inf, float >= 0, None
    "label_norm_floor": 0.1,
    "label_norm_min_db": -80
}

# Other params
algo_id = "fmc2d"
is_boundary_type = False
is_label_type = True
