"""Config for the 2D-FMC."""

# 2D-FMC Params
config = {
    "dirichlet": False,
    "xmeans": False,
    "k": 6,
    "norm_feats": "log",  # "min_max", "log", np.inf,
                          # -np.inf, float >= 0, None
    "norm_floor": 0.1,
    "norm_min_db": -80
}

# Other params
algo_id = "fmc2d"
is_boundary_type = False
is_label_type = True
