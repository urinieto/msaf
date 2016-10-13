"""Config for the Foote algorithm."""

import msaf

# Foote Params
config = {
    "M_gaussian": 66,
    "m_median": 12,
    "L_peaks": 64,
    "bound_norm_feats": "min_max"  # "min_max", "log", np.inf,
                                   # -np.inf, float >= 0, None

    # Framesync
    # "M_gaussian"    : msaf.utils.seconds_to_frames(28),
    # "m_median"      : msaf.utils.seconds_to_frames(12),
    # "L_peaks"       : msaf.utils.seconds_to_frames(18)
}

algo_id = "foote"
is_boundary_type = True
is_label_type = False
