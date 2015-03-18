"""Config for the Structural Features algorithm."""

# Serra params
config = {
    #"M_gaussian"    : 24,
    #"m_embedded"    : 3,
    #"k_nearest"     : 0.06,
    #"Mp_adaptive"   : 24,
    #"offset_thres"  : 0.04
    "M_gaussian"    : 140,
    "m_embedded"    : 3,
    "k_nearest"     : 0.03,
    "Mp_adaptive"   : 120,
    "offset_thres"  : 0.01
}

algo_id = "sf"
is_boundary_type = True
is_label_type = False
