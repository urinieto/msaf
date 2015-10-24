"""Config for the Foote algorithm."""

import msaf

# Foote Params
config = {
    "M_gaussian"    : 66,
    "m_median"      : 12,
    "L_peaks"       : 64

    # Framesync
    #"M_gaussian"    : msaf.utils.seconds_to_frames(28),
    #"m_median"      : msaf.utils.seconds_to_frames(12),
    #"L_peaks"       : msaf.utils.seconds_to_frames(18)
}

algo_id = "foote"
is_boundary_type = True
is_label_type = False
