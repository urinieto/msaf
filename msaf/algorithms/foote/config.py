"""Config for the Foote algorithm."""

import msaf
import os

prefix_path = "/home/uri/Projects/"
#prefix_path = "/Users/uriadmin/NYU/Spring15/"

# Foote Params
config = {
    "M_gaussian"    : 68,
    "m_median"      : 12,
    "L_peaks"       : 64,

    "diag_filter"   : 1,
    "w"             : 5,
    "beats"         : True,
    "recplot_type"  : "proba",  # predict, proba, mask
    "model_type"    : "iso",
    "recplots_dir_beats"  : prefix_path + "similarity_classification/recplots_beats",
    "features_dir_beats"  : prefix_path + "similarity_classification/features_beats",
    "recplots_dir_subbeats"  : prefix_path + "similarity_classification/recplots_subbeats",
    "features_dir_subbeats"  : prefix_path + "similarity_classification/features_subbeats",
    "model"         : os.path.join(os.path.dirname(os.path.realpath(__file__)),
                              "..", "sfdtw", "models")
    #"model"         : None
    # Framesync
    #"M_gaussian"    : msaf.utils.seconds_to_frames(28),
    #"m_median"      : msaf.utils.seconds_to_frames(12),
    #"L_peaks"       : msaf.utils.seconds_to_frames(18)
}

algo_id = "foote"
is_boundary_type = True
is_label_type = False
