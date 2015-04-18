"""Config for the Structural Features algorithm."""
import os

prefix_path = "/home/uri/Projects/"
#prefix_path = "/Users/uriadmin/NYU/Spring15/"

# Serra params
config = {
    "M_gaussian"    : 25,
    "m_embedded"    : 1,
    "k_nearest"     : 0.03,
    "Mp_adaptive"   : 28,
    "offset_thres"  : 0.05,
    "w"             : 5,
    "beats"         : True,
    "recplot_type"  : "proba",  # predict, proba, mask
    "model_type"    : "iso",
    "recplots_dir_beats"  : prefix_path + "similarity_classification/recplots_beats",
    "features_dir_beats"  : prefix_path + "similarity_classification/features_beats",
    "recplots_dir_subbeats"  : prefix_path + "similarity_classification/recplots_subbeats",
    "features_dir_subbeats"  : prefix_path + "similarity_classification/features_subbeats",
    #"model"         : os.path.join(os.path.dirname(os.path.realpath(__file__)),
                              #"models")
    "model"         : None

    # For framesync features
    #"M_gaussian"    : 100,
    #"m_embedded"    : 3,
    #"k_nearest"     : 0.06,
    #"Mp_adaptive"   : 100,
    #"offset_thres"  : 0.01
}

algo_id = "sfdtw"
is_boundary_type = True
is_label_type = False
