"""Config for the Spectral Clustering algorithm."""

import os

prefix_path = "/home/uri/Projects/"
#prefix_path = "/Users/uriadmin/NYU/Spring15/"

# Spectral Clustering Params
config = {
    "verbose"       : False,
    "median"        : False,
    "num_types"     : None,
    "start_layer"   : 1,
    "num_layers"    : 10,
    "hier"          : False,
    "w"             : 5,
    "beats"         : True,
    "bias"          : True,
    "recplot_type"  : "proba",  # predict, proba, mask
    "model_type"    : "salami",
    "recplots_dir_beats"  : prefix_path + "similarity_classification/recplots_beats",
    "features_dir_beats"  : prefix_path + "similarity_classification/features_beats",
    "recplots_dir_subbeats"  : prefix_path + "similarity_classification/recplots_subbeats",
    "features_dir_subbeats"  : prefix_path + "similarity_classification/features_subbeats",
    "model_local"   : os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                   "..", "sfdtw", "models"),
    "model"         : os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                   "..", "sfdtw", "models")
    #"model"         : None
}

algo_id = "scluster"
is_boundary_type = True
is_label_type = True
