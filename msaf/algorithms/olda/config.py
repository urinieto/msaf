"""Config for the OLDA algorithm."""
import os

# OLDA params
config = {
    #"transform": os.path.join(os.path.dirname(os.path.realpath(__file__)),
                              #"models", "EstBeats_SALAMI-i.npy")
    #"transform": os.path.join(os.path.dirname(os.path.realpath(__file__)),
                              #"models", "EstBeats_BeatlesIso.npy")
    "transform": os.path.join(os.path.dirname(os.path.realpath(__file__)),
                              "models", "EstBeats_BeatlesTUT.npy")
}

algo_id = "olda"
is_boundary_type = True
is_label_type = False
