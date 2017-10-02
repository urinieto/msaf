"""Config for the Spectral Clustering algorithm."""

# Spectral Clustering Params
config = {
    "num_layers" : 10,   # How many hierarchical layers to compute (only for the hierarchical case)
    "scluster_k" : 4,    # How many unique labels to have (only for the flat case)
    "evec_smooth": 9,
    "rec_smooth" : 9,
    "rec_width"  : 9
}

algo_id = "scluster"
is_boundary_type = True
is_label_type = True
