"""Config for the Constrained Clustering algorithm."""

# Levy params (from original paper)
config = {
    "nHMMStates"            : 80,
    "nclusters"             : 6,
    "neighbourhoodLimit"    : 16
}

algo_id = "cc"
is_boundary_type = True
is_label_type = True
