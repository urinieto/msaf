"""Configuration file for the Variable Markov Oracle method"""

# Algorithm Params
config = {
    # "method": 'scluster',
    "connectivity": 'lrs',
    "median_filter_width": 9,
    "hier_num_layers": 10,  # How many hierarchical layers to compute (only for the hierarchical case)
    "vmo_k": 10,
}

algo_id = "vmo"  # Identifier of the algorithm
is_boundary_type = True  # Whether the algorithm extracts boundaries
is_label_type = True  # Whether the algorithm labels segments
