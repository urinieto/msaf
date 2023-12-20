"""Example of a configuration file for a new algorithm"""

# Algorithm Params
config = {
    "ssm_type": "rbf",
    "penalty_weight": 1,
    "penalty_func": "modulo8",
    "bands_number": 7,
    "max_size": 32
}

algo_id = "cbm"  # Identifier of the algorithm
is_boundary_type = True  # Whether the algorithm extracts boundaries
is_label_type = False  # Whether the algorithm labels segments
