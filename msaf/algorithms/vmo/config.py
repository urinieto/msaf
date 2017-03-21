"""Configuration file for the Variable Markov Oracle method"""

# Algorithm Params
config = {
    "method": 'symbol_spectral',
    "connectivity": 'sfx'
}

algo_id = "vmo"  # Identifier of the algorithm
is_boundary_type = True  # Whether the algorithm extracts boundaries
is_label_type = True  # Whether the algorithm labels segments
