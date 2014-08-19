"""Config for the 2D-FMC."""

# 2D-FMC Params
config = {
    "niter"             :   200,
    "alphaZ"            :   -0.01,
    "normalize_frames"  :   True,
    "viterbi_segmenter" :   True,
    "min_segment_length":   32
}

# Other params
algo_id = "2dfmc"
is_boundary_type = False
is_label_type = True
