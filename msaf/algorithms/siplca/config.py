"""Config for the SI-PLCA."""

# SI-PLCA Params
config = {
    "niter"             :   200,
    "alphaZ"            :   0.01,
    "viterbi_segmenter" :   False,
    "min_segment_length":   16,
    "win"               :   32
}

algo_id = "siplca"
is_boundary_type = True
is_label_type = True
