#!/usr/bin/env python

import sys

import mir_eval
import numpy as np

def onetotwo(input_file, output_file):

    times = mir_eval.util.import_segment_boundaries(input_file, cols=[0])

    new_times = zip(times[:-1], times[1:])

    np.savetxt(output_file, new_times, fmt='%.8f', delimiter='\t')
    return
    

if __name__ == '__main__':

    onetotwo(sys.argv[1], sys.argv[2])
