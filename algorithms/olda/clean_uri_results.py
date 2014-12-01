#!/usr/bin/env python

import sys

import ujson as json
import numpy as np

def load_json(infile):

    with open(infile, 'r') as f:
        data = json.load(f)['sections']


    times = map(lambda x: x['start'], filter(lambda x: x['bound'], data))
    times.append(0)
    times.sort()
    return np.array(times)

def save_segmentation(outfile, times):

    with open(outfile, 'w') as f:
        for idx, (start, end) in enumerate(zip(times[:-1], times[1:]), 1):
            f.write('%.3f\t%.3f\tSe#%03d\n' % (start, end, idx))

    pass

if __name__ == '__main__':

    times = load_json(sys.argv[1])
    save_segmentation(sys.argv[2], times)
