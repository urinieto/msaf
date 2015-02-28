#ifndef _CLUSTER_SEGMENTER_H
#define _CLUSTER_SEGMENTER_H

/*
 *  cluster_segmenter.h
 *  soundbite
 *
 *  Created by Mark Levy on 06/04/2006.
 *  Copyright 2006 Centre for Digital Music, Queen Mary, University of London.

    This program is free software; you can redistribute it and/or
    modify it under the terms of the GNU General Public License as
    published by the Free Software Foundation; either version 2 of the
    License, or (at your option) any later version.  See the file
    COPYING included with this distribution for more information.
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

#include "segment.h"
#include "cluster_melt.h"
#include "hmm/hmm.h"

#ifdef __cplusplus
extern "C" {
#endif

void create_histograms(int* x, int nx, int m, int hlen, double* h);

void cluster_segment(int* q, double** features, int frames_read, int feature_length, int nHMM_states, 
					 int histogram_length, int nclusters, int neighbour_limit);


#ifdef __cplusplus
}
#endif

#endif
