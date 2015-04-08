/*
 *  cluster_segmenter.c
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

#include "cluster_segmenter.h"

extern int readmatarray_size(const char *filepath, int n_array, int* t, int* d);
extern int readmatarray(const char *filepath, int n_array, int t, int d, double** arr);


/* return histograms h[nx*m] of data x[nx] into m bins using a sliding window of length h_len (MUST BE ODD) */
/* NB h is a vector in row major order, as required by cluster_melt() */
/* for historical reasons we normalise the histograms by their norm (not to sum to one) */
void create_histograms(int* x, int nx, int m, int hlen, double* h)
{
	int i, j, t;
	double norm;

	for (i = 0; i < nx*m; i++) 
	        h[i] = 0;

	for (i = hlen/2; i < nx-hlen/2; i++)
	{
		for (j = 0; j < m; j++)
			h[i*m+j] = 0;
		for (t = i-hlen/2; t <= i+hlen/2; t++)
			++h[i*m+x[t]];
		norm = 0;
		for (j = 0; j < m; j++)
			norm += h[i*m+j] * h[i*m+j];
		for (j = 0; j < m; j++)
			h[i*m+j] /= norm;
	}
	
	/* duplicate histograms at beginning and end to create one histogram for each data value supplied */
	for (i = 0; i < hlen/2; i++)
		for (j = 0; j < m; j++)
			h[i*m+j] = h[hlen/2*m+j];
	for (i = nx-hlen/2; i < nx; i++)
		for (j = 0; j < m; j++)
			h[i*m+j] = h[(nx-hlen/2-1)*m+j];
}

/* segment using HMM and then histogram clustering */
void cluster_segment(int* q, double** features, int frames_read, int feature_length, int nHMM_states, 
					 int histogram_length, int nclusters, int neighbour_limit)
{
	int i, j;
	
	/*****************************/
	if (0) {
	/* try just using the predominant bin number as a 'decoded state' */
	nHMM_states = feature_length + 1;	/* allow a 'zero' state */
	double chroma_thresh = 0.05;
	double maxval;
	int maxbin;
	for (i = 0; i < frames_read; i++)
	{
		maxval = 0;
		for (j = 0; j < feature_length; j++)
		{
			if (features[i][j] > maxval) 
			{
				maxval = features[i][j];
				maxbin = j;
			}				
		}
		if (maxval > chroma_thresh)
			q[i] = maxbin;
		else
			q[i] = feature_length;
	}
	
	}
	if (1) {
	/*****************************/
		
	
	/* scale all the features to 'balance covariances' during HMM training */
	double scale = 10;
	for (i = 0; i < frames_read; i++)
		for (j = 0; j < feature_length; j++)
			features[i][j] *= scale;
	
	/* train an HMM on the features */
	
	/* create a model */
	model_t* model = hmm_init(features, frames_read, feature_length, nHMM_states);
	
	/* train the model */
	hmm_train(features, frames_read, model);
/*	
	printf("\n\nafter training:\n");
	hmm_print(model);
*/	
	/* decode the hidden state sequence */
	viterbi_decode(features, frames_read, model, q);  
	hmm_close(model);
	
	/*****************************/
	}
	/*****************************/
	
    
/*
	fprintf(stderr, "HMM state sequence:\n");
	for (i = 0; i < frames_read; i++)
		fprintf(stderr, "%d ", q[i]);
	fprintf(stderr, "\n\n");
*/
	
	/* create histograms of states */
	double* h = (double*) malloc(frames_read*nHMM_states*sizeof(double));	/* vector in row major order */
	create_histograms(q, frames_read, nHMM_states, histogram_length, h);
	
	/* cluster the histograms */
	int nbsched = 20;	/* length of inverse temperature schedule */
	double* bsched = (double*) malloc(nbsched*sizeof(double));	/* inverse temperature schedule */
	double b0 = 100;
	double alpha = 0.7;
	bsched[0] = b0;
	for (i = 1; i < nbsched; i++)
		bsched[i] = alpha * bsched[i-1];
	cluster_melt(h, nHMM_states, frames_read, bsched, nbsched, nclusters, neighbour_limit, q);
	
	/* now q holds a sequence of cluster assignments */
	
	free(h);  
	free(bsched);
}
