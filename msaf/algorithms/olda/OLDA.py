# CREATED:2013-08-19 16:12:01 by Brian McFee <brm2132@columbia.edu>
#
# Ordinal LDA

import itertools
import numpy as np
import scipy.linalg
from sklearn.base import BaseEstimator, TransformerMixin

class OLDA(BaseEstimator, TransformerMixin):

    def __init__(self, sigma=1e-4):
        '''Ordinal linear discriminant analysis

        Arguments:
        ----------
        sigma : float
            Regularization parameter
        '''

        self.sigma = sigma
        self.scatter_ordinal_ = None
        self.scatter_within_ = None


    def fit(self, X, Y):
        '''Fit the OLDA model

        Parameters
        ----------
        X : array-like, shape [n_samples]
            Training data: each example is an n_features-by-* data array

        Y : array-like, shape [n_samples]
            Training labels: each label is an array of change-points
                             (eg, a list of segment boundaries)

        Returns
        -------
        self : object
        '''
        
        # Re-initialize the scatter matrices
        self.scatter_ordinal_ = None
        self.scatter_within_  = None
        
        # Reduce to partial-fit
        self.partial_fit(X, Y)
        
        return self
        
    def partial_fit(self, X, Y):
        '''Partial-fit the OLDA model

        Parameters
        ----------
        X : array-like, shape [n_samples]
            Training data: each example is an n_features-by-* data array

        Y : array-like, shape [n_samples]
            Training labels: each label is an array of change-points
                             (eg, a list of segment boundaries)

        Returns
        -------
        self : object
        '''
        
        for (xi, yi) in itertools.izip(X, Y):
            
            prev_mean       = None
            prev_length     = None
            
            if self.scatter_within_ is None:
                # First round: initialize
                d, n = xi.shape
                
                if yi[0] > 0:
                    yi = np.concatenate([np.array([0]), yi])
                if yi[-1] < n:
                    yi = np.concatenate([yi, np.array([n])])
                    
                self.scatter_within_  = self.sigma * np.eye(d)
                self.scatter_ordinal_ = np.zeros(d)
                
            
            # iterate over segments
            for (seg_start, seg_end) in zip(yi[:-1], yi[1:]):
            
                seg_length = seg_end - seg_start
                
                if seg_length < 2:
                    continue

                seg_mean = np.mean(xi[:, seg_start:seg_end], axis=1, keepdims=True)
                seg_cov  = np.cov(xi[:, seg_start:seg_end])    
                self.scatter_within_ = self.scatter_within_ + seg_length * seg_cov
                
                
                if prev_mean is not None:
                    diff_ord = seg_mean - (prev_length * prev_mean + seg_length * seg_mean) / (prev_length + seg_length)
                    self.scatter_ordinal_ = self.scatter_ordinal_ + seg_length * np.dot(diff_ord, diff_ord.T)
                    
                    diff_ord = prev_mean - (prev_length * prev_mean + seg_length * seg_mean) / (prev_length + seg_length)
                    self.scatter_ordinal_ = self.scatter_ordinal_ + prev_length * np.dot(diff_ord, diff_ord.T)

                prev_mean = seg_mean
                prev_length = seg_length
        
        e_vals, e_vecs = scipy.linalg.eig(self.scatter_ordinal_, self.scatter_within_)
        self.e_vals_ = e_vals
        self.e_vecs_ = e_vecs
        self.components_ = e_vecs.T
        return self

    
    def transform(self, X):
        '''Transform data by FDA

        Parameters
        ----------
        X : array-like, shape [n_samples]
            Data to be transformed. Each example is a d-by-* feature matrix

        Returns
        -------
        X_new : array, shape (n_samples)
        '''

        return [self.components_.dot(xi) for xi in X]
