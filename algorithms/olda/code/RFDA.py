# CREATED:2013-11-30 14:22:33 by Brian McFee <brm2132@columbia.edu>
#
# Restricted FDA
#   only compute between-class scatter within each song

import itertools
import numpy as np
import scipy.linalg
from sklearn.base import BaseEstimator, TransformerMixin

class RFDA(BaseEstimator, TransformerMixin):

    def __init__(self, sigma=1e-4):
        '''Ordinal linear discriminant analysis

        Arguments:
        ----------
        sigma : float
            Regularization parameter
        '''

        self.sigma = sigma
        self.scatter_restricted_ = None
        self.scatter_within_ = None

    def fit(self, X, Y):
        '''Fit the RFDA model

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
        self.scatter_restricted_ = None
        self.scatter_within_  = None
        
        # Reduce to partial-fit
        self.partial_fit(X, Y)
        
        return self
        
    def partial_fit(self, X, Y):
        '''Partial-fit the RFDA model

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
            
            if self.scatter_within_ is None:
                # First round: initialize
                d, n = xi.shape
                
                if yi[0] > 0:
                    yi = np.concatenate([np.array([0]), yi])
                if yi[-1] < n:
                    yi = np.concatenate([yi, np.array([n])])
                    
                self.scatter_within_  = self.sigma * np.eye(d)
                self.scatter_restricted_ = np.zeros(d)
                
            # compute the mean and cov of each segment
            global_mean = np.mean(xi, axis=1, keepdims=True)
            
            # iterate over segments
            for (seg_start, seg_end) in zip(yi[:-1], yi[1:]):
            
                seg_length = seg_end - seg_start
                
                if seg_length < 2:
                    continue

                seg_mean = np.mean(xi[:, seg_start:seg_end], axis=1, keepdims=True)
                
                mu_diff = seg_mean - global_mean
                self.scatter_restricted_ = self.scatter_restricted_ + seg_length * np.dot(mu_diff, mu_diff.T)

                seg_cov  = np.cov(xi[:, seg_start:seg_end])
                
                self.scatter_within_ = self.scatter_within_ + seg_length * seg_cov
                
                
        e_vals, e_vecs = scipy.linalg.eig(self.scatter_restricted_, self.scatter_within_)
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
