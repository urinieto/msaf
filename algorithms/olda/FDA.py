#!/usr/bin/env python
# CREATED:2013-05-12 10:52:56 by Brian McFee <brm2132@columbia.edu>
# fisher discriminant analysis with regularization 


import numpy as np
import scipy.linalg
from sklearn.base import BaseEstimator, TransformerMixin

class FDA(BaseEstimator, TransformerMixin):

    def __init__(self, alpha=1e-4):
        '''Fisher discriminant analysis

        Arguments:
        ----------
        alpha : float
            Regularization parameter
        '''

        self.alpha = alpha


    def fit(self, X, Y):
        '''Fit the LDA model

        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            Training data

        Y : array-like, shape [n_samples]
            Training labels

        Returns
        -------
        self : object
        '''
        

        n, d_orig           = X.shape
        classes             = np.unique(Y)

        assert(len(Y) == n)

        mean_global         = np.mean(X, axis=0, keepdims=True)
        scatter_within      = self.alpha * np.eye(d_orig)
        scatter_between     = np.zeros_like(scatter_within)

        for c in classes:
            n_c             = np.sum(Y==c)
            if n_c < 2:
                continue
            mu_diff         = np.mean(X[Y==c], axis=0, keepdims=True) - mean_global
            scatter_between = scatter_between + n_c * np.dot(mu_diff.T, mu_diff)
            scatter_within  = scatter_within  + n_c * np.cov(X[Y==c], rowvar=0)

        e_vals, e_vecs      = scipy.linalg.eig(scatter_between, scatter_within)

        self.e_vals_        = e_vals
        self.e_vecs_        = e_vecs
        
        self.components_    = e_vecs.T

        return self

    def transform(self, X):
        '''Transform data by FDA

        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            Data to be transformed

        Returns
        -------
        X_new : array, shape (n_samples, n_atoms)
        '''

        return X.dot(self.components_.T)
