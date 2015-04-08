#!/usr/bin/python
#
# Copyright (C) Christian Thurau, 2010. 
# Licensed under the GNU General Public License (GPL). 
# http://www.gnu.org/licenses/gpl.txt
"""
PyMF K-means clustering (unary-convex matrix factorization).
Copyright (C) Christian Thurau, 2010. GNU General Public License (GPL). 
"""



import numpy as np

import dist
from nmf import NMF

__all__ = ["Cmeans"]

class Cmeans(NMF):
    """      
    cmeans(data, num_bases=4)
    
    
    Fuzzy c-means soft clustering. Factorize a data matrix into two matrices s.t.
    F = | data - W*H | is minimal. H is restricted to convexity (columns
    sum to 1) W    is simply the weighted mean over the corresponding samples in 
    data. Note that the objective function is based on distances (?), hence the
    Frobenius norm is probably not a good quality measure.
    
    Parameters
    ----------
    data : array_like, shape (_data_dimension, _num_samples)
        the input data
    num_bases: int, optional
        Number of bases to compute (column rank of W and row rank of H).
        4 (default)        
    

    Attributes
    ----------
    W : "data_dimension x num_bases" matrix of basis vectors
    H : "num bases x num_samples" matrix of coefficients
    ferr : frobenius norm (after calling .factorize()) 
    
    Example
    -------
    Applying C-means to some rather stupid data set:
    
    >>> import numpy as np
    >>> from cmeans import Cmeans
    >>> data = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 1.0]])
    >>> cmeans_mdl = Cmeans(data, num_bases=2, niter=10)
    >>> cmeans_mdl.initialization()
    >>> cmeans_mdl.factorize()
    
    The basis vectors are now stored in cmeans_mdl.W, the coefficients in cmeans_mdl.H. 
    To compute coefficients for an existing set of basis vectors simply    copy W 
    to cmeans_mdl.W, and set compute_w to False:
    
    >>> data = np.array([[1.5], [1.2]])
    >>> W = [[1.0, 0.0], [0.0, 1.0]]
    >>> cmeans_mdl = Cmeans(data, num_bases=2)
    >>> cmeans_mdl.initialization()
    >>> cmeans_mdl.W = W
    >>> cmeans_mdl.factorize(compute_w=False, niter=50)
    
    The result is a set of coefficients kmeans_mdl.H, s.t. data = W * kmeans_mdl.H.
    """

    def update_h(self):                    
        # assign samples to best matching centres ...
        m = 1.75
        tmp_dist = dist.pdist(self.W, self.data, metric='l2') + self._EPS
        self.H[:,:] = 0.0
        
        for i in range(self._num_bases):
            for k in range(self._num_bases):                
                self.H[i,:] += (tmp_dist[i,:]/tmp_dist[k,:])**(2.0/(m-1))
            
        self.H = np.where(self.H>0, 1.0/self.H, 0)    
                    
    def update_w(self):            
        for i in range(self._num_bases):
            tmp = (self.H[i:i+1,:] * self.data).sum(axis=1)
            self.W[:,i] = tmp/(self.H[i,:].sum() + self._EPS)        
