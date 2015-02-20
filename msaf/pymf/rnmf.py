#!/usr/bin/python
#
# Copyright (C) Christian Thurau, 2010.
# Licensed under the GNU General Public License (GPL).
# http://www.gnu.org/licenses/gpl.txt
"""
PyMF Non-negative Matrix Factorization.

    NMF: Class for Non-negative Matrix Factorization

[1] Lee, D. D. and Seung, H. S. (1999), Learning the Parts of Objects by Non-negative
Matrix Factorization, Nature 401(6755), 788-799.
"""


import numpy as np
import logging
import logging.config
import scipy.sparse

from nmf import NMF

__all__ = ["RNMF"]

class RNMF(NMF):
    """
    RNMF(data, num_bases=4)


    Non-negative Matrix Factorization. Factorize a data matrix into two matrices
    s.t. F = | data - W*H | = | is minimal. H, and W are restricted to non-negative
    data. Uses the classicial multiplicative update rule.

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
    Applying NMF to some rather stupid data set:

    >>> import numpy as np
    >>> data = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 1.0]])
    >>> nmf_mdl = NMF(data, num_bases=2, niter=10)
    >>> nmf_mdl.factorize()

    The basis vectors are now stored in nmf_mdl.W, the coefficients in nmf_mdl.H.
    To compute coefficients for an existing set of basis vectors simply    copy W
    to nmf_mdl.W, and set compute_w to False:

    >>> data = np.array([[1.5], [1.2]])
    >>> W = np.array([[1.0, 0.0], [0.0, 1.0]])
    >>> nmf_mdl = NMF(data, num_bases=2)
    >>> nmf_mdl.W = W
    >>> nmf_mdl.factorize(niter=20, compute_w=False)

    The result is a set of coefficients nmf_mdl.H, s.t. data = W * nmf_mdl.H.
    """
    
    def __init__(self, data, num_bases=4, lamb=2.0):
        # call inherited method
        NMF.__init__(self, data, num_bases=num_bases)
        self._lamb = lamb
    
    def soft_thresholding(self, X, lamb):       
        X = np.where(np.abs(X) <= lamb, 0.0, X)
        X = np.where(X > lamb, X - lamb, X)
        X = np.where(X < -1.0*lamb, X + lamb, X)
        return X
        
    def init_w(self):
        self.W = np.random.random((self._data_dimension, self._num_bases))             
                        
    def init_h(self):
        self.H = np.random.random((self._num_bases, self._num_samples))
        self.H[:,:] = 1.0
        # normalized bases
        Wnorm = np.sqrt(np.sum(self.W**2.0, axis=0))
        self.W /= Wnorm
        
        for i in range(self.H.shape[0]):
            self.H[i,:] *= Wnorm[i]
            
        self.update_s()
        
    def update_s(self):                
        self.S = self.data - np.dot(self.W, self.H)
        self.S = self.soft_thresholding(self.S, self._lamb)
    
    def update_h(self):
        # pre init H1, and H2 (necessary for storing matrices on disk)
        H1 = np.dot(self.W.T, self.S - self.data)
        H1 = np.abs(H1) - H1
        H1 /= (2.0* np.dot(self.W.T, np.dot(self.W, self.H)))        
        self.H *= H1
  
        # adapt S
        self.update_s()
  
    def update_w(self):
        # pre init W1, and W2 (necessary for storing matrices on disk)
        W1 = np.dot(self.S - self.data, self.H.T)
        #W1 = np.dot(self.data - self.S, self.H.T)       
        W1 = np.abs(W1) - W1
        W1 /= (2.0 * (np.dot(self.W, np.dot(self.H, self.H.T))))
        self.W *= W1           
    
if __name__ == "__main__":
    import doctest
    doctest.testmod()
