#!/usr/bin/python
#
# Copyright (C) Christian Thurau, 2010.
# Licensed under the GNU General Public License (GPL).
# http://www.gnu.org/licenses/gpl.txt
#$Id$
"""
PyMF Non-negative Double Singular Value Decompositions.

    NNDSVD: Class for Non-negative Double Singular Value Decompositions [1]

[1] C. Boutsidis and E. Gallopoulos (2008), SVD based initialization: A head
start for nonnegative matrix factorization, Pattern Recognition, 41, 1350-1362
"""


import numpy as np

from nmf import NMF
from svd import SVD

__all__ = ["NNDSVD"]

class NNDSVD(NMF):
    """
    NNDSVD(data, num_bases=4)


    Non-negative Double Singular Value Decompositions. Factorize a data 
    matrix into two matrices s.t. F = | data - W*H | = | is minimal. H, and 
    W are restricted to non-negative data. NNDSVD is primarily used for
    initializing NMF. 

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
    Applying NNDSVD to some rather stupid data set:

    >>> import numpy as np
    >>> data = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 1.0]])
    >>> nndsvd_mdl = NNDSVD(data, num_bases=2)
    >>> nndsvd_mdl.factorize()

    The basis vectors are now stored in nndsvd_mdl.W, the coefficients in 
    nndsvd_mdl.H. To initialize NMF with nndsvd_mdl.W, nndsvd_mdl.H 
    simply copy W to nmf_mdl.W and H to nmf_mdl.H:

    >>> data = np.array([[1.5], [1.2]])
    >>> W = np.array([[1.0, 0.0], [0.0, 1.0]])
    >>> nmf_mdl = NMF(data, num_bases=2)
    >>> nmf_mdl.W = nndsvd_mdl.W
    >>> nmf_mdl.H = nndsvd_mdl.H
    >>> nmf_mdl.factorize(niter=20)

    The result is a set of (more optimal) coefficients nmf_mdl.H, nmf_mdl.W.
    """                
    def init_w(self):
        self.W = np.zeros((self._data_dimension, self._num_bases))
                       
    def init_h(self):
        self.H = np.zeros((self._num_bases, self._num_samples))
        
    def update_h(self):
        pass

    def update_w(self):
        svd_mdl = SVD(self.data)
        svd_mdl.factorize()
        
        U, S, V = svd_mdl.U, svd_mdl.S, svd_mdl.V    
        
        # The first left singular vector is nonnegative
        # (abs is only used as values could be all negative)
        self.W[:,0] = np.sqrt(S[0,0]) * np.abs(U[:,0])
        
        #The first right singular vector is nonnegative
        self.H[0,:] = np.sqrt(S[0,0]) * np.abs(V[0,:].T)

        for i in range(1,self._num_bases):
            # Form the rank one factor
            Tmp = np.dot(U[:,i:i+1]*S[i,i], V[i:i+1,:])          
            
            # zero out the negative elements
            Tmp = np.where(Tmp < 0, 0.0, Tmp)
            
            # Apply 2nd SVD
            svd_mdl_2 = SVD(Tmp)
            svd_mdl_2.factorize()
            u, s, v = svd_mdl_2.U, svd_mdl_2.S, svd_mdl_2.V
            
            # The first left singular vector is nonnegative
            self.W[:,i] = np.sqrt(s[0,0]) * np.abs(u[:,0]) 
            
            #The first right singular vector is nonnegative
            self.H[i,:] = np.sqrt(s[0,0]) * np.abs(v[0,:].T) 
        
    def factorize(self, niter=1, show_progress=False, 
                  compute_w=True, compute_h=True, compute_err=True):
                      
        # enforce certain default values, otherwise it won't work
        NMF.factorize(self, niter=1, show_progress=show_progress, 
                  compute_w=True, compute_h=True, compute_err=compute_err)

if __name__ == "__main__":
    import doctest
    doctest.testmod()
