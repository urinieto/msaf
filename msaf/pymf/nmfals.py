#!/usr/bin/python
#
# Copyright (C) Christian Thurau, 2010. 
# Licensed under the GNU General Public License (GPL). 
# http://www.gnu.org/licenses/gpl.txt
"""
PyMF Non-negative Matrix Factorization.

    NMFALS: Class for Non-negative Matrix Factorization using alternating least
            squares optimization (requires cvxopt)

[1] Lee, D. D. and Seung, H. S. (1999), Learning the Parts of Objects by Non-negative 
Matrix Factorization, Nature 401(6755), 788-799.
"""



import numpy as np
from cvxopt import solvers, base
from nmf import NMF

__all__ = ["NMFALS"]

class NMFALS(NMF):
    """      
    NMF(data, num_bases=4)
    
    
    Non-negative Matrix Factorization. Factorize a data matrix into two matrices 
    s.t. F = | data - W*H | = | is minimal. H, and W are restricted to non-negative
    data. Uses the an alternating least squares procedure (quite slow for larger
    data sets)
    
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
    >>> nmf_mdl = NMFALS(data, num_bases=2)
    >>> nmf_mdl.factorize(niter=10)
    
    The basis vectors are now stored in nmf_mdl.W, the coefficients in nmf_mdl.H. 
    To compute coefficients for an existing set of basis vectors simply    copy W 
    to nmf_mdl.W, and set compute_w to False:
    
    >>> data = np.array([[1.5], [1.2]])
    >>> W = np.array([[1.0, 0.0], [0.0, 1.0]])
    >>> nmf_mdl = NMFALS(data, num_bases=2)
    >>> nmf_mdl.W = W
    >>> nmf_mdl.factorize(niter=1, compute_w=False)
    
    The result is a set of coefficients nmf_mdl.H, s.t. data = W * nmf_mdl.H.
    """
  
    def update_h(self):
        def updatesingleH(i):
            # optimize alpha using qp solver from cvxopt
            FA = base.matrix(np.float64(np.dot(-self.W.T, self.data[:,i])))
            al = solvers.qp(HA, FA, INQa, INQb)
            self.H[:,i] = np.array(al['x']).reshape((1,-1))
                                                                
        # float64 required for cvxopt
        HA = base.matrix(np.float64(np.dot(self.W.T, self.W)))            
        INQa = base.matrix(-np.eye(self._num_bases))
        INQb = base.matrix(0.0, (self._num_bases,1))            
    
        map(updatesingleH, xrange(self._num_samples))                        
            
                
    def update_w(self):
        def updatesingleW(i):
        # optimize alpha using qp solver from cvxopt
            FA = base.matrix(np.float64(np.dot(-self.H, self.data[i,:].T)))
            al = solvers.qp(HA, FA, INQa, INQb)                
            self.W[i,:] = np.array(al['x']).reshape((1,-1))            
                                
        # float64 required for cvxopt
        HA = base.matrix(np.float64(np.dot(self.H, self.H.T)))                    
        INQa = base.matrix(-np.eye(self._num_bases))
        INQb = base.matrix(0.0, (self._num_bases,1))            

        map(updatesingleW, xrange(self._data_dimension))
