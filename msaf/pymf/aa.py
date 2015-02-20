#!/usr/bin/python
#
# Copyright (C) Christian Thurau, 2010.
# Licensed under the GNU General Public License (GPL).
# http://www.gnu.org/licenses/gpl.txt
"""
PyMF Archetypal Analysis [1]

    AA: class for Archetypal Analysis

[1] Cutler, A. Breiman, L. (1994), "Archetypal Analysis", Technometrics 36(4), 
338-347.
"""


import numpy as np
from dist import vq
from cvxopt import solvers, base

from svd import pinv
from nmf import NMF

__all__ = ["AA"]

class AA(NMF):
    """
    AA(data, num_bases=4)

    Archetypal Analysis. Factorize a data matrix into two matrices s.t.
    F = | data - W*H | = | data - data*beta*H| is minimal. H and beta
    are restricted to convexity (beta >=0, sum(beta, axis=1) = [1 .. 1]).
    Factorization is solved via an alternating least squares optimization
    using the quadratic programming solver from cvxopt.

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
    beta : "num_bases x num_samples" matrix of basis vector coefficients
        (for constructing W s.t. W = beta * data.T )
    ferr : frobenius norm (after calling .factorize()) 
        
    Example
    -------
    Applying AA to some rather stupid data set:

    >>> import numpy as np
    >>> from aa import AA
    >>> data = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 1.0]])
    
    Use 2 basis vectors -> W shape(data_dimension, 2).
    
    >>> aa_mdl = AA(data, num_bases=2)

    Set number of iterations to 5 and start computing the factorization.    
    
    >>> aa_mdl.factorize(niter=5)

    The basis vectors are now stored in aa_mdl.W, the coefficients in aa_mdl.H.
    To compute coefficients for an existing set of basis vectors simply copy W
    to aa_mdl.W, and set compute_w to False:

    >>> data = np.array([[1.5], [1.2]])
    >>> W = np.array([[1.0, 0.0], [0.0, 1.0]])
    >>> aa_mdl = AA(data, num_bases=2)
    >>> aa_mdl.W = W
    >>> aa_mdl.factorize(niter=5, compute_w=False)

    The result is a set of coefficients aa_mdl.H, s.t. data = W * aa_mdl.H.
    """
    # set cvxopt options
    solvers.options['show_progress'] = False

    def init_h(self):
        self.H = np.random.random((self._num_bases, self._num_samples))     
        self.H /= self.H.sum(axis=0)
            
    def init_w(self):
        self.beta = np.random.random((self._num_bases, self._num_samples))
        self.beta /= self.beta.sum(axis=0)
        self.W = np.dot(self.beta, self.data.T).T            
        self.W = np.random.random((self._data_dimension, self._num_bases))        
        
    def update_h(self):
        """ alternating least squares step, update H under the convexity
        constraint """
        def update_single_h(i):
            """ compute single H[:,i] """
            # optimize alpha using qp solver from cvxopt
            FA = base.matrix(np.float64(np.dot(-self.W.T, self.data[:,i])))
            al = solvers.qp(HA, FA, INQa, INQb, EQa, EQb)
            self.H[:,i] = np.array(al['x']).reshape((1, self._num_bases))

        EQb = base.matrix(1.0, (1,1))
        # float64 required for cvxopt
        HA = base.matrix(np.float64(np.dot(self.W.T, self.W)))
        INQa = base.matrix(-np.eye(self._num_bases))
        INQb = base.matrix(0.0, (self._num_bases,1))
        EQa = base.matrix(1.0, (1, self._num_bases))
        
        for i in xrange(self._num_samples):
            update_single_h(i)        

    def update_w(self):
        """ alternating least squares step, update W under the convexity
        constraint """
        def update_single_w(i):
            """ compute single W[:,i] """
            # optimize beta     using qp solver from cvxopt
            FB = base.matrix(np.float64(np.dot(-self.data.T, W_hat[:,i])))
            be = solvers.qp(HB, FB, INQa, INQb, EQa, EQb)
            self.beta[i,:] = np.array(be['x']).reshape((1, self._num_samples))

        # float64 required for cvxopt
        HB = base.matrix(np.float64(np.dot(self.data[:,:].T, self.data[:,:])))
        EQb = base.matrix(1.0, (1, 1))
        W_hat = np.dot(self.data, pinv(self.H))
        INQa = base.matrix(-np.eye(self._num_samples))
        INQb = base.matrix(0.0, (self._num_samples, 1))
        EQa = base.matrix(1.0, (1, self._num_samples))

        for i in xrange(self._num_bases):
            update_single_w(i)            

        self.W = np.dot(self.beta, self.data.T).T

if __name__ == "__main__":
    import doctest
    doctest.testmod()
