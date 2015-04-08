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

__all__ = ["NMF"]

class NMF():
    """
    NMF(data, num_bases=4)


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
    
    # some small value
    _EPS = 10**-8
    
    def __init__(self, data, num_bases=4):
        
        def setup_logging():
            # create logger       
            self._logger = logging.getLogger("pymf")
       
            # add ch to logger
            if len(self._logger.handlers) < 1:
                # create console handler and set level to debug
                ch = logging.StreamHandler()
                ch.setLevel(logging.DEBUG)
                # create formatter
                formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        
                # add formatter to ch
                ch.setFormatter(formatter)

                self._logger.addHandler(ch)

        setup_logging()
        
        # set variables
        self.data = data       
        self._num_bases = num_bases             
      
        # initialize H and W to random values
        (self._data_dimension, self._num_samples) = self.data.shape
        

    def frobenius_norm(self):
        """ Frobenius norm (||data - WH||) of a data matrix and a low rank
        approximation given by WH

        Returns:
            frobenius norm: F = ||data - WH||
        """

        # check if W and H exist
        if hasattr(self,'H') and hasattr(self,'W') and not scipy.sparse.issparse(self.data):
            err = np.sqrt( np.sum((self.data[:,:] - np.dot(self.W, self.H))**2 ))
        else:
            err = -123456

        return err
        
    def init_w(self):
        self.W = np.random.random((self._data_dimension, self._num_bases)) 
        
    def init_h(self):
        self.H = np.random.random((self._num_bases, self._num_samples)) 
        
    def update_h(self):
            # pre init H1, and H2 (necessary for storing matrices on disk)
            H2 = np.dot(np.dot(self.W.T, self.W), self.H) + 10**-9
            self.H *= np.dot(self.W.T, self.data[:,:])
            self.H /= H2

    def update_w(self):
            # pre init W1, and W2 (necessary for storing matrices on disk)
            W2 = np.dot(np.dot(self.W, self.H), self.H.T) + 10**-9
            self.W *= np.dot(self.data[:,:], self.H.T)
            self.W /= W2

    def converged(self, i):
        derr = np.abs(self.ferr[i] - self.ferr[i-1])/self._num_samples
        if derr < self._EPS:
            return True
        else:
            return False

    def factorize(self, niter=1, show_progress=False, 
                  compute_w=True, compute_h=True, compute_err=True):
        """ Factorize s.t. WH = data
            
            Parameters
            ----------
            niter : int
                    number of iterations.
            show_progress : bool
                    print some extra information to stdout.
            compute_h : bool
                    iteratively update values for H.
            compute_w : bool
                    iteratively update values for W.
            compute_err : bool
                    compute Frobenius norm |data-WH| after each update and store
                    it to .ferr[k].
            
            Updated Values
            --------------
            .W : updated values for W.
            .H : updated values for H.
            .ferr : Frobenius norm |data-WH| for each iteration.
        """
        
        if show_progress:
            self._logger.setLevel(logging.INFO)
        else:
            self._logger.setLevel(logging.ERROR)        
        
        # create W and H if they don't already exist
        # -> any custom initialization to W,H should be done before
        if not hasattr(self,'W'):
               self.init_w()
               
        if not hasattr(self,'H'):
                self.init_h()                   

        if compute_err:
            self.ferr = np.zeros(niter)
             
        for i in xrange(niter):
            if compute_w:
                self.update_w()

            if compute_h:
                self.update_h()                                        
         
            if compute_err:                 
                self.ferr[i] = self.frobenius_norm()
                self._logger.info('Iteration ' + str(i+1) + '/' + str(niter) + 
                ' FN:' + str(self.ferr[i]))
            else:                
                self._logger.info('Iteration ' + str(i+1) + '/' + str(niter))
           

            # check if the err is not changing anymore
            if i > 1 and compute_err:
                if self.converged(i):
                    # adjust the error measure
                    self.ferr = self.ferr[:i]
                    break

if __name__ == "__main__":
    import doctest
    doctest.testmod()
