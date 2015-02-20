#!/usr/bin/python
#
# Copyright (C) Christian Thurau, 2010. 
# Licensed under the GNU General Public License (GPL). 
# http://www.gnu.org/licenses/gpl.txt
"""      
PyMF Convex Matrix Factorization [1]

    CNMF(NMF) : Class for convex matrix factorization
    
[1] Ding, C., Li, T. and Jordan, M.. Convex and Semi-Nonnegative Matrix Factorizations.
IEEE Trans. on Pattern Analysis and Machine Intelligence 32(1), 45-55.
"""


import numpy as np
import logging
from nmf import NMF
from kmeans import Kmeans


__all__ = ["CNMF"]

class CNMF(NMF):
    """      
    CNMF(data, num_bases=4)
    
    
    Convex NMF. Factorize a data matrix into two matrices s.t.
    F = | data - W*H | = | data - data*beta*H| is minimal. H and beta
    are restricted to convexity (beta >=0, sum(beta, axis=1) = [1 .. 1]).    
    
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
    Applying CNMF to some rather stupid data set:
    
    >>> import numpy as np
    >>> from cnmf import CNMF
    >>> data = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 1.0]])
    >>> cnmf_mdl = CNMF(data, num_bases=2)
    >>> cnmf_mdl.factorize(niter=10)
    
    The basis vectors are now stored in cnmf_mdl.W, the coefficients in cnmf_mdl.H. 
    To compute coefficients for an existing set of basis vectors simply    copy W 
    to cnmf_mdl.W, and set compute_w to False:
    
    >>> data = np.array([[1.5, 1.3], [1.2, 0.3]])
    >>> W = [[1.0, 0.0], [0.0, 1.0]]
    >>> cnmf_mdl = CNMF(data, num_bases=2)
    >>> cnmf_mdl.W = W
    >>> cnmf_mdl.factorize(compute_w=False, niter=1)
    
    The result is a set of coefficients acnmf_mdl.H, s.t. data = W * cnmf_mdl.H.
    """

    # see .factorize() for the update of W and H
    # -> proper decoupling of W/H not possible ...
    def update_w(self):
        pass        
        
    def update_h(self):
        pass
    
    def init_h(self):
        if not hasattr(self, 'H'):
            # init basic matrices       
            self.H = np.zeros((self._num_bases, self._num_samples))
            
            # initialize using k-means
            km = Kmeans(self.data[:,:], num_bases=self._num_bases)        
            km.factorize(niter=10)
            assign = km.assigned
    
            num_i = np.zeros(self._num_bases)
            for i in range(self._num_bases):
                num_i[i] = len(np.where(assign == i)[0])
    
            self.H.T[range(len(assign)), assign] = 1.0                
            self.H += 0.2*np.ones((self._num_bases, self._num_samples))
        
        if not hasattr(self, 'G'):
            self.G = np.zeros((self._num_samples, self._num_bases))
            
            self.G[range(len(assign)), assign] = 1.0 
            self.G += 0.01                        
            self.G /= np.tile(np.reshape(num_i[assign],(-1,1)), self.G.shape[1])
            
        if not hasattr(self,'W'):
            self.W = np.dot(self.data[:,:], self.G)
    
    def init_w(self):
        pass
    
    def factorize(self, niter=10, compute_w=True, compute_h=True, 
                  compute_err=True, show_progress=False):
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
        
        if not hasattr(self,'W'):
               self.init_w()
               
        if not hasattr(self,'H'):
                self.init_h()              
        
        def separate_positive(m):
            return (np.abs(m) + m)/2.0 
        
        def separate_negative(m):
            return (np.abs(m) - m)/2.0 
            
        if show_progress:
            self._logger.setLevel(logging.INFO)
        else:
            self._logger.setLevel(logging.ERROR)                 
            
        XtX = np.dot(self.data[:,:].T, self.data[:,:])
        XtX_pos = separate_positive(XtX)
        XtX_neg = separate_negative(XtX)
        
        self.ferr = np.zeros(niter)
        # iterate over W and H
        
        for i in xrange(niter):
            # update H
            XtX_neg_x_W = np.dot(XtX_neg, self.G)
            XtX_pos_x_W = np.dot(XtX_pos, self.G)
            
            if compute_h:
                H_x_WT = np.dot(self.H.T, self.G.T)                                
                ha = XtX_pos_x_W + np.dot(H_x_WT, XtX_neg_x_W)
                hb = XtX_neg_x_W + np.dot(H_x_WT, XtX_pos_x_W) + 10**-9
                self.H = (self.H.T*np.sqrt(ha/hb)).T
        
            # update W            
            if compute_w:
                HT_x_H = np.dot(self.H, self.H.T)
                wa = np.dot(XtX_pos, self.H.T) + np.dot(XtX_neg_x_W, HT_x_H)
                wb = np.dot(XtX_neg, self.H.T) + np.dot(XtX_pos_x_W, HT_x_H) + 10**-9
            
                self.G *= np.sqrt(wa/wb)
                self.W = np.dot(self.data[:,:], self.G)
                                
            if compute_err:                 
                self.ferr[i] = self.frobenius_norm()
                self._logger.info('Iteration ' + str(i+1) + '/' + str(niter) + 
                ' FN:' + str(self.ferr[i]))
            else:                
                self._logger.info('Iteration ' + str(i+1) + '/' + str(niter))

            if i > 1 and compute_err:
                if self.converged(i):                    
                    self.ferr = self.ferr[:i]                    
                    break

if __name__ == "__main__":
    import doctest  
    doctest.testmod()    
