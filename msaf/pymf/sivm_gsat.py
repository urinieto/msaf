#!/usr/bin/python2.6
#
# Copyright (C) Christian Thurau, 2010. 
# Licensed under the GNU General Public License (GPL). 
# http://www.gnu.org/licenses/gpl.txt
""" 
PyMF Simplex Volume Maximization [1]

    SIVM_GSAT: class for gsat-SiVM

[1] C. Thurau, K. Kersting, and C. Bauckhage. Yes We Can - Simplex Volume 
Maximization for Descriptive Web-Scale Matrix Factorization. In Proc. Int. 
Conf. on Information and Knowledge Management. ACM. 2010.
"""


import logging
import numpy as np
from dist import *
from vol import cmdet
from sivm import SIVM

__all__ = ["SIVM_GSAT"]

class SIVM_GSAT(SIVM):
    """      
    SIVM(data, num_bases=4, dist_measure='l2')
    
    
    Simplex Volume Maximization. Factorize a data matrix into two matrices s.t.
    F = | data - W*H | is minimal. H is restricted to convexity. W is iteratively
    found by maximizing the volume of the resulting simplex (see [1]). Can be 
    applied to data streams using the .online_update_w(vec) function which decides
    on adding data sample "vec" to the already selected basis vectors.
    
    Parameters
    ----------
    data : array_like, shape (_data_dimension, _num_samples)
        the input data
    num_bases: int, optional
        Number of bases to compute (column rank of W and row rank of H).
        4 (default)     
    dist_measure : one of 'l2' ,'cosine', 'l1', 'kl'
        Standard is 'l2' which maximizes the volume of the simplex. In contrast,
        'cosine' maximizes the volume of a cone (see [1] for details).
     init : string (default: 'fastmap')
        'fastmap' or 'origin'. Sets the method used for finding the very first 
        basis vector. 'Origin' assumes the zero vector, 'Fastmap' picks one of 
        the two vectors that have the largest pairwise distance.
    Attributes
    ----------
    W : "data_dimension x num_bases" matrix of basis vectors
    H : "num bases x num_samples" matrix of coefficients
    ferr : frobenius norm (after calling .factorize())       
    
    Example
    -------
    Applying SIVM to some rather stupid data set:
    
    >>> import numpy as np
    >>> data = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 1.0]])
    >>> sivm_mdl = SIVM_GSAT(data, num_bases=2)
    >>> sivm_mdl.factorize()
    
    The basis vectors are now stored in sivm_mdl.W, the coefficients in sivm_mdl.H. 
    To compute coefficients for an existing set of basis vectors simply    copy W 
    to sivm_mdl.W, and set compute_w to False:
    
    >>> data = np.array([[1.5, 1.3], [1.2, 0.3]])
    >>> W = np.array([[1.0, 0.0], [0.0, 1.0]])
    >>> sivm_mdl = SIVM_GSAT(data, num_bases=2)
    >>> sivm_mdl.W = W
    >>> sivm_mdl.factorize(compute_w=False)
    
    The result is a set of coefficients sivm_mdl.H, s.t. data = W * sivm_mdl.H.
    """

    def init_w(self):
        self.select = range(self._num_bases)
        self.W = self.data[:, self.select]
        
    def online_update_w(self, vec):
        # update D if it does not exist   
        k = self._num_bases
        if not hasattr(self, 'D'):
            self.D = np.zeros((k + 1, k + 1))
            self.D[:k, :k] = pdist(self.W, self.W)
            self.V = cmdet(self.D[:k, :k])
            
        tmp_d = self._distfunc(self.W, vec.reshape((-1,1)))        
        self.D[k, :-1] = tmp_d
        self.D[:-1, k] = tmp_d
       
        v = np.zeros((self._num_bases + 1))

        for i in range(self._num_bases):
                # compute volume for each combination... 
                s = np.setdiff1d(range(self._num_bases + 1), [i])
                v[i] = cmdet((self.D[s,:])[:,s])

        # select index that maximizes the volume 
        v[-1] = self.V
        s = np.argmax(v)

        if s < self._num_bases:     
            self.W[:,s] = vec
            self.D[:self._num_bases, :self._num_bases] = pdist(self.W, self.W)
            
            if not hasattr(self, '_v'):
                self._v = [self.V]
            self.V = v[s]   
            self._v.append(v[s])                
            
            self._logger.info('Volume increased:' + str(self.V))
            return True, s
            
        return False,-1
        
    def update_w(self): 
        n = np.int(np.floor(np.random.random() * self._num_samples))
        if n not in self.select:
            updated, s = self.online_update_w(self.data[:,n])
            if updated:
                self.select[s] = n    
                self._logger.info('Current selection:' + str(self.select))
                
  
    def factorize(self, show_progress=False, compute_w=True, compute_h=True,
                  compute_err=True, niter=1):
        """ Factorize s.t. WH = data
            
            Parameters
            ----------
            show_progress : bool
                    print some extra information to stdout.
            niter : int
                    number of iterations.
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
            .ferr : Frobenius norm |data-WH|.
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

                  
if __name__ == "__main__":
    import doctest  
    doctest.testmod()    
