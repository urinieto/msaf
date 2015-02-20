#!/usr/bin/python
#
# Copyright (C) Christian Thurau, 2010. 
# Licensed under the GNU General Public License (GPL). 
# http://www.gnu.org/licenses/gpl.txt
""" 
PyMF Simplex Volume Maximization [1]

    SIVM: class for SiVM

[1] C. Thurau, K. Kersting, and C. Bauckhage. Yes We Can - Simplex Volume 
Maximization for Descriptive Web-Scale Matrix Factorization. In Proc. Int. 
Conf. on Information and Knowledge Management. ACM. 2010.
"""


import scipy.sparse
import numpy as np

from dist import *
from aa import AA

__all__ = ["SIVM"]

class SIVM(AA):
    """      
    SIVM(data, num_bases=4, dist_measure='l2')
    
    
    Simplex Volume Maximization. Factorize a data matrix into two matrices s.t.
    F = | data - W*H | is minimal. H is restricted to convexity. W is iteratively
    found by maximizing the volume of the resulting simplex (see [1]).
    
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
    >>> sivm_mdl = SIVM(data, num_bases=2)
    >>> sivm_mdl.factorize()
    
    The basis vectors are now stored in sivm_mdl.W, the coefficients in sivm_mdl.H. 
    To compute coefficients for an existing set of basis vectors simply copy W 
    to sivm_mdl.W, and set compute_w to False:
    
    >>> data = np.array([[1.5, 1.3], [1.2, 0.3]])
    >>> W = np.array([[1.0, 0.0], [0.0, 1.0]])
    >>> sivm_mdl = SIVM(data, num_bases=2)
    >>> sivm_mdl.W = W
    >>> sivm_mdl.factorize(compute_w=False)
    
    The result is a set of coefficients sivm_mdl.H, s.t. data = W * sivm_mdl.H.
    """
    
    # always overwrite the default number of iterations
    # -> any value other does not make sense.
    _NITER = 1

    def __init__(self, data, num_bases=4, dist_measure='l2',  init='fastmap'):
       
        AA.__init__(self, data, num_bases=num_bases)
            
        self._dist_measure = dist_measure            
        self._init = init      
        
        # assign the correct distance function
        if self._dist_measure == 'l1':
            self._distfunc = l1_distance
                
        elif self._dist_measure == 'l2':
            self._distfunc = l2_distance
        
        elif self._dist_measure == 'cosine':                
            self._distfunc = cosine_distance
        
        elif self._dist_measure == 'abs_cosine':                
            self._distfunc = abs_cosine_distance
        
        elif self._dist_measure == 'weighted_abs_cosine':                
            self._distfunc = weighted_abs_cosine_distance
                
        elif self._dist_measure == 'kl':
            self._distfunc = kl_divergence    

  
    def _distance(self, idx):
        """ compute distances of a specific data point to all other samples"""
            
        if scipy.sparse.issparse(self.data):
            step = self.data.shape[1]
        else:    
            step = 50000    
                
        d = np.zeros((self.data.shape[1]))        
        if idx == -1:
            # set vec to origin if idx=-1
            vec = np.zeros((self.data.shape[0], 1))
            if scipy.sparse.issparse(self.data):
                vec = scipy.sparse.csc_matrix(vec)
        else:
            vec = self.data[:, idx:idx+1]    
            
        self._logger.info('compute distance to node ' + str(idx))
                                                
        # slice data into smaller chunks
        for idx_start in range(0, self.data.shape[1], step):                    
            if idx_start + step > self.data.shape[1]:
                idx_end = self.data.shape[1]
            else:
                idx_end = idx_start + step

            d[idx_start:idx_end] = self._distfunc(
                self.data[:,idx_start:idx_end], vec)
            self._logger.info('completed:' + 
                str(idx_end/(self.data.shape[1]/100.0)) + "%")    
        return d
       
    def init_h(self):
        self.H = np.zeros((self._num_bases, self._num_samples))
        
    def init_w(self):
        self.W = np.zeros((self._data_dimension, self._num_bases))
        
    def init_sivm(self):
        self.select = []
        if self._init == 'fastmap':
            # Fastmap like initialization
            # set the starting index for fastmap initialization        
            cur_p = 0        
            
            # after 3 iterations the first "real" index is found
            for i in range(3):                                
                d = self._distance(cur_p)                        
                cur_p = np.argmax(d)
                
            # store maximal found distance -> later used for "a" (->update_w) 
            self._maxd = np.max(d)                        
            self.select.append(cur_p)

        elif self._init == 'origin':
            # set first vertex to origin
            cur_p = -1
            d = self._distance(cur_p)
            self._maxd = np.max(d)
            self.select.append(cur_p)         
        
    def update_w(self): 
        """ compute new W """        
        EPS = 10**-8
        self.init_sivm()       
        
        # initialize some of the recursively updated distance measures ....
        d_square = np.zeros((self.data.shape[1]))
        d_sum = np.zeros((self.data.shape[1]))
        d_i_times_d_j = np.zeros((self.data.shape[1]))
        distiter = np.zeros((self.data.shape[1]))
        a = np.log(self._maxd) 
        a_inc = a.copy()
        
        for l in range(1, self._num_bases):
            d = self._distance(self.select[l-1])
            
            # take the log of d (sually more stable that d)
            d = np.log(d + EPS)            
            
            d_i_times_d_j += d * d_sum
            d_sum += d
            d_square += d**2
            distiter = d_i_times_d_j + a*d_sum - (l/2.0) * d_square                   
            
            # detect the next best data point                      
            self.select.append(np.argmax(distiter))                       
        
            self._logger.info('cur_nodes: ' + str(self.select))

        # sort indices, otherwise h5py won't work
        self.W = self.data[:, np.sort(self.select)]
            
        # "unsort" it again to keep the correct order
        self.W = self.W[:, np.argsort(np.argsort(self.select))]
    
    def factorize(self, show_progress=False, compute_w=True, compute_h=True,
                  compute_err=True, niter=1):
        """ Factorize s.t. WH = data
            
            Parameters
            ----------
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
            .ferr : Frobenius norm |data-WH|.
        """
        
        AA.factorize(self, niter=1, show_progress=show_progress, 
                  compute_w=compute_w, compute_h=compute_h, 
                  compute_err=compute_err)
             
if __name__ == "__main__":
    import doctest  
    doctest.testmod()    
