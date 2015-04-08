#!/usr/bin/python
#
# Copyright (C) Christian Thurau, 2010. 
# Licensed under the GNU General Public License (GPL). 
# http://www.gnu.org/licenses/gpl.txt
""" 
PyMF LAESA
"""


import scipy.sparse
import numpy as np

from dist import *
from sivm import SIVM

__all__ = ["LAESA"]

class LAESA(SIVM):
    """      
    LAESA(data, num_bases=4)
    
    
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
    
    Attributes
    ----------
    W : "data_dimension x num_bases" matrix of basis vectors
    H : "num bases x num_samples" matrix of coefficients
    ferr : frobenius norm (after calling .factorize())        
    
    Example
    -------
    Applying LAESA to some rather stupid data set:
    
    >>> import numpy as np
    >>> data = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 1.0]])
    >>> laesa_mdl = LAESA(data, num_bases=2)
    >>> laesa_mdl.factorize()
    
    The basis vectors are now stored in laesa_mdl.W, the coefficients in laesa_mdl.H. 
    To compute coefficients for an existing set of basis vectors simply    copy W 
    to laesa_mdl.W, and set compute_w to False:
    
    >>> data = np.array([[1.5, 1.3], [1.2, 0.3]])
    >>> W = np.array([[1.0, 0.0], [0.0, 1.0]])
    >>> laesa_mdl = LAESA(data, num_bases=2)
    >>> laesa_mdl.W = W
    >>> laesa_mdl.factorize(niter=1, compute_w=False)
    
    The result is a set of coefficients laesa_mdl.H, s.t. data = W * laesa_mdl.H.
    """
    def update_w(self):    
        # initialize some of the recursively updated distance measures     
        self.init_sivm()
        distiter = self._distance(self.select[-1])                
        
        for l in range(self._num_bases-1):                                        
            d = self._distance(self.select[-1])                                
        
            # replace distances in distiter
            distiter = np.where(d<distiter, d, distiter)
            
            # detect the next best data point                        
            self.select.append(np.argmax(distiter))
            self._logger.info('cur_nodes: ' + str(self.select))

        # sort indices, otherwise h5py won't work
        self.W = self.data[:, np.sort(self.select)]
        
        # but "unsort" it again to keep the correct order
        self.W = self.W[:, np.argsort(np.argsort(self.select))]    
                           
                   
if __name__ == "__main__":
    import doctest  
    doctest.testmod()    
