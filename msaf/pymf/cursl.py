#!/usr/bin/python
#
# Copyright (C) Christian Thurau, 2010. 
# Licensed under the GNU General Public License (GPL). 
# http://www.gnu.org/licenses/gpl.txt
#$Id$
"""
PyMF CUR Decomposition [1]

    CURSL(SVD) : Class for CUR Decomposition (uses statistical leverage based sampling)

[1] Drineas, P., Kannan, R. and Mahoney, M. (2006), 'Fast Monte Carlo Algorithms III: Computing 
a Compressed Approixmate Matrix Decomposition', SIAM J. Computing 36(1), 184-206.
"""


import numpy as np
import scipy.sparse

from svd import pinv, SVD
from cmd import CMD

__all__ = ["CURSL"]

class CURSL(CMD):
    """      
    CURSL(data,  data, rrank=0, crank=0)
        
    CUR/CMD Decomposition. Factorize a data matrix into three matrices s.t.
    F = | data - USV| is minimal. CURSL randomly selects rows and columns from
    data for building U and V, respectively. The importance sampling is based
    on a statistical leverage score from the top-k singular vectors (k is
    currently set to 4/5*rrank and 4/5*crank). 
    
    Parameters
    ----------
    data : array_like [data_dimension x num_samples]
        the input data
    rrank: int, optional 
        Number of rows to sample from data.
        4 (default)
    crank: int, optional
        Number of columns to sample from data.
        4 (default)
    show_progress: bool, optional
        Print some extra information
        False (default)    
    
    Attributes
    ----------
        U,S,V : submatrices s.t. data = USV  (or _C _U _R)      
    
    Example
    -------
    >>> import numpy as np
    >>> from cur import CUR
    >>> data = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 1.0]])
    >>> cur_mdl = CURSL(data, show_progress=False, rrank=1, crank=2)    
    >>> cur_mdl.factorize()
    """
    
    def __init__(self, data, k=-1, rrank=0, crank=0):
        SVD.__init__(self, data, k=k, rrank=rrank, crank=rrank)
        
    def sample_probability(self):
        def comp_prob(d, k):           
            # compute statistical leverage score     
            c = np.round(k - k/5.0)
        
            svd_mdl = SVD(d, k=c)
            svd_mdl.factorize()
            
            if scipy.sparse.issparse(self.data):
                A = svd_mdl.V.multiply(svd_mdl.V)           
                ## Rule 1
                pcol = np.array(A.sum(axis=0)/k)                                    
            else:
                A = svd_mdl.V[:k,:]**2.0         
                ## Rule 1
                pcol = A.sum(axis=0)/k            
                
            #c = k * np.log(k/ (self._eps**2.0))
            #pcol = c * pcol.reshape((-1,1)) 
            pcol /= np.sum(pcol)                     
            return pcol
            
        pcol = comp_prob(self.data, self._rrank)
        prow = comp_prob(self.data.transpose(), self._crank)        
    
        
        return (prow.reshape(-1,1), pcol.reshape(-1,1))  
