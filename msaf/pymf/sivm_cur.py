#!/usr/bin/python
#
# Copyright (C) Christian Thurau, 2010. 
# Licensed under the GNU General Public License (GPL). 
# http://www.gnu.org/licenses/gpl.txt
"""  
PyMF Simplex Volume Maximization for CUR [1]

    SIVMCUR: class for SiVM-CUR

[1] C. Thurau, K. Kersting, and C. Bauckhage. Yes We Can - Simplex Volume 
Maximization for Descriptive Web-Scale Matrix Factorization. In Proc. Int. 
Conf. on Information and Knowledge Management. ACM. 2010.
"""


import numpy as np
import scipy
from sivm import SIVM
from cur import CUR

__all__ = ["SIVM_CUR"]

class SIVM_CUR(CUR):
    '''
    SIVM_CUR(data, num_bases=4, dist_measure='l2')
    
    Simplex Volume based CUR Decomposition. Factorize a data matrix into three 
    matrices s.t. F = | data - USV| is minimal. Unlike CUR, SIVMCUR selects the
    rows and columns using SIVM, i.e. it tries to maximize the volume of the
    enclosed simplex.
    
    Parameters
    ----------
    data : array_like [data_dimension x num_samples]
        the input data
    rrank: int, optional 
        Number of rows to sample from data.
        4 (default)crank
    crank: int, optional
        Number of columns to sample from data.
        4 (default)   
    dist_measure: string, optional
        The distance measure for finding the next best candidate that 
        maximizes the simplex volume ['l2','l1','cosine','sparse_graph_l2']
        'l2' (default)
    
    Attributes
    ----------
        U,S,V : submatrices s.t. data = USV        
    
    Example
    -------
    >>> import numpy as np
    >>> data = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 1.0]])
    >>> sivmcur_mdl = SIVM_CUR(data, show_progress=False, rrank=1, crank=2)    
    >>> sivmcur_mdl.factorize()
    '''
    
    def __init__(self, data, k=-1, rrank=0, crank=0, dist_measure='l2', init='origin'):
        CUR.__init__(self, data, k=k, rrank=rrank, crank=rrank)
        self._dist_measure = dist_measure
        self.init = init

    def sample(self, A, c):
        # for optimizing the volume of the submatrix, set init to 'origin' (otherwise the volume of
        # the ordinary simplex would be optimized) 
        sivm_mdl = SIVM(A, num_bases=c, dist_measure=self._dist_measure,  
                            init=self.init)                        
        sivm_mdl.factorize(show_progress=False, compute_w=True, niter=1,
                           compute_h=False, compute_err=False)
        
        return sivm_mdl.select    
            
            
    def factorize(self):
        """ Factorize s.t. CUR = data
            
            Updated Values
            --------------
            .C : updated values for C.
            .U : updated values for U.
            .R : updated values for R.          
        """            
        # sample row and column indices that maximize the volume of the submatrix
        self._rid = self.sample(self.data.transpose(), self._rrank)
        self._cid = self.sample(self.data, self._crank)
                        
        self._rcnt = np.ones(len(self._rid))
        self._ccnt = np.ones(len(self._cid))    
                                    
        self.computeUCR()


if __name__ == "__main__":
    import doctest  
    doctest.testmod()                    
