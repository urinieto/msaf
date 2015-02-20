#!/usr/bin/python2.6
#
# Copyright (C) Christian Thurau, 2010. 
# Licensed under the GNU General Public License (GPL). 
# http://www.gnu.org/licenses/gpl.txt
#$Id$
"""  
PyMF CUR-like Sparse Column Based Matrix Reconstruction via Greedy Approximation[1]

	GREEDYCUR: class for CUR-like decompositions using the GREEDY[2] algorithm.

[1] Drineas, P., Kannan, R. and Mahoney, M. (2006), 'Fast Monte Carlo Algorithms III: 
Computing a Compressed Approixmate Matrix Decomposition', SIAM J. Computing 36(1), 184-206.
[2] Ali Civril, Malik Magdon-Ismail. Deterministic Sparse Column Based Matrix
Reconstruction via Greedy Approximation of SVD. ISAAC'2008.
"""


import numpy as np
from greedy import GREEDY
from cur import CUR

__all__ = ["GREEDYCUR"]

class GREEDYCUR(CUR):
    '''
    GREEDYCUR(data,  data, k=-1, rrank=0, crank=0)

    GREEDY-CUR Decomposition. Factorize a data matrix into three matrices s.t. 
    F = | data - USV| is minimal. Unlike CUR, GREEDYCUR selects the rows 
    and columns using GREEDY, i.e. it tries to find rows/columns that are close
    to SVD-based solutions.

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
        U,S,V : submatrices s.t. data = USV        
    
    Example
    -------
    >>> import numpy as np
    >>> from greedycur import GREEDYCUR
    >>> data = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 1.0]])
    >>> cur_mdl = GREEDYCUR(data, show_progress=False, rrank=1, crank=2)    
    >>> cur_mdl.factorize()
    """
    '''

    def sample(self, A, c):
        # set k to a value lower than the number of bases, usually
        # gives better results.
        k = np.round(c - c/5.0)
        greedy_mdl = GREEDY(A, k=k, num_bases=c)
        greedy_mdl.factorize(compute_h=False, compute_err=False, niter=1)        
        return greedy_mdl.select
            
            
    def factorize(self):
        # sample row and column indices that maximize the volume of the submatrix
        self._rid = self.sample(self.data.transpose(), self._rrank)
        self._cid = self.sample(self.data, self._crank)
        self._rcnt = np.ones(len(self._rid))
        self._ccnt = np.ones(len(self._cid))
                                    
        self.computeUCR()


if __name__ == "__main__":
    import doctest  
    doctest.testmod()
