#!/usr/bin/python2.6
#
# Copyright (C) Christian Thurau, 2010. 
# Licensed under the GNU General Public License (GPL). 
# http://www.gnu.org/licenses/gpl.txt
""" 
PyMF Simplex Volume Maximization [1]

    SIVM_SEARCH: class for search-SiVM

[1] C. Thurau, K. Kersting, and C. Bauckhage. Yes We Can - Simplex Volume 
Maximization for Descriptive Web-Scale Matrix Factorization. In Proc. Int. 
Conf. on Information and Knowledge Management. ACM. 2010.
"""


import scipy.sparse
import numpy as np
from scipy import inf
try:
    from scipy.misc.common import factorial
except:
    from scipy.misc import factorial

from dist import *
from vol import *
from sivm import SIVM

__all__ = ["SIVM_SEARCH"]

class SIVM_SEARCH(SIVM):
    """      
    SIVM_SEARCH(data, num_bases=4, dist_measure='l2')
    
    
    Simplex Volume Maximization. Factorize a data matrix into two matrices s.t.
    F = | data - W*H | is minimal. H is restricted to convexity. W is iteratively
    found by maximizing the volume of the resulting simplex (see [1]). A solution
    is found by employing a simple A-star like search strategy.
    
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
    >>> sivm_mdl = SIVM_SEARCH(data, num_bases=2)
    >>> sivm_mdl.factorize()
    
    The basis vectors are now stored in sivm_mdl.W, the coefficients in sivm_mdl.H. 
    To compute coefficients for an existing set of basis vectors simply copy W 
    to sivm_mdl.W, and set compute_w to False:
    
    >>> data = np.array([[1.5, 1.3], [1.2, 0.3]])
    >>> W = np.array([[1.0, 0.0], [0.0, 1.0]])
    >>> sivm_mdl = SIVM_SEARCH(data, num_bases=2)
    >>> sivm_mdl.W = W
    >>> sivm_mdl.factorize(compute_w=False)
    
    The result is a set of coefficients sivm_mdl.H, s.t. data = W * sivm_mdl.H.
    """
    
    def update_w(self):
        def h(sel,D,k):
            # compute the volume for a selection of sel columns
            # and a k-1 simplex (-> k columns have to be selected)
            mv = np.max(D)
           
            # fill the remaining distance by the maximal overall found distance
            d = np.zeros((k,k)) + mv
            for i in range(k):
                d[i,i] = 0.0
            
            for idx_i,i in enumerate(sel):
                for idx_j,j in enumerate(sel):
                    d[idx_i,idx_j] = D[i,j]
    
            return d
  
        # compute distance matrix -> required for the volume
        D = pdist(self.data, self.data)
        Openset = {} 
        
        for i in range(self._num_samples):
            # compute volume for temp selection
            d = h([i],D,self._num_bases)
            Vtmp = cmdet(d)
            Openset[tuple([i])] = Vtmp
        
        Closedset = {}
        finished = False
        self._v = []
        self.init_sivm()
        next_sel = np.array([self.select[0]])
        iter = 0

        while not finished:
            # add the current selection to closedset
            Closedset[(tuple(next_sel))] = []

            for i in range(D.shape[0]):            
                # create a temp selection
                tmp_sel = np.array(next_sel).flatten()
                tmp_sel = np.concatenate((tmp_sel, [i]),axis=0)
                tmp_sel = np.unique(tmp_sel)
                tmp_sel = list(tmp_sel)
                hkey = tuple(tmp_sel)

                if len(tmp_sel) > len(next_sel) and (
                    not Closedset.has_key(hkey)) and (
                    not Openset.has_key(hkey)):
                    
                    # compute volume for temp selection
                    d = h(tmp_sel, D, self._num_bases)
                    Vtmp = cmdet(d)
                    
                    # add to openset
                    Openset[hkey] = Vtmp

            # get next best tuple
            vmax = 0.0
            for (k,v) in Openset.iteritems():
                if v > vmax:
                    next_sel = k
                    vmax = v

            self._logger.info('Iter:' + str(iter))
            self._logger.info('Current selection:' + str(next_sel))
            self._logger.info('Current volume:' + str(vmax))
            self._v.append(vmax)

            # remove next_sel from openset
            Openset.pop(next_sel)

            if len(list(next_sel)) == self._num_bases:
                finished = True
            iter += 1

        # update some values ...
        self.select = list(next_sel)
        self.W = self.data[:, self.select] 

if __name__ == "__main__":
    import doctest  
    doctest.testmod()    
