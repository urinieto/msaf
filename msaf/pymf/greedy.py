#!/usr/bin/python2.6
#
# Copyright (C) Christian Thurau, 2010. 
# Licensed under the GNU General Public License (GPL). 
# http://www.gnu.org/licenses/gpl.txt
#$Id$
""" 
PyMF GREEDY[1]

	GREEDY: class for a deterministic SVD based greedy matrix reconstruction [1].


[1] Ali Civril, Malik Magdon-Ismail. Deterministic Sparse Column Based Matrix
Reconstruction via Greedy Approximation of SVD. ISAAC'2008.
"""


import time
import scipy.sparse
import numpy as np
from svd import *
from nmf import NMF

__all__ = ["GREEDY"]

class GREEDY(NMF):
    """ 
    GREEDYVOL(data, num_bases=4, niter=100, show_progress=True, compW=True)


    Deterministic Sparse Column Based Matrix Reconstruction via Greedy 
    Approximation of SVD. Factorize a data matrix into two matrices s.t.
    F = | data - W*H | is minimal. W is iteratively selected as columns
    of data.

    Parameters
    ----------
    data : array_like, shape (_data_dimension, _num_samples)
        the input data
    num_bases: int, optional
        Number of bases to compute (column rank of W and row rank of H).
        4 (default) 
    k   : number of singular vectors for the SVD step of the algorithm
        num_bases (default)

    Attributes
    ----------
    W : "data_dimension x num_bases" matrix of basis vectors
    H : "num bases x num_samples" matrix of coefficients
    ferr : frobenius norm (after calling .factorize())       

    Example
    -------
    Applying GREEDY to some rather stupid data set:

    >>> import numpy as np
    >>> data = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 1.0]])
    >>> greedy_mdl = GREEDY(data, num_bases=2, niter=10)
    >>> greedy_mdl.factorize()

    The basis vectors are now stored in greedy_mdl.W, the coefficients in 
    greedy_mdl.H. To compute coefficients for an existing set of basis 
    vectors simply  copy W to greedy_mdl.W, and set compW to False:

    >>> data = np.array([[1.5, 1.3], [1.2, 0.3]])
    >>> W = np.array([[1.0, 0.0], [0.0, 1.0]])
    >>> greedy_mdl = GREEDY(data, num_bases=2)
    >>> greedy_mdl.W = W
    >>> greedy_mdl.factorize(compute_w=False)

    The result is a set of coefficients greedy_mdl.H, s.t. data = W * greedy_mdl.H.
    """


    def __init__(self, data, k=-1, num_bases=4):
        # call inherited method
        NMF.__init__(self, data, num_bases=num_bases)
        self._k = k
        if self._k == -1:
            self._k = num_bases

    def update_h(self):
        if scipy.sparse.issparse(self.data):
            self.H = pinv(self.W) * self.data
        else:
            self.H = np.dot(pinv(self.W), self.data)
        
    def update_w(self):
        def normalize_matrix(K):
            """ Normalize a matrix K s.t. columns have Euclidean-norm |1|
            """
            if scipy.sparse.issparse(K):                
                L = np.sqrt(np.array(K.multiply(K).sum(axis=0)))[0,:]                
                s = np.where(L > 0.0)[0]                
                L[s] = L[s]**-1
                KN = scipy.sparse.spdiags(L,0,len(L),len(L),format='csc')      
                K = K*KN
            else:
                L = np.sqrt((K**2).sum(axis=0))               
                s = np.where(L > 0.0)[0]            
                L[s] = L[s]**-1
                K = K*L                   
            return K
            
        self._t = np.zeros((self._num_bases))
        t0 = time.time()
        self.select = []       
            
        # normalize data
        A = self.data.copy()               

        svd_mdl = SVD(A, k=self._k)
        svd_mdl.factorize()
        
        if scipy.sparse.issparse(self.data):
            B = svd_mdl.U * svd_mdl.S
            B = B.tocsc()   
        else:
            B = np.dot(svd_mdl.U, svd_mdl.S)
            B = B[:, :self._num_bases]            
       
        for i in range(self._num_bases):
            A = normalize_matrix(A)  
           
            if scipy.sparse.issparse(self.data):
                T = B.transpose() * A
                T = np.array(T.multiply(T).sum(axis=0))[0,:]
                                
                # next selected column index
                T[self.select] = 0.0
                idx = np.argmax(T)
                Aidx = A[:, idx].copy()
                self.select.append(idx)
                
                # update B
                BC = Aidx.transpose() * B
                B = B - (Aidx*BC)
                
                # update A               
                AC = Aidx.transpose() * A            
                A = A - (Aidx*AC)

            else:
                T = np.dot(B.transpose(), A)
                T = np.sum(T**2.0, axis=0)
                
                # next selected column index
                T[self.select] = 0.0
                idx = np.argmax(T)
                self.select.append(idx)
                
                # update B
                BC = np.dot(B.transpose(),A[:,idx])            
                B -= np.dot(A[:,idx].reshape(-1,1), BC.reshape(1,-1))
                
                # and A
                AC = np.dot(A.transpose(),A[:,idx])
                A -= np.dot(A[:,idx].reshape(-1,1), AC.reshape(1,-1))


            # detect the next best data point
            self._logger.info('searching for next best column ...')
            self._logger.info('cur_columns: ' + str(self.select))
            self._t[i] = time.time() - t0

        # sort indices, otherwise h5py won't work
        self.W = self.data[:, np.sort(self.select)]

        # "unsort" it again to keep the correct order
        self.W = self.W[:, np.argsort(np.argsort(self.select))]       
        
if __name__ == "__main__":
    import doctest  
    doctest.testmod()
