#!/usr/bin/python
#
# Copyright (C) Christian Thurau, 2010. 
# Licensed under the GNU General Public License (GPL). 
# http://www.gnu.org/licenses/gpl.txt
"""  
PyMF Singular Value Decomposition.

    SVD : Class for Singular Value Decomposition
    pinv() : Compute the pseudoinverse of a Matrix
     
"""



from numpy.linalg import eigh
import scipy.sparse

try:
    import scipy.sparse.linalg.eigen.arpack as linalg
except (ImportError, AttributeError):
    import scipy.sparse.linalg as linalg


import numpy as np

def pinv(A, k=-1, eps=10**-8):    
    # Compute Pseudoinverse of a matrix
    # calculate SVD
    svd_mdl =  SVD(A, k=k)
    svd_mdl.factorize()
    
    S = svd_mdl.S
    Sdiag = S.diagonal()
    Sdiag = np.where(Sdiag >eps, 1.0/Sdiag, 0.0)
    
    for i in range(S.shape[0]):
        S[i,i] = Sdiag[i]
            
    if scipy.sparse.issparse(A):            
        A_p = svd_mdl.V.T * (S *  svd_mdl.U.T)
    else:    
        A_p = np.dot(svd_mdl.V.T, np.core.multiply(np.diag(S)[:,np.newaxis], svd_mdl.U.T))

    return A_p


class SVD():    
    """      
    SVD(data, show_progress=False)
    
    
    Singular Value Decomposition. Factorize a data matrix into three matrices s.t.
    F = | data - USV| is minimal. U and V correspond to eigenvectors of the matrices
    data*data.T and data.T*data.
    
    Parameters
    ----------
    data : array_like [data_dimension x num_samples]
        the input data
    
    Attributes
    ----------
        U,S,V : submatrices s.t. data = USV                
    
    Example
    -------
    >>> import numpy as np
    >>> data = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 1.0]])
    >>> svd_mdl = SVD(data, show_progress=False)    
    >>> svd_mdl.factorize()
    """
    
    _EPS=10**-8
    
    def __init__(self, data, k=-1, rrank=0, crank=0):
        self.data = data
        (self._rows, self._cols) = self.data.shape
        if rrank > 0:
            self._rrank = rrank
        else:
            self._rrank = self._rows
            
        if crank > 0:            
            self._crank = crank
        else:
            self._crank = self._cols
        
        # set the rank to either rrank or crank
        self._k = k
    
    def frobenius_norm(self):
        """ Frobenius norm (||data - USV||) for a data matrix and a low rank
        approximation given by SVH using rank k for U and V
        
        Returns:
            frobenius norm: F = ||data - USV||
        """    
        if scipy.sparse.issparse(self.data):
            err = self.data - self.U*self.S*self.V    
            err = err.multiply(err)
            err = np.sqrt(err.sum())
        else:                
            err = self.data[:,:] - np.dot(np.dot(self.U, self.S), self.V)
            err = np.sqrt(np.sum(err**2))
                            
        return err
        
    
    def factorize(self):    
        def _right_svd():            
            AA = np.dot(self.data[:,:], self.data[:,:].T)
            values, u_vectors = eigh(AA)            
                
            # get rid of too low eigenvalues
            u_vectors = u_vectors[:, values > self._EPS] 
            values = values[values > self._EPS]
                            
            # sort eigenvectors according to largest value
            idx = np.argsort(values)
            values = values[idx[::-1]]

            # argsort sorts in ascending order -> access is backwards
            self.U = u_vectors[:,idx[::-1]]
            
            # compute S
            self.S = np.diag(np.sqrt(values))
            
            # and the inverse of it
            S_inv = np.diag(np.sqrt(values)**-1)
                    
            # compute V from it
            self.V = np.dot(S_inv, np.dot(self.U[:,:].T, self.data[:,:]))    
            
        
        def _left_svd():
            AA = np.dot(self.data[:,:].T, self.data[:,:])
            values, v_vectors = eigh(AA)    
        
            # get rid of too low eigenvalues
            v_vectors = v_vectors[:, values > self._EPS] 
            values = values[values > self._EPS]
            
            # sort eigenvectors according to largest value
            # argsort sorts in ascending order -> access is backwards
            idx = np.argsort(values)[::-1]
            values = values[idx]
            
            # compute S
            self.S= np.diag(np.sqrt(values))
            
            # and the inverse of it
            S_inv = np.diag(1.0/np.sqrt(values))    
                        
            Vtmp = v_vectors[:,idx]
            
            self.U = np.dot(np.dot(self.data[:,:], Vtmp), S_inv)                
            self.V = Vtmp.T
    
        def _sparse_right_svd():
            ## for some reasons arpack does not allow computation of rank(A) eigenvectors (??)    #
            AA = self.data*self.data.transpose()    
            if self.data.shape[0] > 1:                    
                # do not compute full rank if desired
                if self._k > 0 and self._k < self.data.shape[0]-1:
                    k = self._k
                else:
                    k = self.data.shape[0]-1

                values, u_vectors = linalg.eigen_symmetric(AA,k=k)
            else:                
                values, u_vectors = eigh(AA.todense())
                    
            # get rid of too low eigenvalues
            u_vectors = u_vectors[:, values > self._EPS] 
            values = values[values > self._EPS]
            
            # sort eigenvectors according to largest value
            idx = np.argsort(values)
            values = values[idx[::-1]]                        
            
            # argsort sorts in ascending order -> access is backwards            
            self.U = scipy.sparse.csc_matrix(u_vectors[:,idx[::-1]])
                    
            # compute S
            self.S = scipy.sparse.csc_matrix(np.diag(np.sqrt(values)))
            
            # and the inverse of it
            S_inv = scipy.sparse.csc_matrix(np.diag(1.0/np.sqrt(values)))            
                    
            # compute V from it
            self.V = self.U.transpose() * self.data
            self.V = S_inv * self.V
    
        def _sparse_left_svd():        
            # for some reasons arpack does not allow computation of rank(A) eigenvectors (??)
            AA = self.data.transpose()*self.data
            
            if self.data.shape[1] > 1:                
                # do not compute full rank if desired
                if self._k > 0 and self._k < self.data.shape[1]-1:
                    k = self._k
                else:
                    k = self.data.shape[1]-1
                values, v_vectors = linalg.eigen_symmetric(AA,k=k)            
            else:                
                values, v_vectors = eigh(AA.todense())    
            # get rid of too low eigenvalues
            v_vectors = v_vectors[:, values > self._EPS] 
            values = values[values > self._EPS]
            
            # sort eigenvectors according to largest value
            idx = np.argsort(values)
            values = values[idx[::-1]]
            
            # argsort sorts in ascending order -> access is backwards            
            self.V = scipy.sparse.csc_matrix(v_vectors[:,idx[::-1]])
                    
            # compute S
            self.S = scipy.sparse.csc_matrix(np.diag(np.sqrt(values)))
            
            # and the inverse of it            
            S_inv = scipy.sparse.csc_matrix(np.diag(1.0/np.sqrt(values)))                                
            
            self.U = self.data * self.V * S_inv        
            self.V = self.V.transpose()
    
        
        if self._rows > self._cols:
            if scipy.sparse.issparse(self.data):
                _sparse_left_svd()
            else:            
                _left_svd()
        else:
            if scipy.sparse.issparse(self.data):
                _sparse_right_svd()
            else:            
                _right_svd()
            
if __name__ == "__main__":
    import doctest  
    doctest.testmod()    
