#!/usr/bin/python
#
# Copyright (C) Christian Thurau, 2010. 
# Licensed under the GNU General Public License (GPL). 
# http://www.gnu.org/licenses/gpl.txt
"""
PyMF several distance functions

    kl_divergence(): KL Divergence
    l1_distance(): L1 distance
    l2_distance(): L2 distance
    cosine_distance(): Cosine distance 
    pdist(): Pairwise distance computation
    vq(): Vector quantization
    
"""


import numpy as np
import scipy.sparse

__all__ = ["abs_cosine_distance", "kl_divergence", "l1_distance", "l2_distance", 
           "weighted_abs_cosine_distance","cosine_distance","vq", "pdist"]

def kl_divergence(d, vec):    
    b = vec*(1/d)    
    b = np.where(b>0, np.log(b),0)    
    b = vec * b
    b = np.sum(b - vec + d, axis=0).reshape((-1))            
    return b
        
def l1_distance(d, vec):            
    ret_val = np.sum(np.abs(d - vec), axis=0)                    
    ret_val = ret_val.reshape((-1))                        
    return ret_val
    
def sparse_l2_distance(d, vec):
    # compute the norm of d
    nd = (d.multiply(d)).sum(axis=0)
    nv = (vec.multiply(vec)).sum(axis=0)
    ret_val = nd + nv -  2.0*(d.T * vec).T
    return np.sqrt(ret_val)

def approx_l2_distance(d, vec):
    # Use random projections to approximate the conventional l2 distance
    k = np.round(np.log(d.shape[0]))
    #k = d.shape[0]
    R = np.random.randn(k, d.shape[0])
    R = R / np.sqrt((R**2).sum(axis=0))  
    A = np.dot(R,d)
    B = np.dot(R, vec)
    ret_val = np.sum( (A - B)**2, axis=0)                    
    ret_val = np.sqrt(R.shape[1]/R.shape[0]) * np.sqrt(ret_val)
    ret_val = ret_val.reshape((-1))    
    return ret_val
    
def l2_distance(d, vec):    
    if scipy.sparse.issparse(d):
        ret_val = sparse_l2_distance(d, vec)
    else:
        ret_val = np.sqrt(((d[:,:] - vec)**2).sum(axis=0))
            
    return ret_val.reshape((-1))        

def l2_distance_new(d,vec):
    # compute the norm of d
    nd = (d**2).sum(axis=0)
    nv = (vec**2).sum(axis=0)
    ret_val = nd + nv - 2.0*np.dot(d.T,vec.reshape((-1,1))).T

    return np.sqrt(ret_val)
    
def cosine_distance(d, vec):
    tmp = np.dot(np.transpose(d), vec)
    a = np.sqrt(np.sum(d**2, axis=0))
    b = np.sqrt(np.sum(vec**2))
    k = (a*b).reshape(-1) + (10**-9)
    
    # compute distance
    ret_val = 1.0 - tmp/k
    
    return ret_val.reshape((-1))

def abs_cosine_distance(d, vec, weighted=False):
    if scipy.sparse.issparse(d):
        tmp = np.array((d.T * vec).todense(), dtype=np.float32).reshape(-1)
        a = np.sqrt(np.array(d.multiply(d).sum(axis=0), dtype=np.float32).reshape(-1))
        b = np.sqrt(np.array(vec.multiply(vec).sum(axis=0), dtype=np.float32).reshape(-1))
    else:
        tmp = np.dot(np.transpose(d), vec).reshape(-1)
        a = np.sqrt(np.sum(d**2, axis=0)).reshape(-1)
        b = np.sqrt(np.sum(vec**2)).reshape(-1)
        
    k = (a*b).reshape(-1) + 10**-9
    
    # compute distance
    ret_val = 1.0 - np.abs(tmp/k)    
    
    if weighted:    
        ret_val = ret_val * a        
    return ret_val.reshape((-1))

def weighted_abs_cosine_distance(d, vec):
    ret_val = abs_cosine_distance(d, vec, weighted=True)        
    return ret_val

def pdist(A, B, metric='l2' ):
    # compute pairwise distance between a data matrix A (d x n) and B (d x m).
    # Returns a distance matrix d (n x m).
    d = np.zeros((A.shape[1], B.shape[1]))
    if A.shape[1] <= B.shape[1]:
        for aidx in xrange(A.shape[1]):
            if metric == 'l2':
                d[aidx:aidx+1,:] = l2_distance(B[:,:], A[:,aidx:aidx+1]).reshape((1,-1))
            if metric == 'l1':
                d[aidx:aidx+1,:] = l1_distance(B[:,:], A[:,aidx:aidx+1]).reshape((1,-1))
    else:
        for bidx in xrange(B.shape[1]):
            if metric == 'l2':
                d[:, bidx:bidx+1] = l2_distance(A[:,:], B[:,bidx:bidx+1]).reshape((-1,1))
            if metric == 'l1':
                d[:, bidx:bidx+1] = l1_distance(A[:,:], B[:,bidx:bidx+1]).reshape((-1,1))               
    
    return d

def vq(A, B, metric='l2'):
    # assigns data samples in B to cluster centers A and
    # returns an index list [assume n column vectors, d x n]
    assigned = np.argmin(pdist(A,B, metric=metric), axis=0)
    return assigned
