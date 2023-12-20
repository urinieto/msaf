# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 16:30:31 2022

@author: amarmore

Module used to compute autosimilarity matrices.
"""
import numpy as np
import sklearn.metrics.pairwise as pairwise_distances
import warnings

def switch_autosimilarity(an_array, similarity_type, gamma = None, normalize = True):
    """
    High-level function to compute the autosimilarity of this matrix.
    
    Expects a matrix of shape (Bars, Feature representation).
    
    Computes it with different possible similarity function s_{x_i,x_j} (given two bars denoted as x_i and x_j):
        - "cosine" for the cosine similarity, i.e. the normalized dot product:
        .. math::
            s_{x_i,x_j} = \\frac{\\langle x_i, x_j \\rangle}{||x_i|| ||x_j||}
        -"autocorrelation" for a covariance similarity, 
        i.e. the dot product of centered features:
        .. math::
            s_{x_i,x_j} = \\langle x_i - \\hat{x}, x_j - \\hat{x} \\rangle
        -"rbf" for the Radial Basis Function similarity, 
        i.e. the exponent of the opposite of the euclidean distance between features:
        .. math::
            s_{x_i,x_j} = \\exp^{-\\gamma ||x_i - x_j||_2}
        The euclidean distance can be the distance between the normalized features.
        Gamma is a parameter.
        See rbf_kernel from scikit-learn for more details.
    
    Parameters
    ----------
    an_array : numpy array
        The array/matrix seen as array which autosimilarity will be computed.
        Expected to be of shape (Bars, Feature representation).
    similarity_type : string
        Either "cosine", "covariance" or "rbf".
        It represents the similarity function used for computing the autosimilarity.
    gamma : positive float, optional
        The gamma parameter in the rbf function, only used for the "rbf" similarity.
        The default is None, meaning that it is computed as function of the standard deviation,
        see get_gamma_std() for more details.
    normalize : boolean, optional
        Whether features should be normalized or not. 
        Normalization depends on the similarity function.
        The default is True.

    Returns
    -------
    numpy array
        Autosimilarity matrix of the input an_array.

    """
    if similarity_type.lower() == "cosine":
        return get_cosine_autosimilarity(an_array)
    elif similarity_type.lower() in ["covariance", "autocorrelation"]:
        return get_autocorrelation_autosimilarity(an_array, normalize = normalize)
    elif similarity_type.lower() == "rbf":
        return get_rbf_autosimilarity(an_array, gamma, normalize = normalize)
    else:
        raise ValueError(f"Incorrect similarity type: {similarity_type}. Should be cosine, covariance or rbf.")
        
def l2_normalize_barwise(an_array):
    """
    Normalizes the array barwise (i.e., in its first dimension) by the l_2 norm.
    
    Null values are replaced by the small positive value of 10^{-10}.

    Parameters
    ----------
    an_array : numpy array
        The array which needs to be normalized.

    Returns
    -------
    numpy array
        The normalized array.

    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="invalid value encountered in true_divide") # Avoiding to show the warning, as it's handled, not te confuse the user.
        an_array_T = an_array.T/np.linalg.norm(an_array, axis = 1)
        an_array_T = np.where(np.isnan(an_array_T), 1e-10, an_array_T) # Replace null lines, avoiding future errors in handling values.
    return an_array_T.T

def get_cosine_autosimilarity(an_array):
    """
    Computes the autosimilarity matrix with the cosine similarity function.
    
    The cosine similarity function is the normalized dot product between two bars, i.e.:
    .. math::
        s_{x_i,x_j} = \\frac{\\langle x_i, x_j \\rangle}{||x_i|| ||x_j||}
    
    Parameters
    ----------
    an_array : numpy array
        The array/matrix seen as array which autosimilarity os to compute.
        Expected to be of shape (Bars, Feature representation).

    Returns
    -------
    numpy array
        The autosimilarity of this array, with the cosine similarity function.

    """
    if type(an_array) is list:
        this_array = np.array(an_array)
    else:
        this_array = an_array
    this_array = l2_normalize_barwise(this_array)
    return this_array@this_array.T

def get_autocorrelation_autosimilarity(an_array, normalize = True):
    """
    Computes the autosimilarity matrix, where the similarity function is the autocorrelation (sometimes called 'covariance').
    
    The autocorrelation similarity function corresponds to the dot product of centered features:
    .. math::
        s_{x_i,x_j} = \\langle x_i - \\hat{x}, x_j - \\hat{x} \\rangle

    Parameters
    ----------
    an_array : numpy array
        The array/matrix seen as array which autosimilarity will be computed.
    normalize : boolean, optional
        Whether features should be normalized or not. 
        Normalization here means that each centered feature is normalized by its norm.
        The default is True.
        
    Returns
    -------
    numpy array
        The autocorrelation autosimilarity of this array.

    """
    if type(an_array) is list:
        this_array = np.array(an_array)
    else:
        this_array = an_array
    this_array = this_array - this_array.mean(axis=0) # centering, i.e. subtracting the average value row-wise
    if normalize:
        this_array = l2_normalize_barwise(this_array)
    return this_array@this_array.T

def get_rbf_autosimilarity(an_array, gamma = None, normalize = True):
    """
    Computes the autosimilarity matrix, where the similarity function is the Radial Basis Function (RBF).
    
    The RBF corresponds to the exponent of the opposite of the euclidean distance between features:
    .. math::
        s_{x_i,x_j} = \\exp^{-\\gamma ||x_i - x_j||_2}
        
    The RBF is computed via scikit-learn.
    The default gamma value is computed in function get_gamma_std(), refer to that function for further details.

    Parameters
    ----------
    an_array : numpy array
        The array/matrix seen as array which autosimilarity will be computed.
    gamma : positive float, optional
        The gamma parameter in the rbf function.
        The default is None, meaning that it is computed as function of the standard deviation,
        see get_gamma_std() for more details.
    normalize : boolean, optional
        Whether features should be normalized or not. 
        Normalization here means that the euclidean norm is computed between normalized vectors.
        The default is True.

    Returns
    -------
    numpy array
        The RBF autosimilarity of this array.

    """
    if type(an_array) is list:
        this_array = np.array(an_array)
    else:
        this_array = an_array
    if gamma == None:
        gamma = get_gamma_std(this_array, scaling_factor = 1, no_diag = True, normalize = normalize)
    if normalize:
        this_array = l2_normalize_barwise(this_array)
    return pairwise_distances.rbf_kernel(this_array, gamma = gamma)
    
def get_gamma_std(an_array, scaling_factor = 1, no_diag = True, normalize = True):
    """
    Default value for the gamma in the RBF similarity function.
    
    This default value is proportional to the inverse of the standard deviation of the values, more experiments should be made to fit it.
    For now, it has been set quite empirically.

    Parameters
    ----------
    an_array : numpy array
        The array/matrix seen as array which autosimilarity will be computed.
    scaling_factor : positive float, optional
        Weigthing parameter, relating to the inverse of the standard deviation. 
        The default is 1.
    no_diag : boolen, optional
        Whether the diagonal values (self similarity values) should be discarded (True) or taken into account (False). 
        The default is True.
    normalize : boolean, optional
        Whether features should be normalized or not. 
        Normalization here means that the euclidean norm is computed between normalized vectors.
        The default is True.

    Returns
    -------
    gamma : float
        The gamma parameter in the RBF similarity function.

    """
    if normalize:
        an_array = l2_normalize_barwise(an_array)
    euc_dist = pairwise_distances.euclidean_distances(an_array)
    if not no_diag:
        return scaling_factor/(2*np.std(euc_dist))
    else:
        for i in range(len(euc_dist)):
            euc_dist[i,i] = float('NaN')
        return scaling_factor/(2*np.nanstd(euc_dist))