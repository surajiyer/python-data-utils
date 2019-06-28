# coding: utf-8

"""
    description: Multi-view document clustering via ensemble method
        https://link.springer.com/article/10.1007/s10844-014-0307-6
        Paper authors: Syed Fawad Hussain, Muhammad Mushtaq, Zahid Halim
    author: Suraj Iyer
"""

import pandas as pd
import numpy as np


def cluster_based_similarity_matrix(partitions: np.ndarray):
    """
    Calculate cluster based similarity matrix for given cluster partitions.

    Parameters:
    -----------
    partitions: np.ndarray
        Output of clustering algorithms on multiple views of the data.
        Each row corresponds to a data point and each column a view.
        The cells are the output of cluster algorithm(s) for given data point
        conditioned on each view.

    Returns:
    --------
    np.ndarray
        cluster based similarity matrix
    """
    k = partitions.shape[1]
    df = pd.concat([pd.get_dummies(partitions[c], prefix=c) for c in partitions], axis=1)
    return (1 / k * (df @ df.transpose(copy=True))).values


def pdm(partitions: np.ndarray):
    from ..numpy_utils import rowwise_dissimilarity
    return rowwise_dissimilarity(partitions)


def pairwise_dissimilarity_matrix(partitions: np.ndarray):
    """
    Calculate pairwise dissimilarity matrix for given cluster partitions.

    Parameters:
    -----------
    partitions: np.ndarray
        Output of clustering algorithms on multiple views of the data.
        Each row corresponds to a data point and each column a view.
        The cells are the output of cluster algorithm(s) for given data point
        conditioned on each view.

    Returns:
    --------
    np.ndarray
        Pairwise dissimilarity matrix
    """
    from ..numpy_utils import rowwise_cosine_similarity
    pmd = rowwise_cosine_similarity(pdm(partitions))
    return pmd


def affinity_matrix(distance_matrix: np.ndarray, c: float):
    """
    Calculate afinity matrix for given ditance matrix.

    Parameters:
    -----------
    distance_matrix: np.ndarray
        distance matrix (opposite of similarity matrix)
    c: float
        scaling factor

    Returns:
    --------
    np.ndarray
        Affinity matrix
    """
    return np.exp(- (distance_matrix ** 2) / c)


def aggregate_matrices(cbsm, pdm, am):
    """
    Combine the given similarity matrices into one similarity matrix.

    Parameters:
    -----------
    cbsm: np.ndarray
        Cluster based similarity matrix
    pdm: np.ndarray
        Pairwise dissimilarity matrix
    am: np.ndarray
        Affinity matrix

    Returns:
    --------
    np.ndarray
        Combined similarity matrix
    """
    # D = distance matrix; D = 1 - S.
    D = 1. - ((cbsm + pdm + am) / 3.)

    # Fix triangular inequality within distance matrix
    # by converting to ultra-metric by ensuring the following
    # condition: d_{ij} = min(d_{ij}, max(d_{ik}, d_{kj}))
    _D = np.zeros(D.shape, dtype=np.float32)
    for i, j in np.ndindex(D.shape):
        _D[i, j] = min(
            D[i, j], max([max(D[i, k], D[k, j]) for k in np.ndindex(D.shape[0])]))

    return 1. - _D
