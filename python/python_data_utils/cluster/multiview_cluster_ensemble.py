# coding: utf-8

"""
    description: Multi-view document clustering via ensemble method
        https://link.springer.com/article/10.1007/s10844-014-0307-6
        Paper authors: Syed Fawad Hussain, Muhammad Mushtaq, Zahid Halim
    author: Suraj Iyer
"""

__package__ = "python_data_utils.cluster"
__all__ = [
    'cluster_based_similarity_matrix',
    'pairwise_dissimilarity_matrix',
    'affinity_matrix',
    'aggregate_matrices',
    'multiview_ensemble_similarity'
]

import pandas as pd
import time
from ..numpy import utils as npu
from scipy.spatial import distance
import numpy as np


def cluster_based_similarity_matrix(partitions: pd.DataFrame):
    """
    Calculate cluster based similarity matrix.

    Parameters:
    -----------
    partitions: pd.DataFrame
        Output of clustering algorithms on multiple views of the data.
        Each row corresponds to a data point and each column a view.
        The cells are the output of cluster algorithm(s) for given data point
        conditioned on each view.

    Returns:
    --------
    np.ndarray
        cluster based similarity matrix
    """
    # # construct a hyper graph adjacency matrix from these partitions
    # df = pd.concat([
    #     pd.get_dummies(partitions[c], prefix=c) for c in partitions], axis=1)

    # # calculate a new cluster based similarity matrix
    # k = partitions.shape[1]
    # return (1 / k * (df @ df.transpose(copy=True))).values

    # Encode consensus partitions into integer labels
    partitions = partitions.apply(
        lambda s: s.astype('category').cat.codes, axis=0).values

    # calculate a new cluster based similarity matrix
    result = np.full((partitions.shape[0], partitions.shape[0]), 0.)
    for i in range(partitions.shape[0]):
        result[i, i] = 1.
        result[i, i + 1:] = distance.cdist(
            partitions[None, i], partitions[i + 1:], "hamming")[0]
        result[i + 1:, i] = result[i, i + 1:]

    return 1. - result


def pdm(partitions: np.ndarray):
    return npu.rowwise_dissimilarity(partitions)


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
    pmd = npu.rowwise_cosine_similarity(pdm(partitions))
    return pmd


def affinity_matrix(distance_matrix: np.ndarray, c: float):
    """
    Calculate affinity matrix for given distance matrix.

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


def aggregate_matrices(cbsm, pdm, afm, tol=1e-08):
    """
    Combine the given similarity matrices into one similarity matrix.

    Parameters:
    -----------
    cbsm: np.ndarray
        Cluster based similarity matrix
    pdm: np.ndarray
        Pairwise dissimilarity matrix
    afm: np.ndarray
        Affinity matrix
    tol: float
        tolerance value

    Returns:
    --------
    np.ndarray
        Combined similarity matrix
    """
    # D = distance matrix; D = 1 - S.
    D = 1. - ((cbsm + pdm + afm) / 3.)
    assert -tol <= np.min(D) and (np.max(D) - 1.) <= tol
    D_new = npu.convert_to_ultrametric(D)
    return 1. - D_new


def multiview_ensemble_similarity(partitions, *similarity_matrices,
                                  affinity_c=.1, verbose=True):
    if verbose:
        print("Creating cluster-based similarity matrix.")
        start = time.time()
    cbsm = cluster_based_similarity_matrix(partitions)
    if verbose:
        print(f"Time to run: {time.time() - start}")
        print("Creating pairwise dissimilarity matrix.")
        start = time.time()
    pdm = pairwise_dissimilarity_matrix(partitions.values)
    if verbose:
        print(f"Time to run: {time.time() - start}")
        print("Creating affinity matrix.")
        start = time.time()
    afm_arr = []
    for sim_m in similarity_matrices:
        afm_arr.append(affinity_matrix(sim_m, c=affinity_c))
    # take the average of all affinity matrices
    afm = np.mean(np.array(afm_arr), axis=0)
    if verbose:
        print(f"Time to run: {time.time() - start}")
        print("Aggregating the matrices.")
        start = time.time()
    similarity_matrix = aggregate_matrices(cbsm, pdm, afm)
    if verbose:
        print(f"Time to run: {time.time() - start}")
    return similarity_matrix, cbsm, pdm, afm
