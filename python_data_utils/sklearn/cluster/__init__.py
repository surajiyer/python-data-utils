# coding: utf-8

"""
    description: Clustering algorithms
    author: Suraj Iyer
"""

__all__ = [
    'ap_precomputed',
    'hdbscan_precomputed',
    'ap_jaccard']


import numpy as np
from typing import Any, Dict, FrozenSet, Iterable, Tuple, Set


def ap_precomputed(
        similarity_matrix: Iterable[Iterable[float]],
        verbose: bool = False, **kwargs) -> Dict[int, FrozenSet[int]]:
    """
    Create clusters with affinity propagation using
    given similarity matrix between items as input.

    Paper
    -----
    L Frey, Brendan J., and Delbert Dueck.
    "Clustering by passing messages between data points."
    science 315.5814 (2007): 972-976..

    Params
    ------
    similarity_matrix: Iterable
        N x N matrix of similarity scores between
        each pair of N items.

    verbose: bool (default = False)
        Verbosity

    kwargs:
        Additional arguments for sklearn.cluster.AffinityPropagation

    Returns
    -------
    Dict[int, FrozenSet[int]]
        Mapping from i_th index (0 <= i < N) to
        set of `j` indices of the similarity matrix.
    """
    assert isinstance(similarity_matrix, Iterable)\
        and isinstance(similarity_matrix[0], Iterable)\
        and len(similarity_matrix) == len(similarity_matrix[0]),\
        'similarity_matrix must be square shape iterable.'
    assert isinstance(verbose, bool), 'verbose must be a boolean.'

    from sklearn.cluster import AffinityPropagation
    clusterer = AffinityPropagation(affinity="precomputed", **kwargs)
    clusterer.fit(similarity_matrix)
    clusters = dict()
    for cluster_id in np.unique(clusterer.labels_):
        exemplar = clusterer.cluster_centers_indices_[cluster_id]
        clusters[exemplar] = frozenset(
            np.flatnonzero(clusterer.labels_ == cluster_id).tolist())
        if verbose:
            print(f" - *{exemplar}:*", ", ".join(str(d) for d in clusters[exemplar]))
    return clusters


def hdbscan_precomputed(
        similarity_matrix: Iterable[Iterable[float]],
        verbose: bool = False, **kwargs) -> Dict[int, FrozenSet[int]]:
    """
    Create clusters with HDBSCAN using
    given similarity matrix between items as input.

    URL: https://github.com/scikit-learn-contrib/hdbscan

    Params
    ------
    similarity_matrix: Iterable
        N x N matrix of similarity scores between
        each pair of N items.

    verbose: bool (default = False)
        Verbosity

    kwargs:
        Additional arguments for hdbscan.HDBSCAN

    Returns
    -------
    Dict[int, FrozenSet[int]]
        Mapping from i_th index (0 <= i < N) to
        set of `j` indices of the similarity matrix.
    """
    assert isinstance(similarity_matrix, Iterable)\
        and isinstance(similarity_matrix[0], Iterable)\
        and len(similarity_matrix) == len(similarity_matrix[0]),\
        'similarity_matrix must be square shape iterable.'
    assert isinstance(verbose, bool), 'verbose must be a boolean.'

    import hdbscan
    clusterer = hdbscan.HDBSCAN(metric='precomputed', **kwargs)
    clusterer.fit(-similarity_matrix)
    clusters = dict()
    for cluster_id in np.unique(clusterer.labels_):
        cluster = np.flatnonzero(clusterer.labels_ == cluster_id)
        # compute the exemplar by taking the least outlier point
        # per cluster, i.e., point near the densest region.
        exemplar = np.argmin(clusterer.outlier_scores_[cluster])
        clusters[exemplar] = frozenset(cluster.tolist())
        if verbose:
            print(f" - *{exemplar}:*", ", ".join(str(d) for d in clusters[exemplar]))
    return clusters


def ap_jaccard(
        items: Iterable[Set[Any]],
        verbose: bool = False, **kwargs) -> Dict[int, FrozenSet[int]]:
    """
    Cluster items with affinity propagation based on Jaccard similarity scores.

    Params
    ------
    items: Iterable[Set[Any]]
        Each item is of type Set.

    verbose: bool (default = False)

    kwargs:
        Additional arguments for ap_precomputed()

    Returns
    -------
    Dict[int, FrozenSet[int]]
        Mapping from i_th index (0 <= i < N) to
        set of `j` indices of the similarity matrix.
    """
    assert isinstance(verbose, bool), 'verbose must be a boolean.'

    # Compute Jaccard similarity between items
    import distance
    from python_data_utils.numpy.utils import create_symmetric_matrix
    jaccard_similarity = [
        1. if i == j else -1 * distance.jaccard(doc1, doc2)
        for i, doc1 in enumerate(items)
        for j, doc2 in enumerate(items) if i <= j]
    jaccard_similarity = create_symmetric_matrix(jaccard_similarity)

    # Create clusters with affinity propagation using
    # jaccard similarity between documents as input.
    return ap_precomputed(
        jaccard_similarity, verbose, **kwargs)
