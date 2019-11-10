# coding: utf-8

"""
    description: Affinity propagation clustering algorithm
    author: Suraj Iyer
"""

__all__ = [
    'ap_precomputed',
    'ap_jaccard',
    'ap_ujaccard']

import numpy as np
from sklearn.cluster import AffinityPropagation
from typing import Iterable, Tuple


def ap_precomputed(
        items: Iterable, similarity_matrix: Iterable[Iterable[float]],
        verbose: bool = True, **kwargs) -> dict:
    """
    Create clusters with affinity propagation using
    given similarity matrix between items as input.

    Paper:
        L Frey, Brendan J., and Delbert Dueck.
        "Clustering by passing messages between data points."
        science 315.5814 (2007): 972-976..

    :param items: Iterable
        N items to cluster. Can be the actual object like a string
    :param similarity_matrix: Iterable
        N x N matrix of similarity scores between each pair of N item.
    :param verbose: bool
    :param kwargs: Additional arguments for sklearn.cluster.AffinityPropagation
    :return: dict
    """
    try:
        items_iterator = iter(items)
        del items_iterator
    except TypeError:
        print('items must be an iterable.')
    assert isinstance(similarity_matrix, Iterable)\
        and isinstance(similarity_matrix[0], Iterable)\
        and len(similarity_matrix) == len(similarity_matrix[0]),\
        'similarity_matrix must be square shape iterable.'
    assert isinstance(verbose, bool), 'verbose must be a boolean.'

    items = np.array(items)
    clusterer = AffinityPropagation(affinity="precomputed", **kwargs)
    clusterer.fit(similarity_matrix)
    clusters = dict()
    for cluster_id in np.unique(clusterer.labels_):
        exemplar = items[clusterer.cluster_centers_indices_[cluster_id]]
        clusters[exemplar] = frozenset([d for d in items[
            np.flatnonzero(clusterer.labels_ == cluster_id)]])
        if verbose:
            print(" - *%s:* %s" % (
                exemplar, ", ".join(str(d) for d in clusters[exemplar])))
    return clusters


def ap_jaccard(items: Iterable[Tuple[int, set]],
               verbose: bool = True, **kwargs) -> dict:
    """
    Cluster items with affinity propagation based on Jaccard similarity scores.

    :param items: Iterable of tuples of type [(int, set),...]
        Each tuple in list of items is a pair of item id (int) and item (str).
    :param verbose: bool
    :param kwargs: Additional arguments for sklearn.cluster.AffinityPropagation
    :return: dict
    """
    assert isinstance(verbose, bool), 'verbose must be a boolean.'

    # Compute Jaccard similarity between items
    import distance
    from python_data_utils.numpy.utils import create_symmetric_matrix
    jaccard_similarity = [
        1. if idx1 == idx2 else -1 * distance.jaccard(doc1[1], doc2[1])
        for idx1, doc1 in enumerate(items)
        for idx2, doc2 in enumerate(items) if idx1 <= idx2]
    jaccard_similarity = create_symmetric_matrix(jaccard_similarity)
    items = [idx for idx, _ in items]

    # Create clusters with affinity propagation using
    # jaccard similarity between documents as input.
    return ap_precomputed(
        items, jaccard_similarity, verbose, **kwargs)


def ap_ujaccard(items: Iterable[Tuple[int, set]],
                depth: int = 3, n_jobs: int = 1,
                verbose: bool = True, **kwargs) -> dict:
    """
    Cluster text documents with affinity propagation
    based on Unilateral Jaccard similarity scores.

    :param items: Iterable of tuples of type [(int, set),...]
        Each tuple in list of items is a pair of item id (int) and item (set).
    :param depth: int
    :param n_jobs: int
        Number of processes to parallelize computation of the similarity matrix.
    :param verbose: bool
    :param kwargs: Additional arguments for sklearn.cluster.AffinityPropagation
    :return: dict
    """
    assert isinstance(depth, int) and depth > 0,\
        'depth must be a positive integer.'
    assert isinstance(verbose, bool), 'verbose must be a boolean.'

    # Computer unilateral Jaccard similarity between documents
    from .unilateral_jaccard import ujaccard_similarity_score, calculate_edges_list
    V = [doc for _, doc in items]
    E = calculate_edges_list(V)
    uJaccard_similarity = ujaccard_similarity_score(
        (range(len(V)), E), depth=depth, n_jobs=n_jobs)
    return ap_precomputed(
        [idx for idx, _ in items], uJaccard_similarity, verbose, **kwargs)
