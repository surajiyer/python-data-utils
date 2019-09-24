# coding: utf-8

"""
    description: Affinity propagation clustering algorithm
    author: Suraj Iyer
"""

import numpy as np
from sklearn.cluster import AffinityPropagation


def ap_precomputed(items, similarity_matrix,
                   verbose=True, **kwargs):
    """
    Create clusters with affinity propagation using
    given similarity matrix between items as input.

    Paper:
        L Frey, Brendan J., and Delbert Dueck.
        "Clustering by passing messages between data points."
        science 315.5814 (2007): 972-976..
    """
    try:
        items_iterator = iter(items)
        del items_iterator
    except TypeError:
        print('items must be an iterable.')
    assert isinstance(similarity_matrix, (list, tuple, np.ndarray))\
        and len(similarity_matrix) == len(similarity_matrix[0]),\
        'similarity_matrix must be square shape list, tuple or numpy array.'

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


def ap_jaccard(items, verbose=True, **kwargs):
    """
    Cluster items with affinity propagation based on Jaccard similarity scores.

    :param items: list of tuples of type [(int, str),...]
        Each tuple in list of items is a pair of item id (int) and item (str).
    """
    assert isinstance(verbose, bool), 'verbose must be a boolean.'

    # Compute Jaccard similarity between items
    import distance
    from ..numpy_utils import create_symmetric_matrix
    items = [(idx, set(doc.split(" "))) for idx, doc in items]
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


def ap_ujaccard(items, depth=3, n_jobs=1, verbose=True, **kwargs):
    """
    Cluster text documents with affinity propagation
    based on Unilateral Jaccard similarity scores.

    :param items: list of tuples of type [(int, str),...]
        Each tuple in list of items is a pair of item id (int) and item (str).
    """
    assert isinstance(depth, int) and depth > 0,\
        'depth must be a non-zero integer.'
    assert isinstance(verbose, bool), 'verbose must be a boolean.'

    # Computer unilateral Jaccard similarity between documents
    from .unilateral_jaccard import ujaccard_similarity_score
    uJaccard_similarity = ujaccard_similarity_score(
        [set(doc.split(" ")) for _, doc in items], depth=depth, n_jobs=n_jobs)
    return ap_precomputed(
        [idx for idx, _ in items], uJaccard_similarity, verbose, **kwargs)
