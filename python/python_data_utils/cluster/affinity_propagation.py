# coding: utf-8

"""
    description: Affinity propagation based clustering algorithm
    author: Suraj Iyer
"""

import numpy as np
from sklearn.cluster import AffinityPropagation


def affinity_propagation(items, similarity_matrix, item_id_included=False,
                         verbose=True, **kwargs):
    """
    Create clusters with affinity propagation using
    given similarity matrix between items as input.

    Paper:
        L Frey, Brendan J., and Delbert Dueck.
        "Clustering by passing messages between data points."
        science 315.5814 (2007): 972-976..
    """
    items = np.array(items)
    affprop = AffinityPropagation(affinity="precomputed", **kwargs)
    affprop.fit(similarity_matrix)
    clusters = dict()
    for cluster_id in np.unique(affprop.labels_):
        exemplar = items[affprop.cluster_centers_indices_[cluster_id]]
        if item_id_included:
            exemplar = exemplar[0]
            clusters[exemplar] = frozenset([idx for idx, _ in items[
                np.flatnonzero(affprop.labels_ == cluster_id)]])
        else:
            clusters[exemplar] = frozenset([d for d in items[
                np.flatnonzero(affprop.labels_ == cluster_id)]])
        if verbose:
            print(" - *%s:* %s" % (
                exemplar, ", ".join(str(d) for d in clusters[exemplar])))
    return clusters


def affinity_jaccard(items, verbose=True, **kwargs):
    """
    Cluster items with affinity propagation based on Jaccard similarity scores.

    :param items: list of tuples of type [(int, str),...]
        Each tuple in list of items is a pair of item id (int) and item (str).
    """
    assert isinstance(verbose, bool)

    # Compute Jaccard similarity between items
    import distance
    from ..numpy_utils import create_symmetric_matrix
    items = np.array([(idx, set(doc.split(" "))) for idx, doc in items])
    jaccard_similarity = [
        0 if idx1 == idx2 else -1 * distance.jaccard(doc1, doc2)
        for idx1, doc1 in items for idx2, doc2 in items if idx1 <= idx2]
    jaccard_similarity = create_symmetric_matrix(jaccard_similarity)

    # Create clusters with affinity propagation using
    # jaccard similarity between documents as input.
    return affinity_propagation(
        items, jaccard_similarity, True, verbose, **kwargs)


def affinity_ujaccard(items, depth=3, verbose=True, **kwargs):
    """
    Cluster text documents with affinity propagation
    based on Unilateral Jaccard similarity scores.

    :param items: list of tuples of type [(int, str),...]
        Each tuple in list of items is a pair of item id (int) and item (str).
    """
    assert isinstance(verbose, bool)

    # Computer unilateral Jaccard similarity between documents
    from unilateral_jaccard import unilateral_jaccard
    uJaccard_similarity = unilateral_jaccard(
        [set(doc.split(" ")) for _, doc in items], depth=depth)
    return affinity_propagation(
        items, uJaccard_similarity, True, verbose, **kwargs)
