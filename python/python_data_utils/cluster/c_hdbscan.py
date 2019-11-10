# coding: utf-8

"""
    description: HDBSCAN clustering algorithm
    author: Suraj Iyer
"""

__all__ = ['hdbscan_precomputed']

import numpy as np
import hdbscan
from typing import Iterable


def hdbscan_precomputed(
        items: Iterable, similarity_matrix: Iterable[Iterable[float]],
        verbose: bool = True, **kwargs) -> dict:
    """
    Create clusters with HDBSCAN using
    given similarity matrix between items as input.

    URL: https://github.com/scikit-learn-contrib/hdbscan
    :param items:
    :param similarity_matrix:
    :param verbose:
    :param kwargs:
    :return:
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
    kwargs.pop('metric', None)
    clusterer = hdbscan.HDBSCAN(metric='precomputed', **kwargs)
    clusterer.fit(-similarity_matrix)
    clusters = dict()
    for cluster_id in np.unique(clusterer.labels_):
        mask = np.flatnonzero(clusterer.labels_ == cluster_id)
        cluster = items[mask]

        # compute the exemplar by taking the least outlier point
        # per cluster, i.e., point near the densest region.
        exemplar = cluster[np.argmin(clusterer.outlier_scores_[mask])]

        clusters[exemplar] = frozenset([d for d in cluster])
        if verbose:
            print(" - *%s:* %s" % (
                exemplar, ", ".join(str(d) for d in clusters[exemplar])))
    return clusters
