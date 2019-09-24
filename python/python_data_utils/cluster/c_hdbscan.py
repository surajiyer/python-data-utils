# coding: utf-8

"""
    description: HDBSCAN clustering algorithm
    author: Suraj Iyer
"""

import numpy as np
import hdbscan


def hdbscan_precomputed(items, similarity_matrix,
                        verbose=True, **kwargs):
    """
    Create clusters with HDBSCAN using
    given similarity matrix between items as input.

    URL: https://github.com/scikit-learn-contrib/hdbscan
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
    clusterer = hdbscan.HDBSCAN(**kwargs)
    clusterer.fit(similarity_matrix)
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


if __name__ == "__main__":
    import sklearn.datasets as data
    from sklearn.metrics.pairwise import pairwise_distances
    moons, _ = data.make_moons(n_samples=50, noise=0.05)
    blobs, _ = data.make_blobs(
        n_samples=50, centers=[(-0.75, 2.25), (1.0, 2.0)], cluster_std=0.25)
    test_data = np.vstack([moons, blobs])
    similarity_matrix = pairwise_distances(test_data)
    clusters = hdbscan_precomputed(
        np.arange(test_data.shape[0]),
        similarity_matrix,
        min_cluster_size=5)
    print(clusters)
