import numpy as np
import sklearn.datasets as data
from sklearn.metrics.pairwise import pairwise_distances
import random
import string
from python_data_utils.sklearn.cluster import (
    ap_precomputed,
    hdbscan_precomputed,
    ap_jaccard
)


def test_1():
    # Test ap_precomputed()
    moons, _ = data.make_moons(n_samples=50, noise=0.05)
    blobs, _ = data.make_blobs(
        n_samples=50, centers=[(-0.75, 2.25), (1.0, 2.0)], cluster_std=0.25)
    test_data = np.vstack([moons, blobs])
    similarity_matrix = pairwise_distances(test_data)
    clusters = ap_precomputed(similarity_matrix)
    assert isinstance(clusters, dict)


def test_2():
    # Test hdbscan_precomputed()
    moons, _ = data.make_moons(n_samples=50, noise=0.05)
    blobs, _ = data.make_blobs(
        n_samples=50, centers=[(-0.75, 2.25), (1.0, 2.0)], cluster_std=0.25)
    test_data = np.vstack([moons, blobs])
    similarity_matrix = pairwise_distances(test_data)
    clusters = hdbscan_precomputed(
        similarity_matrix, min_cluster_size=5)
    assert isinstance(clusters, dict)


def test_3():
    # Test ap_jaccard()
    N = 10
    n_samples = 50
    test_data = []
    for i in range(n_samples):
        test_data.append((i, set(
            ''.join(random.choice(string.ascii_uppercase[:5]) for _ in range(N)))))
    clusters = ap_jaccard(test_data)
    assert isinstance(clusters, dict)
