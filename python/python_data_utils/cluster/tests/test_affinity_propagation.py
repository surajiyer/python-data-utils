import numpy as np
import sklearn.datasets as data
from sklearn.metrics.pairwise import pairwise_distances
import random
import string
import python_data_utils.cluster.affinity_propagation as ap


def test_1():
    # Test ap_precomputed()
    moons, _ = data.make_moons(n_samples=50, noise=0.05)
    blobs, _ = data.make_blobs(
        n_samples=50, centers=[(-0.75, 2.25), (1.0, 2.0)], cluster_std=0.25)
    test_data = np.vstack([moons, blobs])
    similarity_matrix = pairwise_distances(test_data)
    clusters = ap.ap_precomputed(
        np.arange(test_data.shape[0]),
        similarity_matrix)
    assert isinstance(clusters, dict)


def test_2():
    # Test ap_jaccard()
    N = 10
    n_samples = 50
    test_data = []
    for i in range(n_samples):
        test_data.append((i, set(
            ''.join(random.choice(string.ascii_uppercase[:5]) for _ in range(N)))))
    clusters = ap.ap_jaccard(test_data)
    assert isinstance(clusters, dict)


def test_3():
    # Test ap_ujaccard()
    N = 10
    n_samples = 50
    test_data = []
    for i in range(n_samples):
        test_data.append((i, set(
            ''.join(random.choice(string.ascii_uppercase[:5]) for _ in range(N)))))
    clusters = ap.ap_ujaccard(test_data)
    assert isinstance(clusters, dict)
