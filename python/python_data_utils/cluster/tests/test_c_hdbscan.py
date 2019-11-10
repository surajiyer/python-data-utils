import numpy as np
import sklearn.datasets as data
from sklearn.metrics.pairwise import pairwise_distances
from python_data_utils.cluster import c_hdbscan


def test_1():
    moons, _ = data.make_moons(n_samples=50, noise=0.05)
    blobs, _ = data.make_blobs(
        n_samples=50, centers=[(-0.75, 2.25), (1.0, 2.0)], cluster_std=0.25)
    test_data = np.vstack([moons, blobs])
    similarity_matrix = pairwise_distances(test_data)
    clusters = c_hdbscan.hdbscan_precomputed(
        np.arange(test_data.shape[0]),
        similarity_matrix,
        min_cluster_size=5)
    assert isinstance(clusters, dict)
