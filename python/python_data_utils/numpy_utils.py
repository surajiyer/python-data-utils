# coding: utf-8

"""
    description: Numpy utility functions and classes
    author: Suraj Iyer
"""

import numpy as np


def create_upper_matrix(values):
    """
    Create an upper triangular matrix with any input array.
    Example:
        1. IN: values = [1,2,3]
            OUT: [[1, 2
                        0, 3]]
        2. IN: values = [1,2,3,4,5,6]
            OUT: [[1, 2, 3]
                       [0, 4, 5]
                       [0, 0, 6]]
        3. IN: values = [1,2,3,4]
            OUT: [[1, 2, 3]
                       [0, 4, 0]
                       [0, 0, 0]]
    :param values: list of numbers
    :return: numpy upper triangular matrix.
    """
    def closest_inverse_sum2n(n):
        for i in range(1, n + 1):
            sumx = sum(range(i + 1))
            if sumx >= n:
                return i, sumx

    n = len(values)
    size, length = closest_inverse_sum2n(n)
    upper = np.zeros((size, size))
    upper[np.triu_indices(size, 0)] = values + [0] * (length - len(values))
    return upper


def symmetrize(a):
    """
    Input 2D-Matrix and get a symmetric 2D matrix.
    """
    return a + a.T - np.diag(a.diagonal())


def create_symmetric_matrix(values):
    """
    Create a symmetric matrix with given input array.
    """
    return symmetrize(create_upper_matrix(values))


def pairwise_difference(values):
    """
    """
    return values - values[:, None]


def is_pos_def(values):
    """
    Is matrix :values: positive definite?
    """
    if np.allclose(values, values.T):
        try:
            np.linalg.cholesky(values)
            return True
        except np.linalg.LinAlgError:
            return False
    else:
        return False


def rowwise_dissimilarity(values):
    """
    Compare every row with each other row and count number
    of differences along column axis per row pairs.

    Example:
        input: [[1, 2, 3],
                [1, 3, 1],
                [2, 2, 2]]
        output: [[0, 2, 2],
                 [2, 0, 3]
                 [2, 3, 0]]
    """
    return np.sum(values != values[:, None], axis=-1)


def rowwise_cosine_similarity(values):
    """
    Using every pair of rows in :values: as input, compute
    pairwise cosine similarity between each row.

    URL: https://stackoverflow.com/questions/41905029/create-cosine-similarity-matrix-numpy
    """
    norm = (values * values).sum(0, keepdims=True) ** .5
    values = values / norm
    return (values.T @ values)


def combinations_with_replacements(*arr, k=2):
    """
    Take k-size combination pairs from given list of arrays.
    """
    # return np.stack(np.meshgrid(*arr)).reshape(-1, k)
    return np.array(np.meshgrid(*arr)).T.reshape(-1, k)
