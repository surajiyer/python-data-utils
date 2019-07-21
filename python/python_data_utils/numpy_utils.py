# coding: utf-8

"""
    description: Numpy utility functions and classes
    author: Suraj Iyer
"""

import numpy as np
import warnings

try:
    import numba as nb
except (ImportError, ModuleNotFoundError):
    warnings.warn("Numba is not installed. Continuing without it.")

    class Object(object):
        pass
    nb = Object()
    nb.jit = lambda x: x
    nb.njit = lambda x: x


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
    upper[np.triu_indices(size, 0)] = np.append(
        values, [0] * (length - len(values)))
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
    Take difference between every element pair.
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


def filter_rows_with_unique_values_only(values):
    """
    Keep rows containing all unique elements only.

    URL: https://stackoverflow.com/questions/26958233/numpy-row-wise-unique-elements/
    """
    k = values.shape[1]
    for i in range(k - 1):
        values = values[(
            values[:, i, None] != values[:, range(i + 1, k)]).all(axis=1)]
    return values


def unique_per_row(values):
    """
    Count number of unique elements per row.

    URL: https://stackoverflow.com/questions/26958233/numpy-row-wise-unique-elements/
    """
    weight = 1j * np.linspace(
        0, values.shape[1], values.shape[0], endpoint=False)
    b = values + weight[:, None]
    u, ind, cnt = np.unique(b, return_index=True, return_counts=True)
    b = np.zeros_like(values)
    np.put(b, ind, cnt)
    return b


def drop_duplicates(values, ignore_order=True):
    """
    Drop duplicate rows.

    Parameters:
    -------------
    values: np.ndarray
        2D rectangular matrix.
    ignore_order: bool
        If true, ignore order of elements in each row
        when computing uniqueness, else different
        orders of same elements will be treated as
        unique rows.

    Returns:
    ---------
    np.ndarray:
        2D array with duplicate rows removed.
    """
    val = np.sort(values, axis=1) if ignore_order else values
    u, idx = np.unique(val, return_index=True, axis=0)
    return values[idx]


def permutations_with_replacement(*arr, k=2, shape=None):
    """
    Take k-size combinations of elements from given
    list of arrays.
    """
    if shape:
        arr = (range(s) for s in shape)
        k = len(shape)
    return np.array(np.meshgrid(*arr)).T.reshape(-1, k)


def permutations(*arr, k=2, shape=None):
    """
    Take k-size permutations of elements from give
    list of arrays without replacement, i.e., each element
    in any permutation only occurs once.
    """
    if shape:
        arr = (range(s) for s in shape)
        shape = None
    values = permutations_with_replacement(*arr, k=k, shape=shape)
    values = filter_rows_with_unique_values_only(values)
    return values


def combinations_with_replacement(*arr, k=2, shape=None):
    """
    Take k-size combinations of elements from given
    list of arrays.
    """
    return drop_duplicates(
        permutations_with_replacement(*arr, k=k, shape=shape))


def combinations(*arr, k=2, shape=None):
    """
    Take k-size combinations of elements from give
    list of arrays without replacement, i.e., each element
    in any combination only occurs once.
    """
    return drop_duplicates(permutations(*arr, k=k, shape=shape))


@nb.njit(parallel=True, fastmath=True)
def convert_to_ultrametric(values):
    """
    Fix triangular inequality within distance matrix (values)
    by converting to ultra-metric by ensuring the following
    condition: d_{ij} = min(d_{ij}, max(d_{ik}, d_{kj}))

    Parameters:
    ------------
    values: np.ndarray
        2D square distance matrix.

    Returns:
    --------
    np.ndarray
        Ultrametrified distance matrix.
    """
    assert len(values.shape) == 2 and values.shape[0] == values.shape[1],\
        "Values must be a 2D square matrix."
    result = np.full(values.shape, 1.)
    R = range(values.shape[0])
    for i in nb.prange(values.shape[0]):
        for j in range(i + 1, values.shape[0]):
            tmp = values[i, j]
            for k in R:
                tmp = min(tmp, max(values[i, k], values[j, k]))
            result[i, j] = tmp
            # result[i, j] = np.min(np.append(
            #     np.fmax(values[i, R], values[R, j]), values[i, j]))
            result[j, i] = result[i, j]
    return result
