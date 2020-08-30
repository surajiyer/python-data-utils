# coding: utf-8

"""
    description: Numpy utility functions and classes
    author: Suraj Iyer
"""

__all__ = [
    'create_upper_matrix',
    'symmetrize',
    'create_symmetric_matrix',
    'pairwise_difference',
    'is_pos_def',
    'rowwise_dissimilarity',
    'rowwise_cosine_similarity',
    'filter_rows_with_unique_values_only',
    'unique_per_row',
    'drop_duplicates',
    'permutations_with_replacement',
    'permutations',
    'combinations_with_replacement',
    'combinations',
    'convert_to_ultrametric'
]

import numpy as np
import warnings

try:
    import numba as nb
except (ImportError, ModuleNotFoundError):
    warnings.warn("Numba is not installed. Continuing without it.")

    class Object:
        jit = lambda x, *_, **__: x
        njit = lambda x, *_, **__: x
        prange = range
    nb = Object()

def create_upper_matrix(array):
    """
    Create an upper triangular matrix with any input array.

    Params
    ------
    array: array-like of numbers

    Returns
    -------
    upper : np.ndarray
        upper triangular matrix.

    Usage
    -----
    >>> create_upper_matrix([1,2,3])
    [[1, 2], [0, 3]]
    >>> create_upper_matrix([1,2,3,4,5,6])
    [[1, 2, 3], [0, 4, 5], [0, 0, 6]]
    >>> create_upper_matrix([1,2,3,4])
    [[1, 2, 3], [0, 4, 0], [0, 0, 0]]
    """
    def closest_inverse_sum2n(n):
        for i in range(1, n + 1):
            sumx = sum(range(i + 1))
            if sumx >= n:
                return i, sumx

    n = len(array)
    size, length = closest_inverse_sum2n(n)
    upper = np.zeros((size, size))
    upper[np.triu_indices(size, 0)] = np.append(
        array, [0] * (length - len(array)))
    return upper


def symmetrize(a):
    """
    Input 2D array and get a symmetric 2D matrix.
    """
    return a + a.T - np.diag(a.diagonal())


def create_symmetric_matrix(array):
    """
    Create a symmetric matrix with given input array.
    """
    return symmetrize(create_upper_matrix(array))


def pairwise_difference(array):
    """
    Take difference between every element pair.
    """
    return array - array[:, None]


def is_pos_def(array):
    """
    Is matrix :array: positive definite?
    """
    if np.allclose(array, array.T):
        try:
            np.linalg.cholesky(array)
            return True
        except np.linalg.LinAlgError:
            return False
    else:
        return False


def rowwise_dissimilarity(array):
    """
    Compare every row with each other row and count number
    of differences along column axis per row pairs.

    Usage
    -----
    >>> rowwise_dissimilarity([[1, 2, 3], [1, 3, 1], [2, 2, 2]])
    [[0, 2, 2], [2, 0, 3], [2, 3, 0]]
    """
    return np.sum(array != array[:, None], axis=-1)


def rowwise_cosine_similarity(array):
    """
    Using every pair of rows in :array: as input, compute
    pairwise cosine similarity between each row.
    URL: https://stackoverflow.com/questions/41905029/create-cosine-similarity-matrix-numpy
    """
    norm = (array * array).sum(0, keepdims=True) ** .5
    array = array / norm
    return (array.T @ array)


def filter_rows_with_unique_values_only(array):
    """
    Keep rows containing all unique elements only.
    URL: https://stackoverflow.com/questions/26958233/numpy-row-wise-unique-elements/
    """
    k = array.shape[1]
    for i in range(k - 1):
        array = array[(
            array[:, i, None] != array[:, range(i + 1, k)]).all(axis=1)]
    return array


def unique_per_row(array):
    """
    Count number of unique elements per row.
    URL: https://stackoverflow.com/questions/26958233/numpy-row-wise-unique-elements/
    """
    weight = 1j * np.linspace(
        0, array.shape[1], array.shape[0], endpoint=False)
    b = array + weight[:, None]
    _, ind, cnt = np.unique(b, return_index=True, return_counts=True)
    b = np.zeros_like(array)
    np.put(b, ind, cnt)
    return b


def drop_duplicates(array, ignore_order=False):
    """
    Drop duplicate rows.

    Params
    ------
    array : np.ndarray
        2D rectangular matrix

    ignore_order : bool
        If true, ignore order of elements in each row
        when computing uniqueness, else different
        orders of same elements will be treated as
        unique rows

    Returns
    ---------
    array with duplicate rows removed.
    """
    val = np.sort(array, axis=1) if ignore_order else array
    _, idx = np.unique(val, return_index=True, axis=0)
    return array[idx]


def permutations_with_replacement(*arr, k=2, shape=None):
    """
    Take k-size combinations of elements from given
    list of arrays.
    """
    if shape:
        arr = (range(s) for s in shape)
        k = len(shape)
    return np.array(np.meshgrid(*arr)).T.reshape(-1, k)


def permutations(*arrays, k=2, shape=None):
    """
    Take k-size permutations of elements from give
    list of arrays without replacement, i.e., each element
    in any permutation only occurs once.
    """
    if shape:
        arrays = (range(s) for s in shape)
        shape = None
    array = permutations_with_replacement(*arrays, k=k, shape=shape)
    array = filter_rows_with_unique_values_only(array)
    return array


def combinations_with_replacement(*arrays, k=2, shape=None):
    """
    Take k-size combinations of elements from given
    list of arrays.
    """
    return drop_duplicates(
        permutations_with_replacement(*arrays, k=k, shape=shape))


def combinations(*arrays, k=2, shape=None):
    """
    Take k-size combinations of elements from give
    list of arrays without replacement, i.e., each element
    in any combination only occurs once.
    """
    return drop_duplicates(permutations(*arrays, k=k, shape=shape))


@nb.njit(parallel=True, fastmath=True)
def convert_to_ultrametric(array):
    """
    Fix triangular inequality within 2D distance matrix (array)
    by converting to ultra-metric by ensuring the following
    condition: `d_{ij} = min(d_{ij}, max(d_{ik}, d_{kj}))`
    """
    array = np.atleast_2d(array)
    result = np.full(array.shape, 1.)
    for i in nb.prange(array.shape[0]):
        for j in range(i + 1, array.shape[0]):
            result[i, j] = result[j, i] = min(np.min(
                np.fmax(array[i], array[j])), array[i, j])
    return result


if __name__ == '__main__':
    import doctest
    doctest.testmod()
