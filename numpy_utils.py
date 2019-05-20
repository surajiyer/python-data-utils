# coding: utf-8

"""
    description: Numpy utility functions and classes
    author: Suraj Iyer
"""

from __future__ import print_function

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
