# coding: utf-8

"""
    description: Scikit-learn scoring functions
    author: Suraj Iyer
"""

__all__ = [
    'apk',
    'mapk',
    'brier_score',
    'SCORERS']

import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer
from typing import Iterable


def apk(y_true: Iterable, y_pred: Iterable, k: int = 10) -> float:
    """
    Computes the average precision at k between two lists of items.
    Source: https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/average_precision.py

    Parameters
    ----------
    y_true : Iterable
        A list of elements that are to be predicted (order doesn't matter)
    y_pred : Iterable
        A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
        The average precision at k over the input lists
    """
    if len(y_pred) > k:
        y_pred = y_pred[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(y_pred):
        if p in y_true and p not in y_pred[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    if y_true is None:
        return 0.0

    return score / min(len(y_true), k)


def mapk(y_true: Iterable, y_pred: Iterable, k: int = 10) -> float:
    """
    Computes the mean average precision at k between two lists of items.
    Source: https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/average_precision.py

    Parameters
    ----------
    y_true : Iterable
        A list of lists of elements that are to be predicted
        (order doesn't matter in the lists)
    y_pred : Iterable
        A list of lists of predicted elements (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
        The mean average precision at k over the input lists
    """
    return np.mean([apk(a, p, k) for a, p in zip(y_true, y_pred)])


def brier_score(
        y_true: Iterable, y_prob: Iterable, labels: Iterable = None) -> float:
    """
    Brier loss for multi-class classification
    """
    y_true = y_true.values if isinstance(y_true, pd.Series) else y_true
    n_classes = len(labels) if labels is not None else y_true.max() + 1
    y_ohe = np.zeros((y_true.size, n_classes))
    y_ohe[np.arange(y_true.size), y_true] = 1
    inside_sum = np.sum([
        (fo - y_ohe[i]) ** 2 for i, fo in enumerate(y_prob)], axis=1)
    return np.average(inside_sum)


SCORERS = dict(
    apk_scorer=make_scorer(
        apk, greater_is_better=True, needs_proba=False),
    mapk_scorer=make_scorer(
        mapk, greater_is_better=True, needs_proba=False),
    brier_scorer=make_scorer(
        brier_score, greater_is_better=False, needs_proba=True)
)
