# coding: utf-8

"""
    description: Statsmodels scoring functions
    author: Suraj Iyer
"""

__all__ = [
    'tjur_r2',
    'squared_pearson_correlation_r2',
    'sum_of_squares_r2']


import numpy as np


def tjur_r2(results, labels, pos=1, neg=0):
    """
    Compute the Tjur R^2 (Coefficient of Discrimination D) of logit model.
    D ≥ 0. D = 0 if and only if all estimated probabilities are equal — the model has no discriminatory power.
    D ≤ 1. D = 1 if and only if the observed and estimated probabilities are equal for all observations — the model
    discriminates perfectly.
    D will not always increase when predictors are added to the model.

    Params
    ------
    results : statsmodels.discrete.discrete_model.LogitResults
        Logit model results object

    labels : Iterable[int]
        Actual binary response values of all observarions

    pos : int
        Value for positive response case

    neg : int
        Value for negative response case

    Returns
    -------
    score : float
    """
    # convert log odds to estimated probabilities
    y_prob = np.exp(results.fittedvalues)
    y_prob = y_prob / (1 + y_prob)

    # calculate difference in mean estimated probability of each binary response
    return np.mean(y_prob.where(labels == pos)) -\
        np.mean(y_prob.where(labels == neg))


def squared_pearson_correlation_r2(results, labels, pos=1, neg=0):
    """
    Citation: MITTLBÖCK, M. and SCHEMPER, M. (1996), EXPLAINED VARIATION FOR LOGISTIC REGRESSION. Statist. Med., 15: 1987-1997.
    doi:10.1002/(SICI)1097-0258(19961015)15:19<1987::AID-SIM318>3.0.CO;2-9.
    https://pdfs.semanticscholar.org/39f1/82f6733f1b37c42539703c80e920f862ee6c.pdf

    Params
    ------
    results : statsmodels.discrete.discrete_model.LogitResults
        Logit model results object

    labels : Iterable[int]
        Actual binary response values of all observarions

    pos : int
        Value for positive response case

    neg : int
        Value for negative response case

    Returns
    -------
    score : float
    """
    # replace positive and negative values in labels (if needed)
    if pos != 1:
        labels[labels == pos] = 1
    if neg != 0:
        labels[labels == neg] = 0

    # convert log odds to estimated probabilities
    y_prob = np.exp(results.fittedvalues)
    y_prob = y_prob / (1 + y_prob)

    # calculate the numerator of the formula
    n = len(labels)
    pos_prob = np.sum(labels) / n
    top = np.sum(y_prob * labels) - (n * pos_prob**2)

    # calculate the denominator of the formula
    bottom = np.sqrt(n * pos_prob * (1 - pos_prob) * np.sum((y_prob - pos_prob) ** 2))

    return top / bottom


def sum_of_squares_r2(results, labels, pos=1, neg=0):
    """
    Citation: MITTLBÖCK, M. and SCHEMPER, M. (1996), EXPLAINED VARIATION FOR LOGISTIC REGRESSION. Statist. Med., 15: 1987-1997.
    doi:10.1002/(SICI)1097-0258(19961015)15:19<1987::AID-SIM318>3.0.CO;2-9.
    https://pdfs.semanticscholar.org/39f1/82f6733f1b37c42539703c80e920f862ee6c.pdf

    Params
    ------
    results : statsmodels.discrete.discrete_model.LogitResults
        Logit model results object

    labels : Iterable[int]
        Actual binary response values of all observarions

    pos : int
        Value for positive response case

    neg : int
        Value for negative response case

    Returns
    -------
    score : float
    """
    # replace positive and negative values in labels (if needed)
    if pos != 1:
        labels[labels == pos] = 1
    if neg != 0:
        labels[labels == neg] = 0

    # convert log odds to estimated probabilities
    y_prob = np.exp(results.fittedvalues)
    y_prob = y_prob / (1 + y_prob)

    # calculate the numerator of the formula
    n = len(labels)
    pos_prob = np.sum(labels) / n
    top = 2 * np.sum(y_prob * labels) - np.sum(y_prob**2) - (n * pos_prob**2)

    # calculate the denominator of the formula
    bottom = n * pos_prob * (1 - pos_prob)

    return top / bottom
