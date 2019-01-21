# coding: utf-8

"""
    description: Pandas utility functions and classes
    author: Suraj Iyer
"""

from __future__ import print_function

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels as statsm
import statsmodels.formula.api as smf
from statsmodels.graphics.gofplots import ProbPlot

plt.style.use('seaborn') # pretty matplotlib plots
# plt.rc('font', size=14)
# plt.rc('figure', titlesize=18)
# plt.rc('axes', labelsize=15)
# plt.rc('axes', titlesize=18)
# plt.rc('figure', autolayout=True)


def residual_plot(y, model_results):    
    # fitted values (need a constant term for intercept)
    model_fitted_y = model_results.fittedvalues

    # model residuals
    if isinstance(model_results, statsm.discrete.discrete_model.L1BinaryResultsWrapper):
        model_resultsiduals = model_results.resid_dev
    elif isinstance(model, statsm.regression.linear_model.RegressionResultsWrapper):
        model_resultsiduals = model_results.resid
    else:
        raise NotImplementedError('Model results for this type of model: {} is not supported'.format(type(model)))

    # absolute residuals
    model_abs_resid = np.abs(model_resultsiduals)
    
    fig = plt.figure(1)
    fig.set_figheight(8)
    fig.set_figwidth(12)

    fig.axes[0] = sns.residplot(model_fitted_y, y, 
                                lowess=True, 
                                scatter_kws={'alpha': 0.5}, 
                                line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})

    fig.axes[0].set_title('Residuals vs Fitted')
    fig.axes[0].set_xlabel('Fitted values')
    fig.axes[0].set_ylabel('Residuals')

    # annotations
    abs_resid = pd.Series(model_abs_resid[::-1]).sort_values(ascending=False)
    abs_resid_top_3 = abs_resid[:3]

    for i in abs_resid_top_3.index:
        fig.axes[0].annotate(i, xy=(model_fitted_y[i], model_resultsiduals[i]))
    
    plt.close()
    return fig

def qq_plot(model_results):
    # normalized residuals
    model_norm_residuals = model_results.get_influence().resid_studentized_internal

    QQ = ProbPlot(model_norm_residuals)
    fig = QQ.qqplot(line='45', alpha=0.5, color='#4C72B0', lw=1)

    fig.set_figheight(8)
    fig.set_figwidth(12)

    fig.axes[0].set_title('Normal Q-Q')
    fig.axes[0].set_xlabel('Theoretical Quantiles')
    fig.axes[0].set_ylabel('Standardized Residuals');

    # annotations
    abs_norm_resid = np.flip(np.argsort(np.abs(model_norm_residuals)), 0)
    abs_norm_resid_top_3 = abs_norm_resid[:3]

    for r, i in enumerate(abs_norm_resid_top_3):
        fig.axes[0].annotate(i, xy=(np.flip(QQ.theoretical_quantiles, 0)[r], 
                                    model_norm_residuals[i]))
    
    plt.close()
    return fig

def scale_location_plot(df, model_results):
    # fitted values (need a constant term for intercept)
    model_fitted_y = model_results.fittedvalues
    
    # normalized residuals
    model_norm_residuals = model_results.get_influence().resid_studentized_internal

    # absolute squared normalized residuals
    model_norm_residuals_abs_sqrt = np.sqrt(np.abs(model_norm_residuals))

    fig = plt.figure(3)
    fig.set_figheight(8)
    fig.set_figwidth(12)

    plt.scatter(model_fitted_y, model_norm_residuals_abs_sqrt, alpha=0.5)
    sns.regplot(model_fitted_y, model_norm_residuals_abs_sqrt, 
                scatter=False, 
                ci=False, 
                lowess=True,
                line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})

    fig.axes[0].set_title('Scale-Location')
    fig.axes[0].set_xlabel('Fitted values')
    fig.axes[0].set_ylabel('$\sqrt{|Standardized Residuals|}$');

    # annotations
    abs_sq_norm_resid = np.flip(np.argsort(model_norm_residuals_abs_sqrt), 0)
    abs_sq_norm_resid_top_3 = abs_sq_norm_resid[:3]

    for i in abs_norm_resid_top_3:
        fig.axes[0].annotate(i, xy=(model_fitted_y[i], model_norm_residuals_abs_sqrt[i]))
    
    plt.close()
    return fig

def leverage_plot(model_results):
    """
    This plot shows if any outliers have influence over the regression fit. 
    Anything outside the group and outside “Cook’s Distance” lines, may have an influential effect on model fit.

    :param model_results: statsmodel
    """
    # normalized residuals
    model_norm_residuals = model_results.get_influence().resid_studentized_internal
    
    # leverage, from statsmodels internals
    model_leverage = model_results.get_influence().hat_matrix_diag

    # cook's distance, from statsmodels internals
    model_cooks = model_results.get_influence().cooks_distance[0]
    
    fig = plt.figure(4)
    fig.set_figheight(8)
    fig.set_figwidth(12)

    plt.scatter(model_leverage, model_norm_residuals, alpha=0.5)
    sns.regplot(model_leverage, model_norm_residuals, 
                scatter=False, 
                ci=False, 
                lowess=True,
                line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})

    fig.axes[0].set_xlim(0, 0.20)
    fig.axes[0].set_ylim(-3, 5)
    fig.axes[0].set_title('Residuals vs Leverage')
    fig.axes[0].set_xlabel('Leverage')
    fig.axes[0].set_ylabel('Standardized Residuals')

    # annotations
    leverage_top_3 = np.flip(np.argsort(model_cooks), 0)[:3]

    for i in leverage_top_3:
        fig.axes[0].annotate(i, xy=(model_leverage[i], model_norm_residuals[i]))

    # shenanigans for cook's distance contours
    def graph(formula, x_range, label=None):
        x = x_range
        y = formula(x)
        plt.plot(x, y, label=label, lw=1, ls='--', color='red')

    p = len(model_results.params) # number of model parameters

    graph(lambda x: np.sqrt((0.5 * p * (1 - x)) / x), 
          np.linspace(0.001, 0.200, 50), 
          'Cook\'s distance') # 0.5 line
    graph(lambda x: np.sqrt((1 * p * (1 - x)) / x), 
          np.linspace(0.001, 0.200, 50)) # 1 line
    plt.legend(loc='upper right')
    
    plt.close()
    return fig

def tjur_r2(model_results, labels, pos=1, neg=0):
    """
    Compute the Tjur R^2 (Coefficient of Discrimination D) of logit model.
    D ≥ 0. D = 0 if and only if all estimated probabilities are equal — the model has no discriminatory power.
    D ≤ 1. D = 1 if and only if the observed and estimated probabilities are equal for all observations — the model
    discriminates perfectly.
    D will not always increase when predictors are added to the model.

    :param model_results: Logit model results object
    :param labels: Actual binary response values of all observarions.
    :param pos: Value for positive response case.
    :param neg: Value for negative response case.
    """
    # convert log odds to estimated probabilities
    y_prob = np.exp(model_results.fittedvalues)
    y_prob = y_prob/(1+y_prob)
    
    # calculate difference in mean estimated probability of each binary response.
    y = np.mean(y_prob.where(labels == pos)) - np.mean(y_prob.where(labels == neg))
    return y

def squared_pearson_correlation_r2(model_results, labels, pos=1, neg=0):
    """
    Citation: MITTLBÖCK, M. and SCHEMPER, M. (1996), EXPLAINED VARIATION FOR LOGISTIC REGRESSION. Statist. Med., 15: 1987-1997. 
    doi:10.1002/(SICI)1097-0258(19961015)15:19<1987::AID-SIM318>3.0.CO;2-9.
    https://pdfs.semanticscholar.org/39f1/82f6733f1b37c42539703c80e920f862ee6c.pdf
    """
    # replace positive and negative values in labels (if needed)
    if pos != 1:
        labels[labels == pos] = 1
    if neg != 0:
        labels[labels == neg] = 0
    
    # convert log odds to estimated probabilities
    y_prob = np.exp(model_results.fittedvalues)
    y_prob = y_prob/(1+y_prob)
    
    # calculate the numerator of the formula
    n = len(labels)
    pos_prob = np.sum(labels) / n
    top = np.sum(y_prob * labels) - (n * pos_prob**2)
    
    # calculate the denominator of the formula
    bottom = np.sqrt(n * pos_prob * (1 - pos_prob) * np.sum((y_prob - pos_prob)**2))
    
    return top / bottom

def sum_of_squares_r2(model_results, labels, pos=1, neg=0):
    """
    Citation: MITTLBÖCK, M. and SCHEMPER, M. (1996), EXPLAINED VARIATION FOR LOGISTIC REGRESSION. Statist. Med., 15: 1987-1997. 
    doi:10.1002/(SICI)1097-0258(19961015)15:19<1987::AID-SIM318>3.0.CO;2-9.
    https://pdfs.semanticscholar.org/39f1/82f6733f1b37c42539703c80e920f862ee6c.pdf
    """
    # replace positive and negative values in labels (if needed)
    if pos != 1:
        labels[labels == pos] = 1
    if neg != 0:
        labels[labels == neg] = 0

    # convert log odds to estimated probabilities
    y_prob = np.exp(model_results.fittedvalues)
    y_prob = y_prob/(1+y_prob)
    
    # calculate the numerator of the formula
    n = len(labels)
    pos_prob = np.sum(labels) / n
    top = 2 * np.sum(y_prob * labels) - np.sum(y_prob**2) - (n * pos_prob**2)
    
    # calculate the denominator of the formula
    bottom = n * pos_prob * (1 - pos_prob)
    
    return top / bottom
