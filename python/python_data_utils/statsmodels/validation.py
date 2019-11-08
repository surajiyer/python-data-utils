# coding: utf-8

"""
    description: Statsmodels model validation functions
    author: Suraj Iyer
"""

__all__ = [
    'residual_plot',
    'qq_plot',
    'scale_location_plot',
    'leverage_plot',
    'plot_coefficients',
    'plot_logit_marginal_effects'
]

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels as sm
from statsmodels.graphics.gofplots import ProbPlot

plt.style.use('seaborn')  # pretty plots


def residual_plot(results, y, n_annotate=3):
    # fitted values (need a constant term for intercept)
    model_fitted_y = results.fittedvalues

    # model residuals
    if isinstance(
            results,
            sm.discrete.discrete_model.L1BinaryResultsWrapper):
        residuals = results.resid_dev
    elif isinstance(
            results,
            sm.regression.linear_model.RegressionResultsWrapper):
        residuals = results.resid
    else:
        raise NotImplementedError(
            f'Model results for this type of model: {type(results)} is not supported.')

    # absolute residuals
    model_abs_resid = np.abs(residuals)

    fig = plt.figure()
    fig.set_figheight(8)
    fig.set_figwidth(12)

    fig.axes[0] = sns.residplot(
        model_fitted_y, y,
        lowess=True,
        scatter_kws={'alpha': 0.5},
        line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})

    fig.axes[0].set_title('Residuals vs Fitted')
    fig.axes[0].set_xlabel('Fitted values')
    fig.axes[0].set_ylabel('Residuals')

    # annotations
    abs_resid = pd.Series(model_abs_resid[::-1]).sort_values(ascending=False)
    n_annotate = min(n_annotate, len(abs_resid))
    abs_resid_top_n = abs_resid[:n_annotate]

    for i in abs_resid_top_n.index:
        fig.axes[0].annotate(i, xy=(model_fitted_y[i], residuals[i]))

    plt.close()
    return fig


def qq_plot(results, n_annotate=3):
    # normalized residuals
    model_norm_residuals = results.get_influence().resid_studentized_internal

    QQ = ProbPlot(model_norm_residuals)
    fig = QQ.qqplot(line='45', alpha=0.5, color='#4C72B0', lw=1)

    fig.set_figheight(8)
    fig.set_figwidth(12)

    fig.axes[0].set_title('Normal Q-Q')
    fig.axes[0].set_xlabel('Theoretical Quantiles')
    fig.axes[0].set_ylabel('Standardized Residuals')

    # annotations
    abs_norm_resid = np.flip(np.argsort(np.abs(model_norm_residuals)), 0)
    n_annotate = min(n_annotate, len(abs_norm_resid))
    abs_norm_resid_top_n = abs_norm_resid[:n_annotate]

    for r, i in enumerate(abs_norm_resid_top_n):
        fig.axes[0].annotate(i, xy=(np.flip(QQ.theoretical_quantiles, 0)[r], model_norm_residuals[i]))

    plt.close()
    return fig


def scale_location_plot(results, n_annotate=3):
    # fitted values (need a constant term for intercept)
    model_fitted_y = results.fittedvalues

    # normalized residuals
    model_norm_residuals = results.get_influence().resid_studentized_internal

    # absolute squared normalized residuals
    model_norm_residuals_abs_sqrt = np.sqrt(np.abs(model_norm_residuals))

    fig = plt.figure()
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
    fig.axes[0].set_ylabel('$\sqrt{|Standardized Residuals|}$')

    # annotations
    abs_sq_norm_resid = np.flip(np.argsort(model_norm_residuals_abs_sqrt), 0)
    n_annotate = min(n_annotate, len(abs_sq_norm_resid))
    abs_sq_norm_resid_top_n = abs_sq_norm_resid[:n_annotate]

    for i in abs_sq_norm_resid_top_n:
        fig.axes[0].annotate(i, xy=(model_fitted_y[i], model_norm_residuals_abs_sqrt[i]))

    plt.close()
    return fig


def leverage_plot(results, n_annotate=3):
    """
    This plot shows if any outliers have influence over the regression fit.
    Anything outside the group and outside “Cook’s Distance” lines, may
    have an influential effect on model fit.
    """
    # normalized residuals
    model_norm_residuals = results.get_influence().resid_studentized_internal

    # leverage, from statsmodels internals
    model_leverage = results.get_influence().hat_matrix_diag

    # cook's distance, from statsmodels internals
    model_cooks = results.get_influence().cooks_distance[0]

    fig = plt.figure()
    fig.set_figheight(8)
    fig.set_figwidth(12)

    plt.scatter(model_leverage, model_norm_residuals, alpha=0.5)
    sns.regplot(
        model_leverage, model_norm_residuals,
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
    cooks_distance = np.flip(np.argsort(model_cooks), 0)
    n_annotate = min(n_annotate, len(cooks_distance))
    leverage_top_3 = cooks_distance[:n_annotate]

    for i in leverage_top_3:
        fig.axes[0].annotate(i, xy=(model_leverage[i], model_norm_residuals[i]))

    # shenanigans for cook's distance contours
    def graph(formula, x_range, label=None):
        x = x_range
        y = formula(x)
        fig.axes[0].plot(x, y, label=label, lw=1, ls='--', color='red')

    p = len(results.params)  # number of model parameters

    graph(lambda x: np.sqrt((0.5 * p * (1 - x)) / x),
          np.linspace(0.001, 0.200, 50),
          'Cook\'s distance')  # 0.5 line
    graph(lambda x: np.sqrt((1 * p * (1 - x)) / x),
          np.linspace(0.001, 0.200, 50))  # 1 line
    fig.axes[0].legend(loc='upper right')

    plt.close()
    return fig


def plot_coefficients(model, ci=95):
    """
    Plot coefficients and their confidence intervals for a statsmodels
    OLS/Logit model. Based on (but heavily modified and simplified)
    seaborn's now deprecated coefplot.

    URL: https://github.com/mwaskom/seaborn/blob/master/seaborn/regression.py

    :param model: model results
    :param ci: float, optional
        size of confidence intervals
    """
    # Get basic information to prepare the plot
    alpha = 1 - ci / 100
    coefs = model.params
    cis = model.conf_int(alpha)
    n_terms = len(coefs)

    # Figure out the dimensions of the plot
    h, w = mpl.rcParams["figure.figsize"]
    f, ax = plt.subplots(1, 1, figsize=(n_terms * (1 / 2), n_terms * (h / (4 * (n_terms / 5)))))
    for i, term in enumerate(coefs.index):
        low, high = cis.loc[term]
        ax.plot([low, high], [i, i], solid_capstyle="round", lw=2.5,
                color='black')
        ax.plot(coefs.loc[term], i, "o", ms=8, color='black')
    ax.set_ylim(-.5, n_terms - .5)
    ax.set_yticks(range(n_terms))
    coef_names = coefs.index.values
    ax.set_yticklabels(coef_names)
    plt.setp(ax.get_xticklabels(), rotation=90)


def plot_logit_marginal_effects(results, ci=95):
    """
    Plot coefficients and their confidence intervals for a statsmodels logit
    model. Based on (but heavily modified and simplified) seaborn's now
    deprecated coefplot.

    URL: https://github.com/mwaskom/seaborn/blob/master/seaborn/regression.py

    :param results: sm.discrete.discrete_model.L1BinaryResultsWrapper
        Logit model results
    :param ci: float, optional
        size of confidence intervals for marginal effects
    """
    # Get basic information to prepare the plot
    alpha = 1 - ci / 100
    margeff = results.get_margeff()
    cis = margeff.conf_int(alpha)
    coefs = results.get_margeff().margeff
    constant_cols_idx = np.argwhere(np.apply_along_axis(lambda x: np.unique(x).shape[0] == 1, 0, results.model.exog)).ravel()
    coef_names = np.delete(results.params.index, constant_cols_idx)
    n_terms = len(coefs)

    # Figure out the dimensions of the plot
    h, w = mpl.rcParams["figure.figsize"]
    f, ax = plt.subplots(1, 1, figsize=(n_terms * (1 / 2), n_terms * (h / (4 * (n_terms / 5)))))
    for i, term in enumerate(coef_names):
        low, high = cis[i]
        ax.plot([low, high], [i, i], solid_capstyle="round", lw=2.5,
                color='black')
        ax.plot(coefs[i], i, "o", ms=8, color='black')
    ax.set_ylim(-.5, n_terms - .5)
    ax.set_yticks(range(n_terms))
    ax.set_yticklabels(coef_names)
    plt.setp(ax.get_xticklabels(), rotation=90)
