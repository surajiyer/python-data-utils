# coding: utf-8

"""
    description: Pandas utility functions and classes
    author: Suraj Iyer
"""

from __future__ import print_function

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import statsmodels as statsm
from statsmodels.graphics.gofplots import ProbPlot
import re

plt.style.use('seaborn')  # pretty matplotlib plots


def residual_plot(results, y, n_annotate=3):
    # fitted values (need a constant term for intercept)
    model_fitted_y = results.fittedvalues

    # model residuals
    if isinstance(results, statsm.discrete.discrete_model.L1BinaryResultsWrapper):
        residuals = results.resid_dev
    elif isinstance(results, statsm.regression.linear_model.RegressionResultsWrapper):
        residuals = results.resid
    else:
        raise NotImplementedError('Model results for this type of model: {} is not supported'.format(type(results)))

    # absolute residuals
    model_abs_resid = np.abs(residuals)

    fig = plt.figure()
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
    Anything outside the group and outside “Cook’s Distance” lines, may have an influential effect on model fit.

    :param results: statsmodel
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


def tjur_r2(results, labels, pos=1, neg=0):
    """
    Compute the Tjur R^2 (Coefficient of Discrimination D) of logit model.
    D ≥ 0. D = 0 if and only if all estimated probabilities are equal — the model has no discriminatory power.
    D ≤ 1. D = 1 if and only if the observed and estimated probabilities are equal for all observations — the model
    discriminates perfectly.
    D will not always increase when predictors are added to the model.

    :param results: Logit model results object
    :param labels: Actual binary response values of all observarions.
    :param pos: Value for positive response case.
    :param neg: Value for negative response case.
    """
    # convert log odds to estimated probabilities
    y_prob = np.exp(results.fittedvalues)
    y_prob = y_prob / (1 + y_prob)

    # calculate difference in mean estimated probability of each binary response.
    y = np.mean(y_prob.where(labels == pos)) - np.mean(y_prob.where(labels == neg))
    return y


def squared_pearson_correlation_r2(results, labels, pos=1, neg=0):
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
    y_prob = np.exp(results.fittedvalues)
    y_prob = y_prob / (1 + y_prob)

    # calculate the numerator of the formula
    n = len(labels)
    pos_prob = np.sum(labels) / n
    top = np.sum(y_prob * labels) - (n * pos_prob**2)

    # calculate the denominator of the formula
    bottom = np.sqrt(n * pos_prob * (1 - pos_prob) * np.sum((y_prob - pos_prob)**2))

    return top / bottom


def sum_of_squares_r2(results, labels, pos=1, neg=0):
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
    y_prob = np.exp(results.fittedvalues)
    y_prob = y_prob / (1 + y_prob)

    # calculate the numerator of the formula
    n = len(labels)
    pos_prob = np.sum(labels) / n
    top = 2 * np.sum(y_prob * labels) - np.sum(y_prob**2) - (n * pos_prob**2)

    # calculate the denominator of the formula
    bottom = n * pos_prob * (1 - pos_prob)

    return top / bottom


def logit_evaluation_summary(results, labels, pos=1, neg=0):
    return pd.DataFrame([
        ('Pseudo R-squared', results.prsquared),
        ('Tjur R-squared', tjur_r2(results, labels, pos, neg)),
        ('Squared Pearson Correlation R-squared', squared_pearson_correlation_r2(results, labels, pos, neg)),
        ('Sum-of-squares R-squared', sum_of_squares_r2(results, labels, pos, neg))])


def summary_params(result, significance_levels=np.array([0.1, 0.05, 0.01, 0.001]), margeff_kwargs={}):
    assert all(isinstance(a, float) and 0. <= a <= 1. for a in significance_levels), 'ValueError: significance levels must be in the range [0, 1].'
    assert all(a > b for a, b in zip(significance_levels, significance_levels[1:])), 'significance_levels must be in strictly decreasing order.'

    params = statsm.iolib.summary2.summary_params(result)

    # Logit model type
    if isinstance(result, statsm.discrete.discrete_model.L1BinaryResultsWrapper):
        params.insert(1, 'OddsRat.', np.exp(params['Coef.']))
        params.insert(2, 'ME', result.get_margeff(**margeff_kwargs).summary_frame()['dy/dx'])

    params['Sign.'] = result.pvalues.apply(lambda x: '*' * np.sum(x < significance_levels))

    return params


def summary(result, float_format='%.4f', summary_kwargs={}, evaluation_kwargs={}):
    summary = result.summary2()
    summary.tables.pop(1), summary.settings.pop(1)
    summary.add_df(summary_params(result, **summary_kwargs), float_format=float_format)
    if 'significance_levels' not in summary_kwargs:
        summary.add_text('Significance level: *: p < 0.1; **: p < 0.05; ***: p < 0.01; ****: p < 0.001')
    else:
        summary.add_text(
            'Significance level: ' + ' '.join(['*' * (i + 1) + ': p < {};'.format(a) for i, a in enumerate(summary_kwargs.alphas)]))
    if 'labels' in evaluation_kwargs:
        # Logit model type
        if isinstance(result, statsm.discrete.discrete_model.L1BinaryResultsWrapper):
            summary.add_df(
                logit_evaluation_summary(result, **evaluation_kwargs),
                header=False, index=False, float_format=float_format)
    return summary


def plot_coefficients(model, ci=95):
    """
    Plots coefficients and their confidence intervals for a statsmodels OLS/Logit
    model. Based on (but heavily modified and simplified)
    seaborn's now deprecated coefplot.
    See https://github.com/mwaskom/seaborn/blob/master/seaborn/regression.py

    Args:
        model: statsmodels OLS model
            model whose params and confidence intervals to plot
        ci: float, optional
            size of confidence intervals

    Returns:

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
    Plots coefficients and their confidence intervals for a statsmodels logit
    model. Based on (but heavily modified and simplified)
    seaborn's now deprecated coefplot.
    See https://github.com/mwaskom/seaborn/blob/master/seaborn/regression.py

    Args:
        results: statsmodels OLS results
            model results whose params and confidence intervals to plot
        ci: float, optional
            size of confidence intervals for marginal effects

    Returns:

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


def summary_to_latex(results):
    stuff = results.as_latex()
    caption = re.search(r'(\\caption.+)\n', stuff).group(0)
    replacements = [
        # Remove the table environment
        (r'\\begin{table}\n', ''),
        (r'\\end{table}\n', ''),
        (r'\\caption.+\n', ''),
        # Replace it with separate environment for each tabular
        (r'(\\begin{center})', r'\\begin{{table}}\n{}\1'.format(caption)),
        (r'(\\end{center})', r'\1\n\\end{table}'),
        # Remove the bizzare placement of \hline between tables
        (r'\\end{table}\n\\hline\n\\begin{table}', r'\\end{table}\n\\begin{table}'),
        # Convert summary results table to longtable
        (r'\\begin{tabular}({\w+}\n\\hline\n\s+&\s+Coef.)', r'\\begin{longtable}\1'),
        (r'(?s)(\\begin{longtable}.*?)(?=\\end{table})', r'\1\\end{longtable}'),
        (r'(\\end{longtable})\\end{table}', r'\1'),
        (r'(?s)(\\end{table}\n\\begin{table}.*?)(?=\\begin{longtable})', r''),
        (r'(?s)\\end{tabular}(((?!\\end{tabular}).)*)(?=\\end{longtable})', r''),
        (r'(\\begin{longtable}{\w+})', r'\1\n{}'.format(caption)),
        # Add a resizebox to fit table to page margins
        #     (r'\\begin{tabular}', r'\\resizebox{\\textwidth}{!}{%\n\\begin{tabular}'),
        #     (r'\\end{tabular}', r'\\end{tabular}}'),
        # Merge the table headers and model summary results into a single table
        (r'\\end{tabular}\n\\begin{tabular}{\w+}\n(\w+)', r'\1'),
        #     (r'lccccccccc', r'lrrrrrrrrr'),
        # Using p{8cm} instead of l adds text wrapping for long variable names
        (r'(\\begin{longtable}){l(\w+)}', r'\1{p{8cm}\2}'),
        # Make double \hlines into single
        (r'\\hline\n\\hline', r'\\hline'),
        (r'\\hline(\nIntercept)', r'\\hline\n\\endhead\n\\hline\n\\multicolumn{10}{r}{\\textit{Continued on next page}} \\\\\\endfoot\n\\hline\n\\endlastfoot\1'),
        (r'\\end{table}\n\\end{table}', r'\\end{table}'),
    ]

    for old, new in replacements:
        stuff = re.sub(old, new, stuff)

    return stuff