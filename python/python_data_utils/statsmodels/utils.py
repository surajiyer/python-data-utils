# coding: utf-8

"""
    description: Statsmodels utility functions
    author: Suraj Iyer
"""

__all__ = [
    'logit_evaluation_summary',
    'summary',
    'summary_to_latex'
]

import numpy as np
import pandas as pd
import statsmodels as sm
import re


def logit_evaluation_summary(results, labels, pos=1, neg=0):
    from .scoring import tjur_r2, squared_pearson_correlation_r2, sum_of_squares_r2
    return pd.DataFrame([
        ('Pseudo R-squared', results.prsquared),
        ('Tjur R-squared', tjur_r2(results, labels, pos, neg)),
        ('Squared Pearson Correlation R-squared', squared_pearson_correlation_r2(results, labels, pos, neg)),
        ('Sum-of-squares R-squared', sum_of_squares_r2(results, labels, pos, neg))])


def summary_params(
        result, significance_levels=np.array([0.1, 0.05, 0.01, 0.001]),
        margeff_kwargs={}):
    if all(isinstance(a, float) and 0. <= a <= 1. for a in significance_levels):
        raise ValueError('significance levels must be in the range [0, 1].')
    if all(a > b for a, b in zip(significance_levels, significance_levels[1:])):
        raise ValueError('significance_levels must be in strictly decreasing order.')

    params = sm.iolib.summary2.summary_params(result)

    # Logit model type
    if isinstance(
            result, sm.discrete.discrete_model.L1BinaryResultsWrapper):
        params.insert(1, 'OddsRat.', np.exp(params['Coef.']))
        params.insert(2, 'ME', result.get_margeff(**margeff_kwargs).summary_frame()['dy/dx'])

    params['Sign.'] = result.pvalues.apply(lambda x: '*' * np.sum(x < significance_levels))

    return params


def summary(
        result, float_format='%.4f', summary_kwargs={}, evaluation_kwargs={}):
    summary = result.summary2()
    summary.tables.pop(1), summary.settings.pop(1)
    summary.add_df(
        summary_params(result, **summary_kwargs),
        float_format=float_format)
    if 'significance_levels' not in summary_kwargs:
        summary.add_text(
            'Significance level: *: p < 0.1; **: p < 0.05; ***: p < 0.01; ****: p < 0.001')
    else:
        summary.add_text(
            'Significance level: ' + ' '.join(['*' * (i + 1) + f': p < {a};' for i, a in enumerate(summary_kwargs.alphas)]))
    if 'labels' in evaluation_kwargs:
        # Logit model type
        if isinstance(
                result, sm.discrete.discrete_model.L1BinaryResultsWrapper):
            summary.add_df(
                logit_evaluation_summary(result, **evaluation_kwargs),
                header=False, index=False, float_format=float_format)
    return summary


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
