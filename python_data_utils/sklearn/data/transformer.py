# coding: utf-8

"""
    description: Scikit-learn data transformation classes.
    author: Suraj Iyer
"""

__all__ = [
    'SelectColumns',
    'Normalize',
    'PipelineLabelEncoder',
    'OneHotEncoder']

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler


class SelectColumns(BaseEstimator, TransformerMixin):
    """
    A. Drop columns with 100% missing values and constant values.
    B. Rejects given columns if invert=False or keeps if invert=True.

    :param selected_cols: list of strings
        Columns to keeps
    :param invert: bool
        If False, selected columns behaves as it should. If True, it
        rejects the selected columns.
    """

    def __init__(self, selected_cols=[], invert=False):
        params = locals()
        del params['self']
        self.__dict__ = params

    def fit(self, X, y=None):
        rejected_cols = X.columns[X.isnull().all()]
        rejected_cols = rejected_cols.union(
            [col for col in X.columns if len(X[col].unique()) <= 1])
        if self.invert:
            self.keep_columns_ = X.columns\
                .difference(rejected_cols)\
                .difference(self.selected_cols)
        else:
            self.keep_columns_ = X.columns\
                .difference(rejected_cols)\
                .intersection(self.selected_cols)
        return self

    def transform(self, X):
        print('dropping columns')
        check_is_fitted(self, attributes=['keep_columns_'])
        cols = X.columns.intersection(self.keep_columns_)
        return X[cols]


class Normalize(BaseEstimator, TransformerMixin):
    """
    Normalize data
    """

    def __init__(self, columns=[], method="minmax"):
        params = locals()
        del params['self']
        self.__dict__ = params

    def fit(self, X, y=None):
        self.columns_ = X.columns.intersection(self.columns)
        if self.method == "zscore":
            self.method_ = StandardScaler().fit(X[self.columns_]).transform
        elif self.method == "minmax":
            self.method_ = MinMaxScaler().fit(X[self.columns_]).transform
        return self

    def transform(self, X):
        print('scaling features')
        check_is_fitted(self, attributes=['columns_', 'method_'])
        X.loc[:, self.columns_] = self.method_(X[self.columns_])
        return X


class PipelineLabelEncoder(BaseEstimator, TransformerMixin):
    """
    Label encoder with pre-defined set of labels provided.
    """

    def __init__(self, label_col="label"):
        self.label_col = label_col

    def fit(self, X, y=None):
        assert self.label_col in X.columns,\
            'Target labels columns must in X, not given as y.'
        self._le = LabelEncoder().fit(X[self.label_col])
        self.classes_ = self._le.classes_[np.argsort(self._le.transform(self._le.classes_))]
        return self

    def transform(self, X):
        print('encoding labels')
        if self.label_col in X.columns:
            X.loc[:, self.label_col] = self._le.transform(X[self.label_col])
        return X


class OneHotEncoder(BaseEstimator, TransformerMixin):
    """
    One hot encoder. ''Pipeline'' compatible.
    Attributes:
        columns : dict, optional (default=None)
            {
                key: str/int
                    column name,
                value: bool
                    if column values are ordered or not
            }
    """

    def __init__(self, columns, drop_first=False):
        assert isinstance(columns, dict) and len(columns) > 0,\
            'categorical should be a non-empty dict.'
        assert all(isinstance(c, bool) for c in columns.values()),\
            'categorical values must be boolean indicating ordered or not'
        params = locals()
        del params['self']
        self.__dict__ = params

    def fit(self, X, y=None, X_test=None):
        """Get all categories over both train and test data."""
        if X_test is None:
            import warnings
            warnings.warn('Test data might be needed to compute all categories per column')
            categories_per_column_ = {
                c: np.unique(X[c].dropna())
                for c in X.columns.intersection(self.columns.keys())}
        else:
            categories_per_column_ = {
                c: np.union1d(X[c].dropna(), X_test[c].dropna())
                for c in X.columns.intersection(self.columns.keys())}

        # Filter out variables with <= 2 levels
        self.categories_per_column_ = {
            k: v for k, v in categories_per_column_.items() if v.shape[0] > 2}

        return self

    def transform(self, X):
        print('one hot encoding')
        check_is_fitted(self, attributes=['categories_per_column_'])

        # convert non-categorical columns to categorical
        cols = self.categories_per_column_.keys()
        X.loc[:, cols] = X[cols].apply(lambda c: c.astype(
            'category',
            categories=self.categories_per_column_[c.name],
            ordered=self.columns[c.name]))

        X = pd.get_dummies(X, columns=cols, drop_first=self.drop_first)
        return X
