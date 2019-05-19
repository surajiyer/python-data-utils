# coding: utf-8

"""
    description: Scikit-learn utility functions and classes
    author: Suraj Iyer
"""

from __future__ import print_function

import six
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import linear_model
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate as _cross_validate
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.metrics import make_scorer, roc_curve, auc


###############################################################################
# SKLEARN GENERAL FUNCTIONS
###############################################################################


def get_estimator_name(clf):
    """
    Extract the estimator name from the the estimator object {clf}
    :param clf: estimator object
    :return: string name
    """
    return str(type(clf)).split('.')[-1].replace("'>", "")


###############################################################################
# SKLEARN CUSTOM ESTIMATORS
###############################################################################


class LinearRegression(linear_model.LinearRegression):
    """
    LinearRegression class after sklearn's, but calculate t-statistics
    and p-values for model coefficients (betas).
    Additional attributes available after .fit()
    are `t` and `p` which are of the shape (y.shape[1], X.shape[1])
    which is (n_features, n_coefs)
    This class sets the intercept to 0 by default, since usually we include it
    in X.
    """

    def __init__(self, *args, **kwargs):
        if "fit_intercept" not in kwargs:
            kwargs['fit_intercept'] = False
        super(LinearRegression, self).__init__(*args, **kwargs)

    def fit(self, X, y, n_jobs=1):
        self = super(LinearRegression, self).fit(X, y, n_jobs)

        sse = np.sum((self.predict(X) - y) ** 2, axis=0) / float(X.shape[0] - X.shape[1])
        se = np.array([np.sqrt(np.diagonal(sse[i] * np.linalg.inv(np.dot(X.T, X)))) for i in range(sse.shape[0])])

        self.t = self.coef_ / se
        self.p = 2 * (1 - stats.t.cdf(np.abs(self.t), y.shape[0] - X.shape[1]))
        return self


###############################################################################
# SKLEARN EVALUATION FUNCTIONS
###############################################################################


def CV(n_splits=5):
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)


def cross_validate(clf, X, y, scorer, cv, fit_params=None):
    """
    Nested cross-validation. Prints average, standard deviation, minimum and maximum
    training and test scores for each metric.
    Attributes:
        clf: estimator obj
        X: pandas.DataFrame
            training data features
        y: pandas.Series
            training data target labels
        scorer: str or list of str
            List of scoring metrics to evaluate in the cv.
        cv: int or obj
            cross validation parameter.
            if integer given, it is the number of outer folds of the cv.
            if obj given, number of outer folds determined by obj.
        fit_params: dict, optional (default=None)
            Parameters to pass to 'fit' method.
    """
    is_multimetric = not (callable(scorer) or isinstance(scorer, six.string_types))
    scores = _cross_validate(clf, X, y, scoring=scorer, cv=cv, return_train_score=True, fit_params=fit_params)
    sign = 1

    if callable(scorer):
        if hasattr(scorer, '_sign'):
            sign = scorer._sign
        scorer = scorer._score_func.__name__

    if not is_multimetric:
        train_score = scores.pop("train_score", None)
        if train_score is not None:
            scores['train_%s' % scorer] = train_score * sign
        scores['test_%s' % scorer] = scores.pop("test_score") * sign
        scorer = [scorer]

    for metric in scorer:
        for score_type in ('train', 'test'):
            s = scores['{0}_{1}'.format(score_type, metric)]
            print("{} scores ({}):".format(score_type, metric))
            print("Mean: {} | Std: {} | Min: {} | Max: {}".format(s.mean(), s.std(), s.min(), s.max()))
        print("")
    return scores


def visualize_RF_feature_importances(forest_model, features, k_features=10):
    """
    :param forest_model: RandomForest model object
    :param k_features: int, 1 <= k_features <= n (n = total number of features)
        Top-k features will be displayed.
    """
    importances = forest_model.feature_importances_[:10]
    std = np.std([tree.feature_importances_ for tree in forest_model.estimators_], axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")
    for f in range(10):
        print("%d. %s (%f)" % (f + 1, features[f], importances[indices[f]]))

    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(10), importances[indices], yerr=std[indices], color="r", align="center")
    plt.xticks(range(10), indices)
    plt.xlim([-1, 10])
    plt.show()


def plot_roc(X, y, clf_class, n_cv=5, **kwargs):
    kf = StratifiedKFold(len(y), n_folds=n_cv, shuffle=True)
    y_prob = np.zeros((len(y), 2))
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    for i, (train_index, test_index) in enumerate(kf):
        X_train, X_test = X[train_index], X[test_index]
        y_train = y[train_index]
        clf = clf_class(**kwargs)
        clf.fit(X_train, y_train)
        # Predict probabilities, not classes
        y_prob[test_index] = clf.predict_proba(X_test)
        fpr, tpr, thresholds = roc_curve(y[test_index], y_prob[test_index, 1])
        mean_tpr += np.interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))
    mean_tpr /= len(kf)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, 'k--', label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)
    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Random')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()

###############################################################################
# SKLEARN PIPELINE FUNCTIONS
###############################################################################


class SelectColumns(BaseEstimator, TransformerMixin):
    """
    Drop columns with 100% missing values and constant values.
    Also, reject given columns if invert:False or only keep given columns if invert:True.
    ''Pipeline'' compatible.
    :param selected_cols: list of strings
        Columns to keeps
    :param invert: bool
        If False, selected columns behaves as it should. If True, it rejects the selected columns.
    """

    def __init__(self, selected_cols=[], invert=False):
        self.selected_cols = selected_cols
        self.invert = invert

    def fit(self, X, y=None):
        rejected_cols = X.columns[X.isnull().all()]
        rejected_cols = rejected_cols.union([col for col in X.columns if len(X[col].unique()) <= 1])
        if self.invert:
            self.keep_columns_ = X.columns.difference(rejected_cols).difference(self.selected_cols)
        else:
            self.keep_columns_ = X.columns.difference(rejected_cols).intersection(self.selected_cols)
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
        self.columns = columns
        self.method = method

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
        assert self.label_col in X.columns, 'Target labels columns must in X, not given as y.'
        self._le = LabelEncoder().fit(X[self.label_col])
        self.classes_ = self._le.classes_[np.argsort(self._le.transform(self._le.classes_))]
        return self

    def transform(self, X):
        print('encoding labels')
        if self.label_col in X.columns:
            X.loc[:, self.label_col] = self._le.transform(X[self.label_col])
        return X


###############################################################################
# SCORING FUNCTIONS
###############################################################################


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
        assert isinstance(columns, dict) and len(columns) > 0, 'categorical should be a non-empty dict.'
        assert all(isinstance(c, bool) for c in columns.values()), \
            'categorical values must be boolean indicating ordered or not'
        self.columns = columns
        self.drop_first = drop_first

    def fit(self, X, y=None, X_test=None):
        """Get all categories over both train and test data."""
        if X_test is None:
            import warnings
            warnings.warn('Test data might be needed to compute all categories per column')
            categories_per_column_ = {
                c: np.unique(X[c].dropna()) for c in X.columns.intersection(self.columns.keys())}
        else:
            categories_per_column_ = {
                c: np.union1d(X[c].dropna(), X_test[c].dropna())
                for c in X.columns.intersection(self.columns.keys())}

        # Filter out variables with <= 2 levels
        self.categories_per_column_ = {k: v for k, v in categories_per_column_.items() if v.shape[0] > 2}

        return self

    def transform(self, X):
        print('one hot encoding')
        from sklearn.utils.validation import check_is_fitted
        check_is_fitted(self, attributes=['categories_per_column_'])

        # convert non-categorical columns to categorical
        cols = self.categories_per_column_.keys()
        X.loc[:, cols] = X[cols].apply(lambda c: c.astype('category',
                                                          categories=self.categories_per_column_[c.name],
                                                          ordered=self.columns[c.name]))

        X = pd.get_dummies(X, columns=cols, drop_first=self.drop_first)
        return X


def apk(y_true, y_pred, k=10):
    """
    Source: https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/average_precision.py
    Computes the average precision at k.
    This function computes the average precision at k between two lists of items.
    Parameters
    ----------
    y_true : list
             A list of elements that are to be predicted (order doesn't matter)
    y_pred : list
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


def mapk(y_true, y_pred, k=10):
    """
    Source: https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/average_precision.py
    Computes the mean average precision at k.
    This function computes the mean average precision at k between two lists
    of lists of items.
    Parameters
    ----------
    y_true : list
        A list of lists of elements that are to be predicted (order doesn't matter in the lists)
    y_pred : list
        A list of lists of predicted elements (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    return np.mean([apk(a, p, k) for a, p in zip(y_true, y_pred)])


def brier_scorer(**kwargs):
    """
    Brier loss for multi-class classification
    """
    def brier_score(y_true, y_prob, labels=None):
        y_true = y_true.values if isinstance(y_true, pd.Series) else y_true
        n_classes = len(labels) if labels is not None else y_true.max() + 1
        y_ohe = np.zeros((y_true.size, n_classes))
        y_ohe[np.arange(y_true.size), y_true] = 1
        inside_sum = np.sum([(fo - y_ohe[i]) ** 2 for i, fo in enumerate(y_prob)], axis=1)
        return np.average(inside_sum)

    return make_scorer(brier_score, greater_is_better=False, needs_proba=True, **kwargs)


###############################################################################
# OTHER FUNCTIONS
###############################################################################


def display_topics(model, feature_names, no_top_words):
    """
    Display keywords associated with topics detected with topic modeling models, e.g., LDA, TruncatedSVD (LSA) etc.
    """
    for topic_idx, topic in enumerate(model.components_):
        print("Topic {}:".format(topic_idx))
        print(" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))
