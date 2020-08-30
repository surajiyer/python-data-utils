# coding: utf-8

"""
    description: Scikit-learn validation functions
    author: Suraj Iyer
"""

__all__ = [
    'CV',
    'cross_validate',
    'visualize_RF_feature_importances',
    'plot_roc',
    'plot_clusters'
]

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate as _cross_validate
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_context('poster')
sns.set_color_codes()


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
    is_multimetric = not (callable(scorer) or isinstance(scorer, str))
    scores = _cross_validate(
        clf, X, y, scoring=scorer, cv=cv, return_train_score=True,
        fit_params=fit_params)
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
            print(f"{score_type} scores ({metric}):")
            print(f"Mean: {s.mean()} | Std: {s.std()} | Min: {s.min()} | Max: {s.max()}")
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
    plt.bar(
        range(10), importances[indices],
        yerr=std[indices], color="r", align="center")
    plt.xticks(range(10), indices)
    plt.xlim([-1, 10])
    plt.show()


def plot_roc(X, y, clf_class, n_cv=5, **kwargs):
    kf = StratifiedKFold(len(y), n_splits=n_cv, shuffle=True)
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
        plt.plot(fpr, tpr, lw=1, label=f'ROC fold {i} (area = {roc_auc:.2f})')
    mean_tpr /= len(kf)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(
        mean_fpr, mean_tpr, 'k--',
        label=f'Mean ROC (area = {mean_auc:.2f})', lw=2)
    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Random')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()


def plot_clusters(data, algorithm, args, kwds):
    import time
    start_time = time.time()
    labels = algorithm(*args, **kwds).fit_predict(data)
    end_time = time.time()
    palette = sns.color_palette('deep', np.unique(labels).max() + 1)
    colors = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in labels]
    plot_kwds = {'alpha': 0.25, 's': 80, 'linewidths': 0}
    plt.scatter(data.T[0], data.T[1], c=colors, **plot_kwds)
    frame = plt.gca()
    frame.axes.get_xaxis().set_visible(False)
    frame.axes.get_yaxis().set_visible(False)
    plt.title(f'Clusters found by {str(algorithm.__name__)}', fontsize=24)
    plt.text(
        -0.5, 0.7,
        f'Clustering took {(end_time - start_time):.2f} s', fontsize=14)
    plt.show()
