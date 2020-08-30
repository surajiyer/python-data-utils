# coding: utf-8

"""
    description: Scikit-learn data utility functions and classes
    author: Suraj Iyer
"""

__all__ = ['mahalanobis_pca_outliers']


import numpy as np


def mahalanobis_pca_outliers(X, n_components=2, threshold=2, plot=False):
    """
    Compute PCA on X, then compute the malanobis distance
    of all data points from the PCA components.

    Params
    ------
    X: data

    n_components: int (default=2)
        Number of PCA components to use to calculate the distance

    threshold: float (default=2)
        If None, returns the unaltered distance values.
        If float, output is binarized to True (Outlier)
        or False (Not outlier) based on threshold * stddev
        from the mean distance.

    plot: bool (default=False)
        If True, displays a 2D plot of the points colored by
        their distance

    Returns
    -------
    m: np.ndarray
        Distance values. len(m) == len(X).

    Usage
    -----
    >>> X = np.array([[-1, -1], [-2, -1], [-3, -2], [0, 0], [-20, 50], [3, 5]])
    >>> m = mahalanobis_pca_outliers(X)
    >>> m.shape[0] == 6
    True
    >>> print(m)
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.covariance import EmpiricalCovariance, MinCovDet
    import matplotlib.pyplot as plt

    # Define the PCA object
    pca = PCA()

    # Run PCA on scaled data and obtain the scores array
    T = pca.fit_transform(StandardScaler().fit_transform(X))

    # fit a Minimum Covariance Determinant (MCD) robust estimator to data
    robust_cov = MinCovDet().fit(T[:,:n_components])

    # Get the Mahalanobis distance
    md = robust_cov.mahalanobis(T[:,:n_components])

    # plot
    if plot:
        colors = [plt.cm.jet(float(i) / max(md)) for i in md]
        fig = plt.figure(figsize=(8,6))
        with plt.style.context(('ggplot')):
            plt.scatter(T[:, 0], T[:, 1], c=colors, edgecolors='k', s=60)
            plt.xlabel('PC1')
            plt.ylabel('PC2')
            plt.xlim((-60, 60))
            plt.ylim((-60, 60))
            plt.title('Score Plot')
        plt.show()

    if threshold:
        std = np.std(md)
        m = np.mean(md)
        k = threshold * std
        up, lo = m + k, m - k
        return np.logical_or(md >= up, md <= lo)

    return md


if __name__ == '__main__':
    import doctest
    doctest.testmod()
