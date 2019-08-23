# coding: utf-8

"""
    description: Pandas utility functions and classes
    author: Suraj Iyer
"""

from __future__ import print_function

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import six
# from imblearn.over_sampling import smote

plt.style.use('seaborn')  # pretty matplotlib plots
# plt.rc('font', size=12)
# plt.rc('figure', titlesize=18)
# plt.rc('axes', labelsize=15)
# plt.rc('axes', titlesize=18)
# plt.rc('figure', autolayout=True)


def df_mem_usage(df):
    """ Calculate memory usage of pandas dataframe """
    result = dict()

    if isinstance(df, pd.DataFrame):
        usage_b = df.memory_usage(deep=True).sum()
    else:  # we assume if not a df it's a series
        usage_b = df.memory_usage(deep=True)
    usage_mb = usage_b / 1024 ** 2  # convert bytes to megabytes
    result.update({"Total memory usage": "{:03.2f} MB".format(usage_mb)})

    # get average memory usage per datatype
    dtypes = [c.name for c in df.dtypes.unique()]
    result.update({
        'Average memory usage for {} columns'.format(c): None for c in dtypes})
    for dtype in dtypes:
        usage_b = df.select_dtypes(include=[dtype])\
            .memory_usage(deep=True).mean()
        result.update({
            "Average memory usage for {} columns".format(dtype):
            "{:03.2f} MB".format(usage_b / 1024 ** 2)})

    return result


def optimize_dataframe(df, categorical=[], always_positive_ints=[],
                       cat_nunique_ratio=.5, verbose=False):
    """
    Optimize the memory usage of the given dataframe by modifying data types.
    :param df: pd.DataFrame
    :param categorical: list of (str, bool) pairs, optional (default=[])
        List of categorical variables with boolean representing if
        they are ordered or not.
    :param always_positive_ints: list of str, optional (default=[])
        List of always positive INTEGER variables
    :param cat_nunique_ratio: 0.0 <= float <= 1.0, (default=0.5)
        Ratio of unique values to total values. Used for detecting
        categorical columns.
    :param verbose: bool, optional (default=False)
        Print before and after memory usage
    :return df: pd.DataFrame
    """
    cat = [] if len(categorical) == 0 else list(zip(*categorical))[0]

    def getCols(dtype):
        return df.columns[df.dtypes == dtype]\
            .difference(always_positive_ints).difference(cat)

    if verbose:
        print('Before:', df_mem_usage(df))

    # convert always positive columns to unsigned ints
    # nulls are represented as floats so we ignore them
    cols = df.columns[~df.isnull().any()].intersection(always_positive_ints)
    df.loc[:, cols] = df[cols].fillna(-1).astype(
        'int').apply(pd.to_numeric, downcast='unsigned')
    # Converting back to nan changes them back to float64 dtype
    # df.loc[(df[cols] == -1).any(axis=1), cols] = np.nan

    # downcast integer columns
    cols = getCols('int')
    df.loc[:, cols] = df[cols].apply(pd.to_numeric, downcast='integer')

    # downcast float columns
    cols = getCols('float')
    df.loc[:, cols] = df[cols].apply(pd.to_numeric, downcast='float')

    # convert object columns with less than {cat_nunique_ratio}%
    # unique values to categorical
    for col in getCols('object'):
        num_unique_values = len(df[col].unique())
        num_total_values = len(df[col].dropna())
        if num_unique_values / float(num_total_values) < cat_nunique_ratio:
            df.loc[:, col] = df[col].astype('category')

    # convert given columns to categorical
    for col, prop in categorical:
        if col in df.columns and df.dtypes[col].name != 'category':
            if isinstance(prop, (list, tuple)):
                df.loc[:, col] = df[col].astype(
                    pd.api.types.CategoricalDtype(
                        categories=prop[1], ordered=prop[0]))
            elif isinstance(prop, dict):
                df.loc[:, col] = df[col].astype(
                    pd.api.types.CategoricalDtype(
                        categories=prop['categories'],
                        ordered=prop['ordered']))
            elif isinstance(prop, bool):
                df.loc[:, col] = df[col].astype(
                    pd.api.types.CategoricalDtype(ordered=prop))
            else:
                raise ValueError(
                    'Categorical variable {} ill-specified.'.format(col))

    if verbose:
        print('After:', df_mem_usage(df))

    return df


def tsv_to_pandas_generator(file_path, target_label=None, chunksize=None):
    """
    Read data from TSV file as pandas DataFrame and yield them as chunks.
    :param file_path: str
        Path to tsv file
    :param target_label: str
        target label column name
    :param chunksize: int
        function yields dataframes in chunks as a generator function
    :return:
        X: pd.DataFrame
        y: pd.Series
            included if target label is given
    """
    assert isinstance(file_path, six.string_types) and len(file_path) > 0
    assert target_label is None or isinstance(target_label, six.string_types)
    assert isinstance(chunksize, int)

    for chunk in pd.read_csv(
            file_path, delimiter='\t', na_values=r'\N', chunksize=chunksize):
        if target_label:
            yield chunk.drop([target_label], axis=1), chunk[target_label]
        else:
            yield chunk


def tsv_to_pandas(file_path, target_label=None, memory_optimize=True,
                  categorical=[], always_positive=[], save_pickle_obj=False,
                  verbose=False):
    """
    Read data from TSV file as pandas DataFrame.
    :param file_path: str
        Path to tsv file
    :param target_label: str
        target label column name
    :param memory_optimize: bool, optional (default=True)
        Optimize the data types of the columns. Will take some more
        time to compute but will reduce memory usage of the dataframe.
    :param categorical: list of str, optional (default=[])
        List of categorical variables
    :param always_positive: list of str, optional (default=[])
        List of always positive INTEGER variables
    :param verbose: bool, optional (default=False)
        Output time to load the data and memory usage.
    :param save_pickle_obj: bool, str, optional (default=False)
        Save the final result as a pickle object. Save path can be given as a
        string. Otherwise, saved in the same directory as the input file path.
    :return:
        X: pd.DataFrame
        y: pd.Series
            included if target label is given
    """
    assert isinstance(file_path, six.string_types) and len(file_path) > 0
    assert target_label is None or isinstance(target_label, six.string_types)
    import dill
    import os

    if verbose:
        import time
        start_time = time.time()

    # Get number of rows in data
    with open(file_path, 'r') as f:
        filesize = sum(1 for line in f)

    # Initialize progress bar
    chunksize = 100000
    from progressbar.bar import ProgressBar
    progress = ProgressBar(max_value=int(filesize / chunksize)).start()

    data = tsv_to_pandas_generator(
        file_path, target_label, chunksize=chunksize)
    X_list, y_list = [], []

    # To save intermediate results
    if memory_optimize:
        results_dir = os.path.splitext(file_path)[0]

        # delete directory if it already exists
        import shutil
        shutil.rmtree(results_dir, ignore_errors=True)

        # make the results directory
        if results_dir:
            try:
                os.makedirs(results_dir)
            except OSError as e:
                import errno
                if e.errno != errno.EEXIST:
                    raise

    for i, X in enumerate(data):
        if target_label:
            X, y = X[0], X[1]

        # Memory compression
        if memory_optimize:
            if i == 0:
                always_positive = [
                    col for c0 in always_positive for col in X.columns
                    if c0 in col]

            X = optimize_dataframe(
                X, always_positive_ints=always_positive,
                categorical=categorical)

            # save intermediate results
            with open("{}/{}.pkl".format(results_dir, i), 'wb') as f:
                if target_label:
                    dill.dump(pd.concat([X, y], axis=1), f)
                else:
                    dill.dump(X, f)

        X_list.append(X)
        if target_label:
            y_list.append(y)

        # update progress bar
        progress.update(i)

    # concatenate the chunks
    X = pd.concat(X_list)
    if target_label:
        y = pd.concat(y_list)

    if verbose:
        print(df_mem_usage(X))
        print('Finished in {0:.1f} minutes'.format(
            (time.time() - start_time) / 60))

    # save the final result
    if save_pickle_obj:
        if isinstance(save_pickle_obj, six.string_types):
            save_path = save_pickle_obj

            # make the save directory
            if save_path:
                try:
                    os.makedirs(save_path)
                except OSError as e:
                    import errno
                    if e.errno != errno.EEXIST:
                        raise
        else:
            save_path = os.path.dirname(file_path)

        file_name = os.path.splitext(os.path.basename(file_path))[0]
        with open("{}/{}.pkl".format(save_path, file_name), 'wb') as f:
            if target_label:
                dill.dump(pd.concat([X, y], axis=1), f)
            else:
                dill.dump(X, f)

    # delete intermediate results if operation successful
    if memory_optimize:
        shutil.rmtree(results_dir, ignore_errors=True)

    return (X, y) if target_label else X


def pandas_to_tsv(df, save_file_path, index=False, mode='w', header=True):
    """
    Save pre-processed DataFrame as tsv file.
    :param df: pd.DataFrame
        DataFrame to save as tsv
    :param save_file_path: str
        File save path. if path does not exist, it will be
        created automatically.
    :param index: bool
        write the row names to output.
    :param mode: str
        file save mode; 'w' for write and 'a' for append to existing file.
    :param header: bool
        write the column names to output.
    """
    assert isinstance(save_file_path, six.string_types) and\
        len(save_file_path) > 0
    assert isinstance(df, pd.DataFrame) and not df.empty,\
        'df must be a non-empty pd.DataFrame'
    assert isinstance(mode, six.string_types)

    # make a new directory to save the file
    try:
        import os
        os.makedirs(os.path.split(save_file_path)[0])
    except OSError as e:
        import errno
        if e.errno != errno.EEXIST:
            raise

    # save the file
    df.to_csv(save_file_path, sep='\t', na_rep=r'\N',
              index=index, mode=mode, header=header)


def explode_horizontal(s):
    return s.apply(pd.Series)


def explode_multiple(df, lst_cols, fill_value=''):
    # make sure `lst_cols` is a list
    if lst_cols and not isinstance(lst_cols, list):
        lst_cols = [lst_cols]
    # all columns except `lst_cols`
    idx_cols = df.columns.difference(lst_cols)

    # calculate lengths of lists
    lens = df[lst_cols[0]].str.len()

    if (lens > 0).all():
        # ALL lists in cells aren't empty
        return pd.DataFrame({
            col: np.repeat(df[col].values, df[lst_cols[0]].str.len())
            for col in idx_cols
        }).assign(**{col: np.concatenate(df[col].values) for col in lst_cols}) \
          .loc[:, df.columns]
    else:
        # at least one list in cells is empty
        return pd.DataFrame({
            col: np.repeat(df[col].values, df[lst_cols[0]].str.len())
            for col in idx_cols
        }).assign(**{col: np.concatenate(df[col].values) for col in lst_cols}) \
          .append(df.loc[lens == 0, idx_cols]).fillna(fill_value) \
          .loc[:, df.columns]


def explode_horizontal_ohe(s):
    """
    Expands a column of lists to multiple one-hot
    encoded columns per unique element from all
    lists.

    :param s: pd.Series
        Series column of lists.
    :return df: pd.DataFrame
        Dataframe with multiple columns, one per unique item in all lists.
    """
    return pd.get_dummies(s.apply(pd.Series).stack()).sum(level=0)


def get_missingness(df):
    return [(c, df[c].isna().sum()) for c in df if df[c].isna().any()]


def get_missingness_perc(df):
    """
    Get a percentage of missing values per column in input dataframe.
    :param df: Input pandas dataframe
    :return missingness:
        Pandas dataframe with index as columns from df and values
        as missingness level of corresponding column.
    """
    missingness = pd.DataFrame(
        [(len(df[c]) - df[c].count()) * 100.00 / len(df[c]) for c in df],
        index=df.columns, columns=['Missingness %'])
    return missingness


def missingness_heatmap(df):
    import seaborn as sns
    return sns.heatmap(df.isnull(), cbar=False)


def correlations_to_column(df, col, jupyter_nb=False):
    """
    Get correlation of all columns to given column :col:.
    If :jupyter_nb: is true, get interactive widget slider to
    filter columns by spearman correlation strength.

    :param df: pd.DataFrame
    :col: str
        Label column for which we find correlation
        to all other columns.
    :return corr: pd.DataFrame
        Dataframe containing 2 columns: one for correlation values obtained
        using Pearson Rho test and one with Spearman Rank test.
    :return corr_slider: widgets.FloatSlider
        If :jupyter_nb: is true, Jupyter  ranging from 0.0 to 1.0
        over spearman correlation strength to filter columns interactively.
    """
    # Get Pearson correlation - to describe extent of linear correlation
    # with label
    pearson = df.corr(method="pearson")[col].rename("pearson")

    # Get Spearman rank correlation - to describe extent of any
    # monotonic relationship with label
    spearman = df.corr(method="spearman")[col].rename("spearman")

    corr = pd.concat([pearson, spearman], axis=1)

    if jupyter_nb:
        from IPython.display import display
        import ipywidgets as widgets
        from ipywidgets import interactive

        def view_correlations(corr_strength=0.0):
            if corr_strength == 0.0:
                x = corr
            else:
                x = corr.where(
                    (abs(corr.spearman) > corr_strength) &
                    (corr.spearman != 1)).dropna()
            print('N: {}'.format(x.shape[0]))
            display(x)
            return x

        corr_slider = widgets.FloatSlider(
            value=0.0, min=0.0, max=1.0, step=0.0001)
        w = interactive(view_correlations, corr_strength=corr_slider)

        return corr, w

    return corr


def feature_corr_matrix(df, size=10, method="spearman"):
    """
    Function plots a graphical correlation matrix for each
    pair of columns in the dataframe.

    :param df: pd.DataFrame
    :param size: vertical and horizontal size of the plot
    :param method: correlation test method
    :return fig: matplotlib figure object
    """
    corr = df.corr(method=method)
    fig, ax = plt.subplots(figsize=(size, size))
    im = ax.matshow(corr)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.colorbar(im)
    return fig


def feature_corr_matrix_compact(df, method="spearman"):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.matshow(df.corr(method=method))
    fig.colorbar(im)
    return fig


def feature_corr_matrix_sns(df, method="spearman", upsidedown=False):
    corr = df.corr(method=method)

    # Generate a mask for the upper triangle
    if not upsidedown:
        mask = np.ones_like(corr, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = False
    else:
        # Generate a mask for the lower triangle
        mask = np.zeros_like(corr, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    import seaborn as sns
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    x = sns.heatmap(corr, mask=mask, cmap=cmap, center=0, ax=ax,
                    square=True, linewidths=.5, cbar_kws={"shrink": .5})

    if upsidedown:
        for i, t in enumerate(ax.get_yticklabels()):
            t.set_rotation(180)

    return x


def np_to_pd(X, columns=None):
    if isinstance(X, pd.DataFrame):
        return X
    elif isinstance(X, np.ndarray):
        if columns is not None:
            assert len(columns) == len(X[0])
            return pd.DataFrame(X, columns=columns)
        return pd.DataFrame(
            X, columns=['var_{}'.format(k) for k in range(
                np.atleast_2d(X).shape[1])])
    else:
        raise ValueError('Input X must be a numpy array')


def balanced_sampling(df_minority, df_majority, minority_upsampling_ratio=0.2,
                      only_upsample=False, use_smote_upsampling=False):
    # Upsample minority class by minority_upsampling_ratio%
    new_size = int(df_minority.shape[0] * (1 + minority_upsampling_ratio))

    # upsample the minority class to the new size
    from sklearn.utils import resample
    if use_smote_upsampling:
        pass  # smote.SMOTE(kind='borderline1').fit_sample()
    else:
        df_minority = resample(df_minority, replace=True,
                               n_samples=new_size, random_state=0)

    if not only_upsample:
        # downsample the majority class to the new size
        df_majority = resample(df_majority, replace=False,
                               n_samples=new_size, random_state=0)
    df = pd.concat([df_minority, df_majority])

    return df


def feature_distributions_hist(df, figsize=(20, 20)):
    return df.hist(figsize=figsize)


def feature_distributions_boxplot(df, figsize=(20, 5)):
    return df.boxplot(rot=90, figsize=figsize)


def feature_class_relationship(df, by, figsize=(20, 20), ncols=4):
    """
    Plot histograms for every variable in :param df: grouped by :param by:.

    :param df: pd.DataFrame
    :param by: See documentation on Pandas.groupBy(by=...).
    :param figsize: See documentation on matplotlib.figure(figsize=...).
    :param ncols: Number of histogram plots in one row. One plot per variable.
    """
    grps = df.groupby(by)
    f = plt.figure(figsize=figsize)
    nrows = len(df.columns) // ncols + \
        (1 if len(df.columns) % ncols != 0 else 0)
    for i, c in enumerate(df.columns):
        ax = f.add_subplot(nrows, ncols, i + 1)
        for k, v in grps:
            ax.hist(v[c], label=k, bins=25, alpha=0.4)
        ax.set_title(c)
        ax.legend(loc='upper right')
    f.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.close()
    return f


def feature_feature_relationship(df, figsize=(20, 20)):
    from pandas.plotting import scatter_matrix
    sm = scatter_matrix(df + 0.00001 * np.random.rand(*df.shape),
                        alpha=0.2, figsize=figsize, diagonal='kde')

    # change label rotation
    [s.xaxis.label.set_rotation(90) for s in sm.reshape(-1)]
    [s.yaxis.label.set_rotation(0) for s in sm.reshape(-1)]

    # may need to offset label when rotating to prevent overlap of figure
    [s.get_yaxis().set_label_coords(-0.3, 0.5) for s in sm.reshape(-1)]

    # hide all ticks
    [s.set_xticks(()) for s in sm.reshape(-1)]
    [s.set_yticks(()) for s in sm.reshape(-1)]

    return sm


def feature_feature_relationship_one(df, cols, by=lambda x: True):
    assert 1 < len(
        cols) <= 3, 'Number of columns must equal 2 or 3 dimensions.'
    i = 0
    colors = list('rbgym')
    fig = plt.figure()
    if len(cols) == 3:
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = fig.add_subplot(111)
    for name, g in df.groupby(by):
        ax.scatter(*[g[c] for c in cols], label=name,
                   edgecolors='k', alpha=.2, color=colors[i])
        i += 1
    ax.set_xlabel(cols[0])
    ax.set_ylabel(cols[1])
    if len(cols) == 3:
        ax.set_zlabel(cols[2])
    plt.legend(loc="upper right")
    plt.close()
    return fig


def categorical_interaction_plot(df, col1, col2, by, figsize=(6, 6),
                                 **plot_kwargs):
    from statsmodels.graphics.factorplots import interaction_plot
    return interaction_plot(
        x=df[col1], trace=by, response=df[col2],
        colors=['red', 'blue'], markers=['D', '^'], ms=10, **plot_kwargs)


def drop_duplicates_sorted(df, subset=None, sep='', inplace=True):
    '''
    Drop duplicate rows based on unique combination of vales from
    given columns. Combinations will be sorted on row axis before
    dropping duplicates.

    :param df: pd.DataFrame
    :param subset: list of str
        Subset of columns to drop duplicates from.
    :param sep: str
        Separator string for columns when sorting row values.
    :param inplace: bool
        Whether to drop duplicates in place or to return a copy
    :return df: pd.DataFrame with duplicates across all :columns: dropped.
    '''
    if not inplace:
        df = df.copy()
    if subset is None:
        subset = df.columns
    df['check_string'] = df.apply(lambda row: sep.join(
        sorted([f"{row[c]}" for c in subset])), axis=1)
    df.drop_duplicates('check_string', inplace=True)
    df.drop('check_string', axis=1, inplace=True)
    return df


def add_NA_indicator_variables(df, inplace=False):
    """
    Add indicator variables for each column to indicate missingness.
    """
    df_ = df if inplace else df.copy()
    for i, c in enumerate(df_.columns):
        x = df_[c].isna()
        if x.any():
            df_.insert(i + 1, '{}_NA'.format(c), x)
    return df_


def nullity_correlation(df, corr_method='spearman', jupyter_nb=False,
                        fill_na=-1):
    df_ = df.copy()

    # add missingness indicator variables
    df_ = add_NA_indicator_variables(df_, inplace=True)

    # check correlation between each variable and indicator
    corr = df_.fillna(fill_na).corr(method=corr_method)

    # reshaping the correlation matrix
    corr = corr.stack().reset_index()
    corr.columns = ['col1', 'col2', 'value']

    # delete the left-right matching columns (since they are 100% correlated)
    # and right-side columns without the 'is_missing'
    corr = corr[(corr.col1 != corr.col2) &
                (corr.col2.apply(lambda x: 'NA' in x)) &
                (corr.apply(lambda row: not(row['col1'] in row['col2'] or
                    row['col2'] in row['col1']), axis=1))]

    # sort descending based on value column
    corr['value_abs'] = corr.value.apply(np.abs)
    corr = corr.dropna().sort_values(
        by='value_abs', ascending=False).drop(['value_abs'], axis=1)
    corr = drop_duplicates_sorted(
        corr, ['col1', 'col2']).reset_index(drop=True)

    if jupyter_nb:
        from IPython.display import display
        import ipywidgets as widgets
        from ipywidgets import interactive

        def filter(corr_strength=0.0, var_name=""):
            var_name = var_name.lower()
            x = corr
            if corr_strength > 0.0:
                x = x.where(abs(x.value) > corr_strength).dropna()
            if var_name:
                x = x.where(
                    x.apply(
                        lambda r: r.col1.lower().startswith(var_name) or
                        r.col2.lower().startswith(var_name), axis=1)
                ).dropna()
            print('N: {}'.format(x.shape[0]))
            display(x)
            return x

        corr_slider = widgets.FloatSlider(
            value=0.0, min=0.0, max=1.0, step=0.0001)
        text_filter = widgets.Text(value="", placeholder="Type variable name")
        w = interactive(filter, corr_strength=corr_slider,
                        var_name=text_filter)

        return corr, w

    return corr


def linearity_with_logodds(df, col, label_col, ax=None):
    df_ = pd.pivot_table(df, columns=[label_col], index=[
                         col], aggfunc={label_col: len}, fill_value=0)
    df_['Odds(1)'] = df_[(label_col, 1.0)] / df_[(label_col, 0.0)]
    df_['LogOdds(1)'] = np.log(1 + df_['Odds(1)'])
    df_ = df_.reset_index()
    return df_.plot.line(col, "LogOdds(1)", ax=ax)


def linearity_with_logodds_allcols(df, label_col, figsize=(30, 80), ncols=4):
    fig = plt.figure(figsize=figsize)
    n = len(df.columns) - 1  # minus 1 because we discount label column
    nrows = n // ncols + (1 if n % ncols != 0 else 0)
    for i, c in enumerate(df.columns.difference([label_col])):
        ax = fig.add_subplot(nrows, ncols, i + 1)
        linearity_with_logodds(df, c, label_col, ax=ax)
    plt.tight_layout()
    return fig


def filter_columns(df, drop_cols=[], keep_cols=[]):
    # first drop columns; keep_cols takes priority,
    # i.e., columns in both sets will be kept
    df.drop(set(drop_cols).difference(keep_cols),
            axis=1, inplace=True, errors="ignore")

    # then keep only columns given in keep_cols;
    # if none given, then keep all remaining
    x = df.columns.intersection(keep_cols)
    if len(x) > 0:
        df = df[x]
    return df


def drop_constant_columns(df, inplace=False, verbose=False):
    if verbose:
        print('Dropping constant columns')
    drop_cols = [c for c in df if df[c].nunique() == 1]
    if verbose:
        print('Before:', df.shape[1])
    if inplace:
        df.drop(drop_cols, axis=1, inplace=True)
        if verbose:
            print('After:', df.shape[1])
    else:
        X = df.drop(drop_cols, axis=1)
        if verbose:
            print('After:', X.shape[1])
        return X


def keep_top_k_categories(s, k=1, dropna=False):
    if s.nunique() == 1:
        return s
    categories = [str(x) for x in s.value_counts(dropna=dropna).index.tolist()]
    other_categories = categories[k:]
    print('Top-{} categories:'.format(k), categories[:k], '\n')
    print('Other categories:', other_categories)
    return s.apply(lambda r: 'Other' if str(r) in other_categories else r)


def df_value_counts(df, normalize=False, delimiter=';'):
    cols = df.columns
    df = df.copy()
    df = df.apply(lambda row: delimiter.join((
        str(x) for x in row)), axis=1).value_counts(normalize=normalize)
    df = df.reset_index()
    df[cols] = df['index'].str.split(delimiter, expand=True)
    df = df.drop('index', axis=1)
    return df


def reorder_columns(df, src_idx, dest_idx, inplace=False):
    df_ = df if inplace else df.copy()
    c = df_.iloc[:, src_idx]
    df_.drop(c.name, axis=1, inplace=True)
    df_.insert(dest_idx, c.name, c)
    return df_


def get_age_from_dob(s, round=True):
    age = ((pd.to_datetime('today') - s) / np.timedelta64(1, 'Y')).fillna(-1)
    if round:
        age = age.apply(np.round).apply(int)
    return age


def insert_column(df, column_name, column, after_column=None,
                  before_column=None):
    if after_column:
        idx = int((df.columns == after_column).nonzero()[0][0]) + 1
    elif before_column:
        idx = int((df.columns == before_column).nonzero()[0][0])
    else:
        df[column_name] = column
        return df
    df.insert(idx, column_name, column)
    return df


def get_current_datetime_str():
    import time
    return time.strftime("%Y%m%d-%H%M%S")


def feature_select_correlation(df, threshold=.5):
    cols = df.columns.tolist()
    index = 0
    x = []
    while len(cols) > index:
        corr = correlations_to_column(df[cols], cols[index]).spearman
        corr = corr[
            (corr.apply(lambda x: abs(x) > threshold)) &
            (corr.index != cols[index])]
        if len(corr) > 0:
            x.append(pd.DataFrame({
                'selected': [cols[index]] * len(corr),
                'correlated': corr.index,
                'corr_strength': corr.values
            }))
            for c in corr.index:
                cols.remove(c)
        index += 1
    return df[cols], pd.concat(x)


def mahalanobis_distance(df):
    cov_matrix = np.cov(df)
    from . import numpy_utils as npu
    if npu.is_pos_def(cov_matrix):
        inv_cov_matrix = np.linalg.inv(cov_matrix)
        if npu.is_pos_def(inv_cov_matrix):
            means = df.mean(axis=0)
            df = df.apply(lambda row: row - means, axis=1)
            df = df.apply(
                lambda row: np.sqrt(row @ inv_cov_matrix @ row), axis=1)
            return df
        else:
            raise ValueError("Covariance Matrix is not positive definite!")
    else:
        raise ValueError("Covariance Matrix is not positive definite!")


def outlier_detection_mahalanobis(df, threshold=2):
    md = mahalanobis_distance(df)
    std = np.std(md)
    m = np.mean(md)
    k = threshold * std
    up, lo = m + k, m - k
    return np.argwhere(np.logical_or(md >= up, md <= lo))[:, 0]


def save_xls(list_dfs, xls_path, sheet_names=[], to_excel_kws={},
             excelwriter_kws={}):
    """
    Save list of pandas dataframes to single excel with multiple sheets.

    :param list_dfs:
        List of pandas dataframes
    :param xls_path:
        file save path
    :param sheet_names: Optional
        Sheet names for each dataframe
    """
    with pd.ExcelWriter(xls_path, **excelwriter_kws) as writer:
        if sheet_names:
            assert len(sheet_names) == len(list_dfs)
            for n, df in enumerate(list_dfs):
                df.to_excel(writer, sheet_names[n], **to_excel_kws)
        else:
            for n, df in enumerate(list_dfs):
                df.to_excel(writer, 'sheet%s' % n, **to_excel_kws)
        writer.save()
