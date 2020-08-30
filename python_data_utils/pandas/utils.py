# coding: utf-8

"""
    description: Pandas utility functions and classes
    author: Suraj Iyer
"""

__all__ = [
    'apply_parallel',
    'df_mem_usage',
    'optimize_dataframe',
    'tsv_to_pandas_generator',
    'tsv_to_pandas',
    'pandas_to_tsv',
    'explode_horizontal',
    'explode_multiple',
    'explode_horizontal_ohe',
    'get_missingness',
    'get_missingness_perc',
    'plot_missingness_heatmap',
    'correlations_to_column',
    'plot_correlations',
    'plot_correlations_compact',
    'balanced_sampling',
    'plot_histograms',
    'plot_boxplots',
    'feature_class_relationship',
    'feature_feature_relationship',
    'feature_feature_relationship_one',
    'categorical_interaction_plot',
    'drop_duplicates_sorted',
    'add_NA_indicator_variables',
    'nullity_correlation',
    'linearity_with_logodds',
    'linearity_with_logodds_allcols',
    'filter_columns',
    'drop_constant_columns',
    'keep_top_k_categories',
    'df_value_counts',
    'reorder_columns',
    'get_age_from_dob',
    'insert_column',
    'feature_select_correlation',
    'save_xls',
    'get_time_slots'
]


import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def apply_parallel(df, func, axis=0, n_jobs=-1, chunksize=None):
# def apply_parallel(df, func, axis=0, n_processes=4, mode=0):
    """
    Apply function on dataframe in parallel.

    Params
    ------
    df : pandas Dataframe

    func : callable
        function to apply

    axis : int, default=0
        0: index, 1: columns

    n_jobs : int, default=-1
        number of processes to spawn in parallel

    chunksize : int, default=None
        number of samples for each worker process
        if axis=0, then default value=100
        if axis=1, then default value=1
    
    # n_processes : positive int
    # mode : 0/1
    #     Mode is 0 (faster but consumes more memory)
    #     or 1 (slower but might consume less memory)
    """
    # Initialize some temporary global variables
    # Multiprocessing library can only copy globals to forked processes
    # from multiprocessing import Pool
    # if mode == 1:
    #     global _apply_parallel_df, _apply_parallel_func, _apply_parallel_wrapper
    #     _apply_parallel_df = df
    #     _apply_parallel_func = func
    #     if axis == 0:
    #         def _apply_parallel_wrapper(indices):
    #             global _apply_parallel_df
    #             return _apply_parallel_func(_apply_parallel_df.iloc[indices])
    #     elif axis == 1:
    #         def _apply_parallel_wrapper(indices):
    #             global _apply_parallel_df
    #             return _apply_parallel_func(_apply_parallel_df.iloc[:, indices])
    #     else:
    #         raise ValueError('Axis has to be 0 (row index) or 1 (column index).')

    #     # Apply the function parallely
    #     df_split = np.array_split(np.arange(df.shape[axis]), n_processes)
    #     with Pool(n_processes) as pool:
    #         df = pd.concat(pool.map(_apply_parallel_wrapper, df_split), axis=axis)

    #     # Delete the temporary global variables
    #     del _apply_parallel_df, _apply_parallel_func, _apply_parallel_wrapper
    # elif mode == 0:
    #     df_split = pd.np.array_split(df, n_processes)
    #     with Pool(n_processes) as pool:
    #         df = pd.concat(pool.map(func, df_split))
    # else:
    #     raise ValueError('Mode has to be 0/1.')

    # return df
    assert callable(func), 'func must be callable.'
    assert axis in (0, 1), 'axis must be 0 (index) or 1 (columns).'
    if chunksize is None:
        chunksize = 100 if axis == 0 else 1

    from joblib import Parallel, delayed

    def chunker(iterable, total_length, chunksize):
        return (iterable[pos: pos + chunksize] for pos in range(0, total_length, chunksize))

    def process_chunk(chunk):
        return chunk.apply(func, axis=axis)

    def collate_results(list_of_dfs):
        return pd.concat(list_of_dfs, axis=axis, ignore_index=True, copy=False)

    executor = Parallel(n_jobs=n_jobs, backend='multiprocessing', prefer="processes")
    do = delayed(process_chunk)
    iterable = df.loc if axis == 0 else df.swapaxes(0, 1, copy=False).loc
    tasks = (do(chunk) for chunk in chunker(iterable, df.shape[axis], chunksize=chunksize))
    return collate_results(executor(tasks))


def df_mem_usage(df):
    """Calculate memory usage of pandas dataframe."""
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


def optimize_dataframe(
        df, categorical=[], always_positive_ints=[],
        cat_nunique_ratio=.5, verbose=False):
    """
    Optimize the memory usage of the given dataframe by
    modifying data types.

    Params
    ------
    df : pandas DataFrame

    categorical : list of (str, bool) pairs, default=[]
        List of categorical variables with boolean
        representing if they are ordered or not.

    always_positive_ints : list of str, default=[]
        List of always positive INTEGER variables

    cat_nunique_ratio : 0.0 <= float <= 1.0, default=0.5
        Ratio of unique values to total values. Used for
        detecting categorical columns.

    verbose : bool, default=False
        Print before and after memory usage

    Returns
    -------
    df : pandas DataFrame
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
    Read data from TSV file as pandas DataFrame and
    yield them as chunks.

    Params
    ------
    file_path : str
        Path to tsv file

    target_label : str, default=None
        target label column name

    chunksize : int, default=None
        function yields dataframes in chunks as a
        generator function

    Returns
    -------
    X : pandas DataFrame

    y : pandas Series
        included if target label is given
    """
    assert isinstance(file_path, str) and len(file_path) > 0
    assert target_label is None or isinstance(target_label, str)
    assert chunksize is None or isinstance(chunksize, int)

    for chunk in pd.read_csv(
            file_path, delimiter='\t', na_values=r'\N',
            chunksize=chunksize):
        if target_label:
            yield chunk.drop([target_label], axis=1), chunk[target_label]
        else:
            yield chunk


def tsv_to_pandas(
        file_path, target_label=None, memory_optimize=True,
        categorical=[], always_positive=[], save_pickle_obj=False,
        verbose=False):
    """
    Read data from TSV file as pandas DataFrame.

    Params
    ------
    file_path : str
        Path to tsv file

    target_label : str
        target label column name

    memory_optimize : bool, default=True
        Optimize the data types of the columns. Will take
        some more time to compute but will reduce memory
        usage of the dataframe.

    categorical : list of str, default=[]
        List of categorical variables

    always_positive : list of str, default=[]
        List of always positive INTEGER variables

    verbose : bool, default=False
        Output time to load the data and memory usage.

    save_pickle_obj : bool, str, default=False
        Save the final result as a pickle object. Save
        path can be given as a string. Otherwise, saved
        in the same directory as the input file path.

    Returns
    -------
    X : pandas DataFrame

    y : pandas Series
        included if target label is given
    """
    assert isinstance(file_path, str) and len(file_path) > 0
    assert target_label is None or isinstance(target_label, str)
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
        if isinstance(save_pickle_obj, str):
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

    Params
    ------
    df : pandas DataFrame
        DataFrame to save as tsv

    save_file_path : str
        File save path. if path does not exist, it will be
        created automatically.

    index : bool
        write the row names to output.

    mode : str
        file save mode; 'w' for write and 'a' for append to existing file.

    header : bool
        write the column names to output.
    """
    assert isinstance(save_file_path, str) and\
        len(save_file_path) > 0
    assert isinstance(df, pd.DataFrame) and not df.empty,\
        'df must be a non-empty pandas DataFrame'
    assert isinstance(mode, str)

    # make a new directory to save the file
    try:
        import os
        os.makedirs(os.path.split(save_file_path)[0])
    except OSError as e:
        import errno
        if e.errno != errno.EEXIST:
            raise

    # save the file
    df.to_csv(
        save_file_path, sep='\t', na_rep=r'\N',
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
        }).assign(**{col: np.concatenate(df[col].values) for col in lst_cols})\
          .loc[:, df.columns]
    else:
        # at least one list in cells is empty
        return pd.DataFrame({
            col: np.repeat(df[col].values, df[lst_cols[0]].str.len())
            for col in idx_cols
        }).assign(**{col: np.concatenate(df[col].values) for col in lst_cols})\
          .append(df.loc[lens == 0, idx_cols]).fillna(fill_value) \
          .loc[:, df.columns]


def explode_horizontal_ohe(s):
    """
    Expands a column of lists to multiple one-hot
    encoded columns per unique element from all
    lists.

    Params
    ------
    s : pandas Series
        Series column of lists.

    Returns
    -------
    df : pandas DataFrame
        Dataframe with multiple columns, one per unique
        item in all lists.
    """
    return pd.get_dummies(s.apply(pd.Series).stack()).sum(level=0)


def get_missingness(df):
    return [(c, df[c].isna().sum()) for c in df if df[c].isna().any()]


def get_missingness_perc(df):
    """
    Get a percentage of missing values per column in input dataframe.

    Params
    ------
    df : pandas DataFrame

    Returns
    -------
    missingness : pandas DataFrame
    """
    missingness = pd.DataFrame(
        [(len(df[c]) - df[c].count()) * 100.00 / len(df[c]) for c in df],
        index=df.columns, columns=['Missingness %'])
    return missingness


def plot_missingness_heatmap(df):
    import seaborn as sns
    return sns.heatmap(df.isnull(), cbar=False)


def correlations_to_column(df, col, get_slider=False):
    """
    Get correlation of all columns to given column `col`.
    If `get_slider=True`, get interactive widget slider to
    filter columns by spearman correlation strength.

    Params
    ------
    df : pandas DataFrame

    col : str
        Label column for which we find correlation
        to all other columns.

    get_slider : bool
        If true, returns the optional slider widget.
        See below.

    Returns
    -------
    corr : pandas DataFrame
        Dataframe containing 2 columns: one for correlation
        values obtained using Pearson Rho test and one with
        Spearman Rank test.

    slider : widgets.FloatSlider, optional
        Jupyter slider widget ranging from 0.0 to 1.0 to
        filter `corr` over spearman correlation strength
        interactively.
    """
    # Get Pearson correlation - to describe extent of linear correlation
    # with label
    corr = pd.concat([
        df.corr(method="pearson")[col].rename("pearson"),
        df.corr(method="spearman")[col].rename("spearman")], axis=1)

    if get_slider:
        from IPython.display import display
        import ipywidgets as widgets
        from ipywidgets import interactive

        def view_correlations(threshold=0.0):
            if threshold == 0.0:
                x = corr
            else:
                x = corr.where(
                    (abs(corr.spearman) > threshold) &
                    (corr.spearman != 1)).dropna()
            print('N: {}'.format(x.shape[0]))
            display(x)
            return x

        corr_slider = widgets.FloatSlider(
            value=0.0, min=0.0, max=1.0, step=0.0001)
        w = interactive(view_correlations, threshold=corr_slider)

        return corr, w

    return corr


def plot_correlations(df, figsize=10, method="spearman"):
    """
    Function plots a graphical correlation matrix for each
    pair of columns in the dataframe.

    Params
    ------
    df : pandas DataFrame

    figsize : float
        width of square shaped plot

    method : str
        correlation test method. See available methods
        at `pandas.DataFrame.corr`

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    corr = df.corr(method=method)
    fig, ax = plt.subplots(figsize=(figsize, figsize))
    im = ax.matshow(corr)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.colorbar(im)
    return fig


def plot_correlations_compact(df, method="spearman"):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.matshow(df.corr(method=method))
    fig.colorbar(im)
    return fig


def balanced_sampling(
        df, group_by, n_minority=None, frac_minority=None,
        random_state=0):
    """
    Balance the dataframe on the `group_by` column.

    Params
    ------
    df : pandas DataFrame

    group_by : mapping, function, label, or list of labels
        See pandas.DataFrame.groupby.

    n_minority : int
        New size of each group along rows of the `group_by`
        column. If n_minority > group sizes, rows will be
        sampled more than once.

    frac_minority : float
        New size of each group (same as above) given as a
        fraction of the size of the smallest group. Cannot
        be specified together with `n_minority`. If
        frac_minority > 1., rows may be sampled more than
        once.

    Returns
    -------
    df : pandas DataFrame
        Resampled dataframe
    """
    g = df.groupby(group_by)
    n = n_minority if n_minority is not None\
        else g.size().min() * frac_minority
    return g.apply(lambda _: _.sample(
        n=n, replace=n > _.shape[0], random_state=random_state))


def plot_histograms(df, figsize=(20, 20)):
    return df.hist(figsize=figsize)


def plot_boxplots(df, figsize=(20, 5)):
    return df.boxplot(rot=90, figsize=figsize)


def feature_class_relationship(df, group_by, figsize=None, ncols=4):
    """
    Plot histograms for every variable in `df` grouped by `group_by`.

    Params
    ------
    df: pandas DataFrame

    group_by: mapping, function, label, or list of labels
        See pandas.DataFrame.groupby.

    figsize : (float, float), default=None
        See matplotlib.figure.Figure.

    ncols : int
        Number of histogram plots per row.

    Returns
    -------
    f : matplotlib.figure.Figure
    """
    g = df.groupby(group_by)
    f = plt.figure(figsize=figsize)
    nrows = len(df.columns) // ncols + \
        (1 if len(df.columns) % ncols != 0 else 0)
    for i, c in enumerate(df.columns):
        ax = f.add_subplot(nrows, ncols, i + 1)
        for k, v in g:
            ax.hist(v[c], label=k, bins=25, alpha=0.4)
        ax.set_title(c)
        ax.legend(loc='upper right')
    f.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.close()
    return f


def feature_feature_relationship(df, figsize=None):
    """
    Plot 2D scatter plot for every pair of columns to
    see relationships.
    """
    from pandas.plotting import scatter_matrix
    sm = scatter_matrix(
        df + 0.00001 * np.random.rand(*df.shape),
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


def feature_feature_relationship_one(df, *cols, group_by=lambda x: True):
    """
    Plot 2/3D scatter plot over `cols` to see relationship.
    """
    assert 1 < len(cols) <= 3,\
        'Number of columns must equal 2 or 3 dimensions.'
    i = 0
    colors = list('rbgym')
    f = plt.figure()
    if len(cols) == 3:
        ax = f.add_subplot(111, projection='3d')
    else:
        ax = f.add_subplot(111)
    for name, g in df.groupby(group_by):
        ax.scatter(*[g[c] for c in cols], label=name,
                   edgecolors='k', alpha=.2, color=colors[i])
        i += 1
    ax.set_xlabel(cols[0])
    ax.set_ylabel(cols[1])
    if len(cols) == 3:
        ax.set_zlabel(cols[2])
    plt.legend(loc="upper right")
    plt.close()
    return f


def categorical_interaction_plot(df, col1, col2, by, **plot_kwargs):
    from statsmodels.graphics.factorplots import interaction_plot
    return interaction_plot(
        x=df[col1], trace=by, response=df[col2],
        colors=['red', 'blue'], markers=['D', '^'], ms=10,
        **plot_kwargs)


def drop_duplicates_sorted(df, subset=None, sep='', inplace=True):
    """
    Drop duplicate rows based on unique combination of values from
    given columns. Combinations will be sorted on row axis before
    dropping duplicates.

    Params
    ------
    df : pandas DataFrame

    subset : list of str
        Subset of columns to drop duplicates from.

    sep : str
        Separator string for columns when sorting row values.

    inplace : bool
        Whether to drop duplicates in place or to return a copy

    Returns
    -------
    df : pandas DataFrame with duplicates across columns in `subset` dropped.
    """
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


def nullity_correlation(
        df, corr_method='spearman', get_widgets=False, fill_na=-1):
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

    if get_widgets:
        from IPython.display import display
        import ipywidgets as widgets
        from ipywidgets import interactive

        def filter(threshold=0.0, var_name=""):
            var_name = var_name.lower()
            x = corr
            if threshold > 0.0:
                x = x.where(abs(x.value) > threshold).dropna()
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
        w = interactive(filter, threshold=corr_slider, var_name=text_filter)

        return corr, w

    return corr


def linearity_with_logodds(df, col, label_col, ax=None):
    df_ = pd.pivot_table(
        df, columns=[label_col], index=[col],
        aggfunc={label_col: len}, fill_value=0)
    df_['Odds(1)'] = df_[(label_col, 1.0)] / df_[(label_col, 0.0)]
    df_['LogOdds(1)'] = np.log(1 + df_['Odds(1)'])
    df_ = df_.reset_index()
    return df_.plot.line(col, "LogOdds(1)", ax=ax)


def linearity_with_logodds_allcols(df, label_col, figsize=None, ncols=4):
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
        df = df.drop(drop_cols, axis=1)
        if verbose:
            print('After:', df.shape[1])
        return df


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


def get_age_from_dob(s : pd.Series, round : bool = True):
    age = ((pd.to_datetime('today') - s) / np.timedelta64(1, 'Y')).fillna(-1)
    if round:
        age = age.apply(np.round).apply(int)
    return age


def insert_column(
        df, column_name, values, after_column=None, before_column=None):
    if (column_name in df.columns) or (not after_column and not before_column):
        df[column_name] = values
        return df
    elif after_column:
        idx = int((df.columns == after_column).nonzero()[0][0]) + 1
    elif before_column:
        idx = int((df.columns == before_column).nonzero()[0][0])
    df.insert(idx, column_name, values)
    return df


def feature_select_correlation(df, threshold=.5):
    """
    Drop highly correlated features. If pairs of
    features have greater than `threshold` spearman
    correlation, one of them is dropped at random.
    """
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


def save_xls(save_path, excelwriter_kws={}, to_excel_kws={}, **sheets):
    """
    Save list of pandas dataframes to single excel with multiple sheets.

    Params
    ------
    save_path : str
        file save path

    excelwriter_kws : dict, default={}
        Arguments for pandas.ExcelWriter()

    to_excel_kws : dict, default={}
        Arguments for pandas.DataFrame.to_excel()

    sheets : Dict[str, pandas.DataFrame]
        Mapping from sheet name to dataframe
    """
    if sheets:
        with pd.ExcelWriter(save_path, **excelwriter_kws) as writer:
            for k, v in sheets.items():
                v.to_excel(writer, k, **to_excel_kws)
            writer.save()
    else:
        raise ValueError('No sheets passed.')


def get_time_slots(s : pd.Series, time_interval : str = 'daily'):
    """Convert timestamps to time slots"""
    if time_interval.lower() not in (
        'hourly', 'daily', 'weekly', 'monthly',
        'quarterly', 'yearly'):
        raise ValueError
    return pd.to_datetime(s)\
        .dt.to_period(time_interval[0].upper())


if __name__ == '__main__':
    import doctest
    doctest.testmod()
