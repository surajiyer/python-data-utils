# coding: utf-8

"""
    description: Pandas utility functions and classes
    author: Suraj Iyer
"""

from __future__ import print_function

import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
import ipywidgets as widgets
from ipywidgets import interactive, interact
import six
import time
import dill
import shutil
import os
import errno
from sklearn.utils import resample
from progressbar.bar import ProgressBar
# from imblearn.over_sampling import smote


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
    result.update({'Average memory usage for {} columns'.format(c): None for c in dtypes})
    for dtype in dtypes:
        usage_b = df.select_dtypes(include=[dtype]).memory_usage(deep=True).mean()
        result.update({'Average memory usage for {} columns'.format(dtype)
                       : "{:03.2f} MB".format(usage_b / 1024 ** 2)})

    return result

def optimize_dataframe(df, categorical=[], always_positive_ints=[], cat_nunique_ratio=.5, verbose=False):
    """
    Optimize the memory usage of the given dataframe by modifying data types.
    :param df: pandas DataFrame
    :param categorical: list of (str, bool) pairs, optional (default=[])
        List of categorical variables with boolean representing if they are ordered or not.
    :param always_positive_ints: list of str, optional (default=[])
        List of always positive INTEGER variables
    :param cat_nunique_ratio: 0.0 <= float <= 1.0, (default=0.5)
        Ratio of unique values to total values. Used for detecting cateogrical columns.
    :param verbose: bool, optional (default=False)
        Print before and after memory usage
    :return df: pandas DataFrame
    """
    cat = [] if len(categorical) == 0 else list(zip(*categorical))[0]
    getCols = lambda dtype: df.columns[df.dtypes == dtype].difference(always_positive_ints).difference(cat)

    if verbose:
        print('Before:', df_mem_usage(df))

    # convert always positive columns to unsigned ints
    # nulls are represented as floats so we ignore them
    cols = df.columns[~df.isnull().any()].intersection(always_positive_ints)
    df.loc[:, cols] = df[cols].fillna(-1).astype('int').apply(pd.to_numeric, downcast='unsigned')
    # Converting back to nan changes them back to float64 dtype
    # df.loc[(df[cols] == -1).any(axis=1), cols] = np.nan

    # downcast integer columns
    cols = getCols('int')
    df.loc[:, cols] = df[cols].apply(pd.to_numeric, downcast='integer')

    # downcast float columns
    cols = getCols('float')
    df.loc[:, cols] = df[cols].apply(pd.to_numeric, downcast='float')

    # convert object columns with less than {cat_nunique_ratio}% unique values to categorical
    for col in getCols('object'):
        num_unique_values = len(df[col].unique())
        num_total_values = len(df[col].dropna())
        if num_unique_values / float(num_total_values) < cat_nunique_ratio:
            df.loc[:, col] = df[col].astype('category')

    # convert given columns to categorical
    for col, ord in categorical:
        if col in df.columns and df.dtypes[col].name != 'category':
            df.loc[:, col] = df[col].astype('category', ordered=ord)

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
        X: pandas.DataFrame
        y: pandas.Series
            included if target label is given
    """
    assert isinstance(file_path, six.string_types) and len(file_path) > 0
    assert target_label is None or isinstance(target_label, six.string_types)
    assert isinstance(chunksize, int)

    for chunk in pd.read_csv(file_path, delimiter='\t', na_values=r'\N', chunksize=chunksize):
        if target_label:
            yield chunk.drop([target_label], axis=1), chunk[target_label]
        else:
            yield chunk

def tsv_to_pandas(file_path, target_label=None, memory_optimize=True, categorical=[],
                  always_positive=[], save_pickle_obj=False, verbose=False):
    """
    Read data from TSV file as pandas DataFrame.
    :param file_path: str
        Path to tsv file
    :param target_label: str
        target label column name
    :param memory_optimize: bool, optional (default=True)
        Optimize the data types of the columns. Will take some more time to compute but will
        reduce memory usage of the dataframe.
    :param categorical: list of str, optional (default=[])
        List of categorical variables
    :param always_positive: list of str, optional (default=[])
        List of always positive INTEGER variables
    :param verbose: bool, optional (default=False)
        Output time to load the data and memory usage.
    :param save_pickle_obj: bool, str, optional (default=False)
        Save the final result as a pickle object. Save path can be given as a string.
        Otherwise, saved in the same directory as the input file path.
    :return: 
        X: pandas.DataFrame
        y: pandas.Series
            included if target label is given
    """
    assert isinstance(file_path, six.string_types) and len(file_path) > 0
    assert target_label is None or isinstance(target_label, six.string_types)

    if verbose:
        start_time = time.time()

    # Get number of rows in data
    with open(file_path, 'r') as f:
        filesize = sum(1 for line in f)

    # Initialize progress bar
    chunksize = 100000
    progress = ProgressBar(max_value=int(filesize / chunksize)).start()

    data = tsv_to_pandas_generator(file_path, target_label, chunksize=chunksize)
    X_list, y_list = [], []

    # To save intermediate results
    if memory_optimize:
        results_dir = os.path.splitext(file_path)[0]

        # delete directory if it already exists
        shutil.rmtree(results_dir, ignore_errors=True)

        # make the results directory
        if results_dir:
            try:
                os.makedirs(results_dir)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise

    for i, X in enumerate(data):
        if target_label:
            X, y = X[0], X[1]

        # Memory compression
        if memory_optimize:
            if i == 0:
                always_positive = [col for c0 in always_positive for col in X.columns if c0 in col]

            X = optimize_dataframe(X, always_positive_ints=always_positive, categorical=categorical)

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
        print('Finished in {0:.1f} minutes'.format((time.time() - start_time) / 60))

    # save the final result
    if save_pickle_obj:
        if isinstance(save_pickle_obj, six.string_types):
            save_path = save_pickle_obj

            # make the save directory
            if save_path:
                try:
                    os.makedirs(save_path)
                except OSError as e:
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

def pandas_to_tsv(save_file_path, df, index=False, mode='w', header=True):
    """
    Save pre-processed DataFrame as tsv file.
    :param save_file_path: str
        File save path. Path must exist, if not, a path will be created automatically.
    :param df: pandas.DataFrame
        DataFrame to save as tsv
    :param index: bool
        write the row names to output.
    :param mode: str
        file save mode; 'w' for write and 'a' for append to existing file.
    :param header: bool
        write the column names to output.
    :return: 
    """
    assert isinstance(save_file_path, six.string_types) and len(save_file_path) > 0
    assert isinstance(df, pd.DataFrame) and not df.empty, 'df must be a non-empty pandas.DataFrame'
    assert isinstance(mode, six.string_types)

    # make a new directory to save the file
    try:
        os.makedirs(os.path.split(save_file_path)[0])
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    # save the file
    df.to_csv(save_file_path, sep='\t', na_rep=r'\N', index=index, mode=mode, header=header)

def list_column_to_mutliple_columns(s):
    """
    Expands a single column containing variable-length lists to multiple binary columns.
    :param s: List column pandas series
    :return df: Pandas dataframe with multiple columns, one per unique item in all lists.
    """
    return pd.get_dummies(s.apply(pd.Series).stack()).sum(level=0)

def get_missingness_perc(df):
    """
    Get a percentage of missing values per column in input dataframe.
    :param df: Input pandas dataframe
    :return missingness: Pandas dataframe with index as columns from df and values 
                         as missingness level of corresponding column.
    """
    missingness = pd.DataFrame([(len(df[c])-df[c].count())*100.00/len(df[c]) for c in df], 
                               index=df.columns, columns=['Missingness %'])
    return missingness

def jupyter_plot_interactive_correlation_to_label_col(df, label_col):
    """
    Plot (interactive  jupyter NB) correlation vector of label column to all other columns.
    Correlation value obtained using spearman rank test.
    
    :param df: Pandas dataframe
    :label_col: Label column for which we find correlation to all other columns.
    :return corr: Dataframe containing 2 columns: one for correlation values obtained 
                  using Pearson Rho test and one with Spearman Rank test.
    :return corr_slider: Jupyter widgets.FloatSlider ranging from 0.0 to 1.0 to control interactive view.
    """
    # Get Pearson correlation - to describe extent of linear correlation with label
    pearson = df.corr(method="pearson")[label_col].rename("pearson")

    # Get Spearman rank correlation - to describe extent of any monotonic relationship with label
    spearman = df.corr(method="spearman")[label_col].rename("spearman")

    corr = pd.concat([pearson, spearman], axis=1)

    def view_correlations(corr_strength=0.0):
        if corr_strength == 0.0: x = corr
        else: x = corr.where((abs(corr.spearman) > corr_strength) & (corr.spearman != 1)).dropna()
        display(x.shape[0])
        display(x)
        return x

    corr_slider = widgets.FloatSlider(value=0.0, min=0.0, max=1.0, step=0.0001)
    w = interactive(view_correlations, corr_strength=corr_slider)
    
    return corr, corr_slider, w

def plot_corr(df, size=10, method="spearman"):
    """Function plots a graphical correlation matrix for each pair of columns in the dataframe.

    Input:
        df: pandas DataFrame
        size: vertical and horizontal size of the plot
        method: correlation test method
    """

    corr = df.corr(method=method)
    fig, ax = plt.subplots(figsize=(size, size))
    im = ax.matshow(corr)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.colorbar(im)
    plt.show()

def np_to_pd(X, columns=None):
    if isinstance(X, pd.DataFrame):
        return X
    elif isinstance(X, pd.np.ndarray):
        if columns:
            assert len(columns) == len(X[0])
            return pd.DataFrame(X, columns=columns)
        return pd.DataFrame(X, columns=['var_{}'.format(k) for k in range(pd.np.atleast_2d(X).shape[1])])
    else:
        raise ValueError('Input X must be a numpy array')

def balanced_sampling(df_minority, df_majority, minority_upsampling_ratio=0.2, only_upsample=False,
                      use_smote_upsampling=False):
    # Upsample minority class by minority_upsampling_ratio%
    new_size = int(df_minority.shape[0] * (1 + minority_upsampling_ratio))
    # upsample the minority class to the new size
    if use_smote_upsampling:
        pass  #smote.SMOTE(kind='borderline1').fit_sample()
    else:
        df_minority = resample(df_minority, replace=True, n_samples=new_size, random_state=0)
    if not only_upsample:
        # downsample the majority class to the new size
        df_majority = resample(df_majority, replace=False, n_samples=new_size, random_state=0)
    df = pd.concat([df_minority, df_majority])
    return df
