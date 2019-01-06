import pandas as pd
import matplotlib.pyplot as plt
import ipywidgets as widgets
from ipywidgets import interactive, interact

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

def optimize_dataframe(df, categorical=[], always_positive_ints=[], verbose=False):
    """
    Optimize the memory usage of the given dataframe by modifying data types.
    :param df: pandas DataFrame
    :param categorical: list of (str, bool) pairs, optional (default=[])
        List of categorical variables with boolean representing if they are ordered or not.
    :param always_positive_ints: list of str, optional (default=[])
        List of always positive INTEGER variables
    :param verbose: bool, optional (default=False)
        Print before and after memory usage
    :return df: pandas DataFrame
    """
    cat = [] if len(categorical) == 0 else list(zip(*categorical))
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

    # convert object columns with less than 50% unique values to categorical
    for col in getCols('object'):
        num_unique_values = len(df[col].unique())
        num_total_values = len(df[col])
        if num_unique_values / float(num_total_values) < 0.5:
            df.loc[:, col] = df[col].astype('category')

    # convert given columns to categorical
    for col, ord in categorical:
        if col in df.columns and df.dtypes[col].name != 'category':
            df.loc[:, col] = df[col].astype('category', ordered=ord)

    if verbose:
        print('After:', df_mem_usage(df))

    return df

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
