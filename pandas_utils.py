import pandas as pd

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
    return pd.get_dummies(s['commercialsegments'].apply(pd.Series).stack()).sum(level=0)
