# coding: utf-8

"""
    description: Decorators for pandas DataFrame objects
    author: Suraj Iyer
"""


import inspect
# import importlib.resources as pkg_resources


try:
    from pandas import DataFrame, Series
    import python_data_utils.pandas.utils as pdu

    # with pkg_resources.path('python_data_utils.pandas', 'utils.py') as p:
    #     with open(p) as f:
    #         f = f.readlines()

    #         # Find function names which takes "df" as first input
    #         # parameter for DataFrame
    #         df_functions = [
    #             l.split(" ")[1].split("(")[0] for l in f
    #             if l.startswith('def') and ('(df,' in l or '(df)' in l)]

    #         # Find function names which takes "s" as first input
    #         # parameter for Series
    #         series_functions = [
    #             l.split(" ")[1].split("(")[0] for l in f
    #             if l.startswith('def') and ('(s,' in l or '(s)' in l)]

    #     for fun in df_functions:
    #         setattr(DataFrame, fun, getattr(pdu, fun))

    #     for fun in series_functions:
    #         setattr(Series, fun, getattr(pdu, fun))

    # get all functions from pdu
    functions = inspect.getmembers(pdu, inspect.isfunction)

    # get function parameters
    func_signatures = list(map(
        lambda f: (f[0], f[1], list(inspect.signature(f[1]).parameters)),
        functions))

    # for functions where first parameter == 'df'
    # set the function as attribute of DataFrame
    for name, func, params in func_signatures:
        if len(params) > 0 and params[0] == 'df':
            setattr(DataFrame, name, func)

    # for functions where first parameter == 's'
    # set the function as attribute of Series
    for name, func, params in func_signatures:
        if len(params) > 0 and params[0] == 's':
            setattr(DataFrame, name, func)
except (ModuleNotFoundError, AttributeError):
    pass
