# coding: utf-8

"""
    description: Decorators for pandas DataFrame objects
    author: Suraj Iyer
"""

import python_data_utils.pandas.utils as pdu
import importlib.resources as pkg_resources


try:
    from pandas import DataFrame, Series

    with pkg_resources.path('python_data_utils.pandas', 'utils.py') as p:
        with open(p) as f:
            f = f.readlines()

            # Find function names which takes "df" as first input
            # parameter for DataFrame
            df_functions = [
                l.split(" ")[1].split("(")[0] for l in f
                if l.startswith('def') and ('(df,' in l or '(df)' in l)]

            # Find function names which takes "s" as first input
            # parameter for Series
            series_functions = [
                l.split(" ")[1].split("(")[0] for l in f
                if l.startswith('def') and ('(s,' in l or '(s)' in l)]

        for fun in df_functions:
            setattr(DataFrame, fun, getattr(pdu, fun))

        for fun in series_functions:
            setattr(Series, fun, getattr(pdu, fun))
except (ModuleNotFoundError, AttributeError):
    pass
