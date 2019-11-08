# coding: utf-8

"""
    description: Decorators for pandas DataFrame objects
    author: Suraj Iyer
"""

__package__ = "python_data_utils.decorators"

from pandas import DataFrame, Series
from ..pandas import utils as pdu
import os.path as path


with open(path.join(path.dirname(__file__), "..", "pandas", "utils.py")) as f:
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
