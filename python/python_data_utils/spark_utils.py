# coding: utf-8

"""
    description: Pandas utility functions and classes
    author: Suraj Iyer
"""

import pyspark.sql.functions as F
from pyspark.sql import DataFrame
from typing import Iterable
from pyspark.sql.types import IntegerType


def melt(
        df: DataFrame,
        id_vars: Iterable[str], value_vars: Iterable[str],
        var_name: str="variable", value_name: str="value") -> DataFrame:
    """Convert :class:`DataFrame` from wide to long format."""
    # Create array<struct<variable: str, value: ...>>
    _vars_and_vals = F.array(*(
        F.struct(F.lit(c).alias(var_name), F.col(c).alias(value_name))
        for c in value_vars))

    # Add to the DataFrame and explode
    _tmp = df.withColumn("_vars_and_vals", F.explode(_vars_and_vals))

    cols = id_vars + [
        F.col("_vars_and_vals")[x].alias(x) for x in [var_name, value_name]]
    return _tmp.select(*cols)


def one_hot_encode(df: DataFrame, column_name: str, prefix: str = None):
    categories = df.select(column_name).distinct().rdd.flatMap(lambda x: x).collect()
    columns = []
    for category in categories:
        function = F.udf(lambda item: 1 if item == category else 0, IntegerType())
        if isinstance(prefix, str):
            new_column_name = prefix + '_' + category
        else:
            new_column_name = category
        df = df.withColumn(new_column_name, function(F.col(column_name)))
        columns.append(new_column_name)
    return df, columns
