# coding: utf-8

"""
    description: Spark utility functions and classes
    author: Suraj Iyer
"""

__all__ = [
    'empty_df'
    , 'melt'
    , 'one_hot_encode'
    , 'explode_multiple'
    , 'count_per_partition']

import pyspark.sql.functions as F
from pyspark.sql import DataFrame
from pyspark.sql.types import IntegerType, StructType
from typing import Iterable, Tuple


def empty_df(spark):
    schema = StructType([])
    return spark.createDataFrame(spark.sparkContext.emptyRDD(), schema)


def melt(
        df: DataFrame, *,
        id_vars: Iterable[str], value_vars: Iterable[str],
        var_name: str = "variable", value_name: str = "value") -> DataFrame:
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


def one_hot_encode(
        df: DataFrame, *,
        column_name: str, prefix: str = None) -> Tuple[DataFrame, Tuple]:
    """
    :return df: DataFrame
        Spark DF with new one-hot encoded columns.
    :return columns:
        Tuple of new column names.
    """
    categories = df\
        .select(column_name).distinct()\
        .rdd.flatMap(lambda x: x).collect()
    columns = []
    for category in categories:
        function = F.udf(
            lambda item: 1 if item == category else 0, IntegerType())
        if isinstance(prefix, str):
            new_column_name = prefix + '_' + category
        else:
            new_column_name = category
        df = df.withColumn(new_column_name, function(F.col(column_name)))
        columns.append(new_column_name)
    return df, tuple(columns)


def explode_multiple(df: DataFrame, *array_col_names):
    # https://stackoverflow.com/questions/59235308/explode-two-pyspark-arrays-and-keep-elements-from-same-positions
    # tranform array1, array2 => [struct(element1, element2)]
    assert len(array_col_names) > 0
    assert all(c.split('.')[0] in df.columns for c in array_col_names)
    elements = ", ".join([
        f"{array_col_names[j]}[i] as element{j + 1}" for j in range(1, len(array_col_names))])
    transform_expr = f"transform({array_col_names[0]}, (x, i) -> struct(x as element1, {elements}))"

    # explode transformed arrays and extract values of element1 and element2
    df = df.withColumn("merged_arrays", F.explode(F.expr(transform_expr)))
    for i in range(len(array_col_names)):
        df = df.withColumn(
            f"element{i + 1}", F.col(f"merged_arrays.element{i + 1}"))

    df = df.drop("merged_arrays")
    return df


def count_per_partition(df: DataFrame):
    return df.rdd.mapPartitions(lambda it: [sum(1 for _ in it)]).collect()
