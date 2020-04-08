# coding: utf-8

"""
    description: Spark utility functions and classes
    author: Suraj Iyer
"""

__all__ = [
    'empty_df'
    , 'melt'
    , 'one_hot_encode'
    , 'BaseParams'
    , 'CustomParamsWriter'
    , 'CustomParamsWritable'
    , 'CustomTypeConverters']

import pyspark.sql.functions as F
from pyspark.sql import DataFrame
from pyspark.sql.types import IntegerType, StructType
from pyspark.ml.param.shared import Params
from pyspark.ml.util import DefaultParamsWriter, MLWritable
from pyspark.ml.param import TypeConverters
import pandas as pd
from dateutil.parser import parse
import datetime as dt
from typing import Iterable, Tuple


def empty_df(spark):
    schema = StructType([])
    return spark.createDataFrame(spark.sparkContext.emptyRDD(), schema)


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


def one_hot_encode(
        df: DataFrame,
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


# noinspection PyUnresolvedReferences
class BaseParams(Params):
    """ Helper class to get method arguments as kwargs, both in Spark 2.1.0 and 2.1.1+

    The implementation of the _input_kwargs extension method has changed since Spark 2.1.1
    These methods allow your code to work in both Spark versions.
    """

    def _get_params_args_as_kwargs(self):
        if hasattr(self, '_input_kwargs'):
            # the _input_kwargs extension method as implemented in 2.1.1 and higher
            # (which we use in dev, as it is compatible wih python 3.6)
            kwargs = self._input_kwargs
        else:
            # the _input_kwargs extension method as implemented in 2.1.0
            # (which we have on the cluster, but is only compatible with python 3.5 and lower)
            kwargs = self.setParams._input_kwargs
        return {k: v for k, v in kwargs.items() if v is not None}

    def _get_init_args_as_kwargs(self):
        if hasattr(self, '_input_kwargs'):
            # the _input_kwargs extension method as implemented in 2.1.1 and higher
            # (which we use in dev, as it is compatible wih python 3.6)
            kwargs = self._input_kwargs
        else:
            # the _input_kwargs extension method as implemented in 2.1.0
            # (which we have on the cluster, but is only compatible with python 3.5 and lower)
            kwargs = self.__init__._input_kwargs
        return {k: v for k, v in kwargs.items() if v is not None}


class CustomParamsWriter(DefaultParamsWriter):
    def saveImpl(self, path):
        params = self.instance.extractParamMap()
        jsonParams = {}
        for p in params:
            if isinstance(params[p], pd.DataFrame):
                jsonParams[p.name] = params[p].to_json()
            elif isinstance(params[p], dt.datetime):
                jsonParams[p.name] = str(params[p])
            else:
                jsonParams[p.name] = params[p]
        DefaultParamsWriter.saveMetadata(
            self.instance, path, self.sc, paramMap=jsonParams)


class CustomParamsWritable(MLWritable):
    def write(self):
        from pyspark.ml.param import Params

        if isinstance(self, Params):
            return CustomParamsWriter(self)
        else:
            raise TypeError("Cannot use CustomParamsWritable with type %s because it does not " +
                            " extend Params.", type(self))


class CustomTypeConverters(TypeConverters):

    @staticmethod
    def JSONtoPandas(value):
        if isinstance(value, pd.DataFrame):
            return value
        elif isinstance(value, str):
            return pd.read_json(value)
        else:
            raise TypeError("pd.DataFrame Param requires value to be a JSON string (str). Found %s." % type(value))

    @staticmethod
    def toDate(value):
        if isinstance(value, dt.datetime):
            return value
        elif isinstance(value, str):
            return parse(value)
        else:
            raise TypeError("Datetime Param requires value of type datetime. Found %s." % type(value))
