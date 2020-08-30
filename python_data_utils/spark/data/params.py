# coding: utf-8

"""
    description: Spark params
    author: Suraj Iyer
"""

__all__ = [
    'BaseParams'
    , 'CustomParamsWriter'
    , 'CustomParamsWritable'
    , 'CustomTypeConverters'
    , 'HasStartEndDates']

from pyspark.ml.param.shared import Param, Params
from pyspark.ml.util import DefaultParamsWriter, MLWritable
from pyspark.ml.param import TypeConverters

from dateutil.parser import parse
import datetime as dt
import pandas as pd


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
        elif isinstance(value, (list, tuple)):
            return dt.datetime(*value)
        else:
            raise TypeError("Datetime Param requires value of type datetime. Found %s." % type(value))


class HasStartEndDates(Params):
    """
    Mixin for params start_date & end_date.
    """
    start_date = Param(Params._dummy(), "start_date", "date: The start date", CustomTypeConverters.toDate)
    end_date = Param(Params._dummy(), "end_date", "date: The end date", CustomTypeConverters.toDate)

    def __init__(self):
        super(HasStartEndDates, self).__init__()

    def getStartDate(self):
        """
        Gets the value of start_date or its default value.
        """
        return self.getOrDefault(self.start_date)

    def getEndDate(self):
        """
        Gets the value of end_date or its default value.
        """
        return self.getOrDefault(self.end_date)
