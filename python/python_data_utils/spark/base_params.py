# coding=utf-8
from pyspark.ml.param.shared import Params


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
