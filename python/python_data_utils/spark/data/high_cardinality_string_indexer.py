__all__ = ['HighCardinalityStringIndexer']

import json
import logging
from typing import Dict, List

import pyspark.sql.functions as F
from pyspark import SparkContext, keyword_only
from pyspark.ml import Estimator, Transformer
from pyspark.ml.feature import IndexToString, StringIndexer
from pyspark.ml.param import Param, Params, TypeConverters
from pyspark.ml.param.shared import HasInputCols
from pyspark.sql import Column, DataFrame

from python_data_utils.spark.data.params import BaseParams


def with_meta(self, alias, meta):
    """
    In pyspark 2.1 there is no simple way to change the metdata of a column, that only became available in pyspark 2.2.
    This is a function that takes a column and modifies its metadata.
    :param self: A pyspark column
    :param alias:
    :param meta: New meta data for the column
    """
    sc = SparkContext._active_spark_context
    jmeta = sc._gateway.jvm.org.apache.spark.sql.types.Metadata
    return Column(getattr(self._jc, "as")(alias, jmeta.fromJson(json.dumps(meta))))


class HighCardinalityStringIndexerModel(Transformer):
    """
        A Transformer that transforms a DataFrame according to the logic obtained by fitting the HighCardinalityStringIndexer
    """

    def __init__(self,
                 dict_indexers: Dict,
                 inputCols: List[str],
                 outputCols: List[str],
                 dropInputCols: bool = False,
                 groupText: str = 'other',
                 returnIndexed: bool = True) -> None:
        """
        :param dict_indexers: A dictionary with each element being another dictionary containing an element 'indexer'
        with a StringIndexer object and an element 'n_to_keep' that indicates how many indexes to keep.
        :param inputCols: String columns that need to be indexed
        :param outputCols:
        :param dropInputCols: Should the input columns be dropped?
        :param groupText: String to use as replacement for the observations that need to be grouped.
        :param returnIndexed: If True, return the indexed columns. If False, return the columns with their String values,
        where only the grouped observations are changed.
        """
        super().__init__()
        self.dict_indexers = dict_indexers
        self.inputCols = inputCols
        self.outputCols = outputCols
        self.dropInputCols = dropInputCols
        self.groupText = groupText
        self.returnIndexed = returnIndexed

    @staticmethod
    def __logger() -> logging.Logger:
        """ Returns a reference to the logger to be used in this class

        Storing the logger as an attribute, and then referring to it in functions, can get it on the closure.
        Resulting in a lock object related to the logger to be included, which isn't serializable.
        """
        return logging.getLogger(__name__)

    def _transform(self, df) -> DataFrame:
        """
        :param df: A pyspark.sql.dataframe.DataFrame
        """

        # Apply string indexer
        for in_col, out_col in zip(self.inputCols, self.outputCols):
            self.__logger().info("Applying StringIndexer on col {}".format(in_col))
            df = self.dict_indexers[in_col]['indexer'].transform(df)
            n_to_keep = self.dict_indexers[in_col]['n_to_keep']
            # If elements occur below (threshold * number of rows), replace them with n_to_keep.
            this_meta = df.select(out_col).schema.fields[0].metadata
            if n_to_keep != len(this_meta['ml_attr']['vals']):
                this_meta['ml_attr']['vals'] = this_meta['ml_attr']['vals'][0:(n_to_keep + 1)]
                this_meta['ml_attr']['vals'][n_to_keep] = self.groupText
                self.__logger().info("Truncating number of categories of {} at {}".format(in_col, n_to_keep))
                df = df.withColumn(out_col,
                                   F.when(F.col(out_col) >= n_to_keep, F.lit(n_to_keep)).otherwise(
                                           F.col(out_col)))

            # add the new indexed column with correct metadata, remove original indexed column.
            df = df.withColumn(out_col,
                               with_meta(F.col(out_col), "", this_meta))

        if not self.returnIndexed:
            for output_col in self.outputCols:
                df = df.withColumnRenamed(output_col, output_col + '_temp')
                df = IndexToString(inputCol=output_col + '_temp', outputCol=output_col).transform(df)
                df = df.drop(output_col + '_temp')

        if self.dropInputCols:
            df = df.drop(*self.inputCols)

        return df


class HighCardinalityStringIndexer(Estimator, BaseParams, HasInputCols):
    """
    This is a class that can be used in combination with HighCardinalityStringIndexerTransformer to simply reduce the
    cardinality of high-cardinality categorical features, while simultaneously indexing them to be ready for use in a machine learning algorithm.

    For each column, it replaces all observations that occur in less then a 'threshold' fraction of the rows in the dataframe with 'groupText'.
    It does so by calling pyspark.ml.feature.StringIndexer on the column, and subsequently replacing values and changing the metadata of the column.
    By also changing the metadata we ensure that we can later extract the text values from the indexed columns if desired.

    Example --------------------------------------------------------------------

    >>> df = pd.DataFrame({'x1': ['a', 'b', 'a', 'b', 'c'],  # a: 0.4, b: 0.4, c: 0.2
    >>>                      'x2': ['a', 'b', 'a', 'b', 'a'],  # a: 0.6, b: 0.4, c: 0.0
    >>>                      'x3': ['a', 'a', 'a', 'a', 'a'],  # a: 1.0, b: 0.0, c: 0.0
    >>>                      'x4': ['a', 'b', 'c', 'd', 'e']})  # a: 0.2, b: 0.2, c: 0.2, d: 0.2, e: 0.2
    >>>
    >>> df = sqlContext.createDataFrame(df)
    >>> df.show()

    +---+---+---+---+
    | x1| x2| x3| x4|
    +---+---+---+---+
    |  a|  a|  a|  a|
    |  b|  b|  a|  b|
    |  a|  a|  a|  c|
    |  b|  b|  a|  d|
    |  c|  a|  a|  e|
    +---+---+---+---+

    >>> # Replace all values that occur in less than 25% of the rows.
    >>> indexer = HighCardinalityStringIndexer(inputCols=df.columns,
    >>>     outputCols=['ix_' + col for col in df_train.columns],
    >>>     threshold=0.25).fit(df)
    >>> df = indexer.transform(df)
    >>> df.show()

    +---+---+---+---+-----+-----+-----+-----+
    | x1| x2| x3| x4|ix_x1|ix_x2|ix_x3|ix_x4|
    +---+---+---+---+-----+-----+-----+-----+
    |  a|  a|  a|  a|  0.0|  0.0|  0.0|  0.0|
    |  b|  b|  a|  b|  1.0|  1.0|  0.0|  0.0|
    |  a|  a|  a|  c|  0.0|  0.0|  0.0|  0.0|
    |  b|  b|  a|  d|  1.0|  1.0|  0.0|  0.0|
    |  c|  a|  a|  e|  2.0|  0.0|  0.0|  0.0|
    +---+---+---+---+-----+-----+-----+-----+

    >>> # Optionally, obtain the labels after grouping
    >>> indexer = HighCardinalityStringIndexer(inputCols=df.columns,
    >>>     outputCols=['grouped_' + col for col in df_train.columns],
    >>>     threshold=0.25,
    >>>     returnIndexed=False).fit(df)
    >>> df = indexer.transform(df)
    >>> df.show()

    +---+---+---+---+----------+----------+----------+----------+
    | x1| x2| x3| x4|grouped_x1|grouped_x2|grouped_x3|grouped_x4|
    +---+---+---+---+----------+----------+----------+----------+
    |  a|  a|  a|  a|         a|         a|         a|     other|
    |  b|  b|  a|  b|         b|         b|         a|     other|
    |  a|  a|  a|  c|         a|         a|         a|     other|
    |  b|  b|  a|  d|         b|         b|         a|     other|
    |  c|  a|  a|  e|     other|         a|         a|     other|
    +---+---+---+---+----------+----------+----------+----------+

    """

    outputCols = \
        Param(Params._dummy(), "outputCols",
              "The output columns",
              typeConverter=TypeConverters.toListString)

    dropInputCols = \
        Param(Params._dummy(), "dropInputCols",
              "Drop the input columns?",
              typeConverter=TypeConverters.toBoolean)

    threshold = \
        Param(Params._dummy(), "threshold",
              "Group observations if they occur in less than threshold*100% of the rows",
              typeConverter=TypeConverters.toFloat)

    groupText = \
        Param(Params._dummy(), "groupText",
              "The text to use to bin grouped observations",
              typeConverter=TypeConverters.toString)

    returnIndexed = \
        Param(Params._dummy(), "returnIndexed",
              "Return the indexed columns, or their string representations?",
              typeConverter=TypeConverters.toBoolean)

    @keyword_only
    def __init__(self,
                 inputCols: List[str],
                 outputCols: List[str],
                 dropInputCols: bool = False,
                 threshold: float = .01,
                 groupText: str = 'other',
                 returnIndexed: bool = True) -> None:
        """
        :param inputCols: String columns that need to be indexed
        :param dropInputCols: Should the input columns be dropped?
        :param threshold: Replace all observations that occur in less then a 'threshold' fraction of the rows.
        :param groupText: String to use as replacement for the observations that are binned because they occur in low frequency.
        :param returnIndexed: If True, return the indexed columns. If False, return the columns with their String values,
        where only the grouped observations are changed.
        """
        super().__init__()
        self._setDefault(inputCols=None)
        self._setDefault(outputCols=None)
        self._setDefault(dropInputCols=False)
        self._setDefault(threshold=0.01)
        self._setDefault(groupText='other')
        self._setDefault(returnIndexed=True)
        kwargs = self._get_init_args_as_kwargs()
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self,
                  inputCols: List[str],
                  outputCols: List[str],
                  dropInputCols: bool = False,
                  threshold: float = .01,
                  groupText: str = 'other',
                  returnIndexed: bool = True):
        kwargs = self._get_params_args_as_kwargs()
        return self._set(**kwargs)

    def getOutputCols(self) -> List[str]:
        return self.getOrDefault(self.outputCols)

    def getDropInputCols(self) -> bool:
        return self.getOrDefault(self.dropInputCols)

    def getThreshold(self) -> float:
        return self.getOrDefault(self.threshold)

    def getGroupText(self) -> str:
        return self.getOrDefault(self.groupText)

    def getReturnIndexed(self) -> bool:
        return self.getOrDefault(self.returnIndexed)

    @staticmethod
    def __logger() -> logging.Logger:
        """ Returns a reference to the logger to be used in this class

        Storing the logger as an attribute, and then referring to it in functions, can get it on the closure.
        Resulting in a lock object related to the logger to be included, which isn't serializable.
        """
        return logging.getLogger(__name__)

    def _fit(self, df) -> HighCardinalityStringIndexerModel:
        """
        :param df: A pyspark.sql.dataframe.DataFrame
        """
        total = df.count()

        # For each column, calculate the number of unique elements to keep
        dict_indexers = {}
        for in_col, out_col in zip(self.getInputCols(), self.getOutputCols()):
            self.__logger().info("Fitting StringIndexer on '{}'".format(in_col))
            string_indexer = StringIndexer(inputCol=in_col,
                                           outputCol=out_col,
                                           handleInvalid='skip').fit(df)
            self.__logger().info("Determining number of categories of '{}' to keep".format(in_col))
            n_to_keep = df.groupby(in_col) \
                .agg((F.count(in_col) / total).alias('perc')) \
                .filter(F.col('perc') > self.getThreshold()) \
                .count()
            self.__logger().info("Finished processing '{}'.".format(in_col))
            if n_to_keep == 0:
                self.__logger().info("Every unique value of "
                                     "{} occurs less than fraction {} times count {}".format(in_col,
                                                                                             self.getThreshold(),
                                                                                             total)
                                     + "Therefore should exclude the column from the output")  # TODO: exclude it
            dict_indexers[in_col] = {'indexer': string_indexer, 'n_to_keep': n_to_keep}

        return HighCardinalityStringIndexerModel(
                dict_indexers=dict_indexers,
                inputCols=self.getOrDefault(self.inputCols),
                outputCols=self.getOrDefault(self.outputCols),
                dropInputCols=self.getOrDefault(self.dropInputCols),
                groupText=self.getOrDefault(self.groupText),
                returnIndexed=self.getOrDefault(self.returnIndexed)
        )
