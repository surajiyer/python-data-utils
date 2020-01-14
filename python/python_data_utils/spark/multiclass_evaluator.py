from pyspark.ml.evaluation import Evaluator
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import col


class MulticlassEvaluator(Evaluator):
    def __init__(self, metricName, predictionCol="prediction",
                 labelCol="label", **metricParams):
        assert metricName in [m for m in dir(MulticlassMetrics) if not m.startswith('__') and m != 'call'],\
            f'Unsupported metric: {metricName}'
        params = locals()
        del params['self']
        self.__dict__ = params

    def _evaluate(self, dataset):
        rdd = dataset\
            .select(
                col(self.predictionCol).cast(DoubleType()),
                col(self.labelCol).cast(DoubleType()))\
            .rdd.map(tuple)
        return getattr(
            MulticlassMetrics(rdd), self.metricName)(**self.metricParams)

    def isLargerBetter(self):
        return True
