__all__ = ['LightGBMRegressorModel']


from mmlspark.lightgbm.LightGBMRegressor import LightGBMRegressor, LightGBMRegressionModel
from mmlspark.train import ComputeModelStatistics
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import DataFrame
import pyspark.sql.functions as F
from python_data_utils.spark.ml.base import BinaryClassCVModel, Metrics, RegressionCVModel


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


class LightGBMRegressorModel(RegressionCVModel):

    def __init__(
            self, *, estimator=None, evaluator=None,
            label_col: str = 'label', params_map=None):
        estimator = LightGBMRegressor(
            objective='regression_l1',
            # earlyStoppingRound=3,
            # validationIndicatorCol='is_val',
            labelCol=label_col)\
            if not estimator else estimator
        assert isinstance(estimator, LightGBMRegressor)
        evaluator = RegressionEvaluator(metricName='mae')\
            if not evaluator else evaluator
        super().__init__(estimator, evaluator)
        self.params_map = {
            'baggingFraction': [.8, .9, 1.],
            'featureFraction': [.5, .75, 1.],
            'lambdaL1': [.1, .2, .3],
            'learningRate': [0.01, 0.1],
            'maxDepth': [-1, 3, 12],
            'numIterations': [200],
            'numLeaves': [31]
        } if not params_map else params_map

    @Metrics.register('regression_metrics')
    def regression_metrics(self, predictions: DataFrame):
        return ComputeModelStatistics(
            evaluationMetric='regression',
            labelCol=self.estimator.getLabelCol(),
            scoresCol=self.estimator.getPredictionCol())\
            .transform(predictions)\
            .toPandas().to_dict(orient='list')

    @Metrics.register('feature_importances')
    def feature_importances(self, predictions: DataFrame):
        feat_importances = pd.DataFrame(sorted(zip(
            self.best_model.stages[-1].getFeatureImportances()
            , self.features)), columns=['Value', 'Feature'])

        # plot feature importance
        _, ax = plt.subplots(figsize=(20, 10))
        ax = sns.barplot(
            x="Value",
            y="Feature",
            ax=ax,
            data=feat_importances.sort_values(
                by="Value", ascending=False))
        ax.set_title('LightGBM Features (avg over folds)')
        plt.tight_layout()

        return {'data': feat_importances, 'plot': ax}

    @Metrics.register('residuals_plot')
    def residuals_plot(self, predictions: DataFrame):
        # plot residuals
        predictions = predictions.withColumn(
            '_resid', F.col(self.estimator.getPredictionCol())\
                - F.col(self.estimator.getLabelCol()))
        return predictions.select('_resid').toPandas().hist()
