__all__ = ['RandomForestBinaryModel']


from pyspark.sql import DataFrame
from pyspark.ml.classification import RandomForestClassifier
from python_data_utils.spark.evaluation.multiclass import MulticlassEvaluator
from python_data_utils.spark.ml.base import BinaryClassCVModel, Metrics


class RandomForestBinaryModel(BinaryClassCVModel):

    def __init__(
            self, *, estimator=None, evaluator=None,
            label_col: str = 'label', params_map=None):
        estimator = RandomForestClassifier(labelCol=label_col)\
            if not estimator else estimator
        assert isinstance(estimator, RandomForestClassifier)
        evaluator = MulticlassEvaluator(metricName='weightedFMeasure')\
            if not evaluator else evaluator
        super().__init__(estimator=estimator, evaluator=evaluator)
        self.params_map = {
            'maxDepth': [5, 10, 20],
            'numTrees': [20, 30, 40, 50],
            'minInstancesPerNode': [1, 2, 3]
        } if not params_map else params_map

    @Metrics.register('feature_importances')
    def feature_importances(self, predictions):
        self.logger.info('Get feature importances')
        feature_importance = self.best_model.stages[-1].featureImportances
        feature_importance = sorted([
            (self.features[i], fi)
            for i, fi in enumerate(feature_importance)]
        , key=lambda x: -x[1])
        return feature_importance

    def get_feature_importances(self, df: DataFrame):
        """
        Takes in a feature importance from a random forest / GBT model and map it to the column names
        Output as a pandas dataframe for easy reading

        Params
        ------
        df: DataFrame
            Example dataframe with same schema as model input.

        Usage
        ----------
        >>> rf = RandomForestClassifier(featuresCol="features")
        >>> mod = rf.fit(train)
        >>> get_feature_importances(train)
        """         
        import pandas as pd
        featureImp = self.best_model.stages[-1].featureImportances
        featuresCol = self.estimator.getFeaturesCol()
        features = df.schema[featuresCol].metadata["ml_attr"]["attrs"]
        list_extract = [features[i] for i in features]
        varlist = pd.DataFrame(list_extract)
        varlist['score'] = varlist['idx'].apply(lambda x: featureImp[x])
        return varlist.sort_values('score', ascending=False)
