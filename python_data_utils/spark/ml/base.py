__all__ = ['BaseCVModel', 'BinaryClassCVModel', 'RegressionCVModel']


import inspect
import json
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.base import Estimator, Model
from pyspark.ml.evaluation import BinaryClassificationEvaluator, Evaluator, RegressionEvaluator
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.tuning import CrossValidator, CrossValidatorModel, ParamGridBuilder
from pyspark.sql import DataFrame
from python_data_utils.logging import get_logger
from python_data_utils.spark.evaluation.multiclass import MulticlassEvaluator
from typing import Any, Dict, Callable, List, Optional, Tuple, Union


class Metrics(type):
    @staticmethod
    def register(name: str = None):
        """
        Decorator function to register a metric name
        value to a input function
        """
        def inner(func):
            assert inspect.getfullargspec(func).args[:2] == ['self', 'predictions']
            func.metric_name = name
            return func
        return inner

    def __new__(meta, name, bases, class_dict):
        """
        Metaclass instantiation to put all functions
        registered as metrics into a instance
        dictionary variable `metrics`
        """
        klass = super().__new__(meta, name, bases, class_dict)
        i = 0
        for key in class_dict:
            value = class_dict[key]
            if isinstance(value, Callable) and hasattr(value, 'metric_name'):
                if value.metric_name is None:
                    klass.metrics[i] = value
                    i += 1
                else:
                    klass.metrics[value.metric_name] = value
        return klass


class BaseCVModel(metaclass=Metrics):

    best_params : Dict[str, Any] = None
    __best_model : PipelineModel = None
    logger = get_logger()
    metrics : Dict[str, Callable] = dict()

    def __init__(self, estimator=None, evaluator=None):
        self.set_params(estimator, evaluator)

    def set_params(self, estimator=None, evaluator=None):
        assert estimator is None or isinstance(estimator, Estimator),\
            'estimator must be a pyspark.ml.base.Estimator.'
        assert evaluator is None or isinstance(evaluator, Evaluator),\
            'evaluator must be a pyspark.ml.base.Evaluator.'
        if estimator is not None:
            self.estimator = estimator
        if evaluator is not None:
            self.evaluator = evaluator

    def _get_features(self, df: Optional[DataFrame] = None):
        """
        Returns three lists of feature names to be used in
        the model training. Specifically binary, numeric
        (continuous), categorical features in that order.

        Returns
        -------
        binary: List[str]
        numeric: List[str]
        categorical: List[str]
        """
        raise NotImplementedError

    def train(
            self, df: DataFrame, params_map: Optional[Dict[str, List[Any]]] = None,
            num_folds: Optional[int] = 10, collect_sub_models: Optional[bool] = False,
            return_cv: Optional[bool] = False) -> Union[
                PipelineModel, Tuple[PipelineModel, CrossValidatorModel]]:
        """
        Train model.

        Params
        ------
        df: Spark DataFrame
            Input train data

        params_map: Optional[Dict[str, List[Any]]] (default=None)
            Parameters mapping to grid search over

        num_folds: Optional[int] (default=10)
            Number of cross-validation folds

        collect_sub_models: Optional[bool] (default=False)
            Collect models per fold per parameter
            combination

        return_cv: Optional[bool] (default=False)
            Additionally return the CrossValidatorModel
            object or not

        Returns
        -------
            self: PipelineModel
                The (best) model trained on df.
            cv_model: Optional[CrossValidatorModel]
                The CrossValidatorModel object.
        """
        # get input features
        binary, numeric, categorical = self._get_features(df)

        # convert categorical to numeric labels
        indexed_cols = [f'{c}_idx' for c in categorical]
        indexers = [StringIndexer(inputCol=c[:-6], outputCol=c) for c in indexed_cols]
        self.features = binary + numeric + indexed_cols
        self.logger.info(f'Final model features list: {self.features}')

        # assemble features into feature vector
        assembler = VectorAssembler(
            inputCols=self.features,
            outputCol=self.estimator.getFeaturesCol())
        p = Pipeline(stages=indexers + [assembler]).fit(df)
        self.logger.info('Index and vector assemble features')
        df = p.transform(df)\
            .select(self.estimator.getFeaturesCol(), self.estimator.getLabelCol())

        # if provided, set estimator params map
        if params_map:
            self.params_map = params_map

        # run cross-validation and choose the best set of parameters
        self.logger.info('Start Cross Validation')
        cv_params = {
            'estimator': self.estimator,
            'estimatorParamMaps': self.__params_grid,
            'evaluator': self.evaluator,
            'numFolds': num_folds,
            'collectSubModels': collect_sub_models
        }
        cv_model = CrossValidator(**cv_params).fit(df)

        # set the best model
        p.stages.append(cv_model.bestModel)
        self.best_model = p
        self.logger.info(f'Set the best model with best params: {self.best_params}')

        if return_cv:
            return self.best_model, cv_model
        else:
            return self.best_model

    def test(self, df: DataFrame) -> Tuple[DataFrame, Dict[str, Any]]:
        """
        Test the best model found so far.

        Params
        ------
        df: Spark DataFrame
            Input test data

        Returns
        -------
        predictions: DataFrame
            DataFrame `df` with added `prediction` column
        results: Dict[str, Any]
            Dictionary with any results from testing the model
            e.g., metrics, feature importances, plots etc.
        """
        assert self.best_model is not None, 'Call train() or load() first.'
        df = df.withColumnRenamed(self.estimator.getLabelCol(), 'label')
        self.logger.info('Get model predictions')
        predictions = self.best_model.transform(df)

        # execute all metrics
        results = {'best_params': self.best_params}
        for name, metric in self.metrics.items():
            results.update({name: metric(self, predictions)})
        self.logger.info(f'Results: {results}')

        return predictions, results

    def train_final(self, df: DataFrame):
        """
        Train final model using best parameters found
        on given dataframe.

        Params
        ------
        df: Spark DataFrame
            (Ideally) both train and test combined
        """
        assert self.best_params is not None, 'Call train() or load() first.'
        est = self.estimator\
            .setParams(**self.best_params)
#         for k, v in self.best_params.items():
#             getattr(est, 'set' + k[0].upper() + k[1:])(v)
        est = Pipeline(stages=self.best_model.stages[:-1] + [est])
        self.best_model = est.fit(df)

    def score(self, df: DataFrame) -> DataFrame:
        """
        Score on given dataset using best model
        found so far.

        Params
        ------
        df: Spark DataFrame
            Input data to score

        Returns
        -------
        df: Spark DataFrame
            Same as input with additional
            prediction columns
        """
        return self.best_model.transform(df)

    @property
    def params_map(self) -> Dict[str, List[Any]]:
        return self.__params_map

    @params_map.setter
    def params_map(self, params_map : Dict[str, List[Any]]):
        assert isinstance(params_map, dict)
        self.__params_map = params_map
        self.__params_grid = ParamGridBuilder()
        for k, v in params_map.items():
            self.__params_grid.addGrid(getattr(self.estimator, k), v)
        self.__params_grid = self.__params_grid.build()

    @property
    def best_model(self) -> PipelineModel:
        return self.__best_model

    @best_model.setter
    def best_model(self, model):
        assert isinstance(model, PipelineModel),\
            'model must be of type PipelineModel.'
        self.__best_model = model
        est = model.stages[-1]
        # coalesce = lambda *x: next(y for y in x if y is not None)
        if est._java_obj.parent() is not None\
            and self.params_map is not None:
            self.best_params = {
                k: getattr(
                    est._java_obj.parent()
                    , 'get' + k[0].upper() + k[1:])()
                for k in self.params_map.keys()
            }
        elif hasattr(est, 'extractParamMap'):
            self.best_params = {param[0].name: param[1] for param in est.extractParamMap().items()}
        else:
            self.best_params = None

    def save(self, path: str):
        self.best_model.save(path)

        # save additional model metadata
        metadata = {}
        if hasattr(self, 'best_params'):
            metadata.update({'best_params': self.best_params})
        if hasattr(self, 'features'):
            metadata.update({'features': self.features})
        with open(path + '/BaseCVModel_metadata', 'w') as fp:
            json.dump(metadata, fp, separators=[',',  ':'])

    def load(self, path: str):
        self.best_model = PipelineModel.load(path)

        # load additional model metadata
        import os
        metadata_exists = (
            os.path.isfile(path + '/BaseCVModel_metadata'),
            os.path.isfile(path + '/BaseModel_metadata')
        )
        if metadata_exists[0]:
            path = path + '/BaseCVModel_metadata'
        elif metadata_exists[1]:
            # for backward compatibility
            path = path + '/BaseModel_metadata'
        if any(metadata_exists):
            with open(path, 'r') as fp:
                metadata = json.load(fp)
            if 'best_params' in metadata:
                self.best_params = metadata['best_params']
            if 'features' in metadata:
                self.features = metadata['features']

        return self

    @Metrics.register()
    def self_evaluator(self, predictions: DataFrame):
        if hasattr(self, 'evaluator')\
            and hasattr(self.evaluator, 'getMetricName'):
            return {
                self.evaluator.getMetricName(): self.evaluator.evaluate(predictions)}


class BinaryClassCVModel(BaseCVModel):

    def __init__(self, estimator=None, evaluator=None):
        evaluator = BinaryClassificationEvaluator()\
            if not evaluator else evaluator
        super().__init__(estimator, evaluator)

    @Metrics.register('binary_classification_report')
    def binary_classification_report(self, predictions: DataFrame):
        self.logger.info('Get tp, tn, fp, fn')
        tp = predictions[(predictions.label == 1) & (predictions.prediction == 1)].count()
        tn = predictions[(predictions.label == 0) & (predictions.prediction == 0)].count()
        fp = predictions[(predictions.label == 0) & (predictions.prediction == 1)].count()
        fn = predictions[(predictions.label == 1) & (predictions.prediction == 0)].count()
        return {
            'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
            'accuracy': (tp + tn) / (tp + tn + fp + fn),
            'recall': tp / (tp + fn) if (tp + fn) else -1.,
            'precision': tp / (tp + fp) if (tp + fp) else -1.}

    @Metrics.register('F1(1)')
    def f1_score(self, predictions: DataFrame):
        return MulticlassEvaluator(metricName='weightedFMeasure').evaluate(predictions)

    @Metrics.register('AUROC')
    def auroc(self, predictions: DataFrame):
        return BinaryClassificationEvaluator().evaluate(predictions)


class RegressionCVModel(BaseCVModel):

    def __init__(self, estimator=None, evaluator=None):
        evaluator = RegressionEvaluator()\
            if not evaluator else evaluator
        super().__init__(estimator, evaluator)
