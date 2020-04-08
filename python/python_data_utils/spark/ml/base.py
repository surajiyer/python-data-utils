__all__ = ['BaseModel', 'verify_best_model']

from pyspark.ml.base import Estimator, Model
from pyspark.sql import DataFrame
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.tuning import CrossValidatorModel
from typing import Union, Tuple


def verify_best_model(func):
    def wrapper(*args, **kwargs):
        self = args[0]
        assert isinstance(self, BaseModel)
        assert isinstance(self.best_model, PipelineModel),\
            'Call train() first.'
        assert isinstance(self.best_model.stages[-1], self.model_cls),\
            f'The final stage of the best model must be of type {self.model_cls}.'
        return func(*args, **kwargs)
    return wrapper


class BaseModel:

    def __init__(self, estimator, model_cls, label: str=None):
        assert issubclass(estimator, Estimator), 'estimator must be a pyspark.ml.base.Estimator type.'
        assert issubclass(model_cls, Model), 'model_cls must be a pyspark.ml.base.Model type.'
        self.estimator = estimator
        self.model_cls = model_cls
        self.label = 'label' if not label else label
        self.best_model = None
        self.best_params = None

    def train(
            self, df: DataFrame, param_maps: dict=None,
            return_cv: bool=False) -> Union[PipelineModel, Tuple[PipelineModel, CrossValidatorModel]]:
        """
        return:
            best_model: PipelineModel
                The (best) model trained on df.
            cv_model: Optional[CrossValidatorModel]
                The CrossValidatorModel object.
        """
        raise NotImplementedError()

    @verify_best_model
    def test(self, df: DataFrame) -> DataFrame, dict:
        """
        return:
            test_predictions: DataFrame
                DataFrame with "prediction" column
            results: dict
                Dictionary with any kind of results from testing the model
                e.g., metrics, feature importances, plots etc.
        """
        raise NotImplementedError()

    @verify_best_model
    def score(self, df: DataFrame) -> DataFrame:
        return self.best_model.transform(df)

    @verify_best_model
    def train_final(self, df: DataFrame):
        model = self.estimator()\
            .setParams(**self.best_params)
#         for k, v in self.best_params.items():
#             getattr(model, 'set' + k[0].upper() + k[1:])(v)
        model = Pipeline(stages=self.best_model.stages[:-1] + [model])
        self._set_best_model(model.fit(df))
        return self

    def _set_best_model(self, model):
        assert isinstance(model, PipelineModel),\
            'model must be of type PipelineModel.'
        assert isinstance(model.stages[-1], self.model_cls),\
            f'The final stage of model must be of type {self.model_cls}.'
        self.best_model = model
        self.best_params = {
            k.name: getattr(model.stages[-1]._java_obj, 'get' + k.name[0].upper() + k.name[1:])()
            for k in self._param_grid[0].keys()
        }
        return self

    @verify_best_model
    def save(self, path: str):
        self.best_model.save(path)
        return self

    def load(self, path: str):
        self._set_best_model(PipelineModel.load(path))
        return self
