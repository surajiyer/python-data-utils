from pyspark.ml import Transformer
from pyspark.ml.pipeline import PipelineModel, PipelineSharedReadWrite
from pyspark.ml.util import MLReader, DefaultParamsReader
from pyspark.sql import DataFrame
from typing import List


class AddResidualConnectionReader(MLReader):

    def __init__(self, cls):
        super().__init__()
        self.cls = cls

    def load(self, path):
        metadata = DefaultParamsReader.loadMetadata(path, self.sc)
        uid, stages = PipelineSharedReadWrite.load(metadata, self.sc, path)
        return AddResidualConnection(stages=stages)._resetUid(uid)


class AddResidualConnection(PipelineModel):

    def __init__(self, stages: List[Transformer]):
        assert 0 < len(stages) <= 2, "stages must be a List[Transformer] of (max 2) transformers."
        super().__init__(stages)

    def _transform(self, df: DataFrame):
        result = self.stages[0].transform(df)
        if len(self.stages) > 1:
            result = self.stages[1].transform((result, df))
        return result

    @classmethod
    def read(cls):
        return AddResidualConnectionReader(cls)
