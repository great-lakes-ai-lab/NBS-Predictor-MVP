from typing import Union

from sklearn.pipeline import Pipeline

from src.step3_modeling.modeling import ModelBase
from src.step4_postprocessing.postprocessing import PostprocessingPipeline

__all__ = ["ModelPipeline"]


class ModelPipeline(object):
    """
    A pipeline object that composes a preprocessor, model, and postprocessor together
    to make predictions from the lake data using a single object. This differs from
    the scikit-learning pipeline by applying the postprocessing steps to model predictions.
    However, scikit-learn pipelines are supported for both the model object and preprocessor.

    Attributes:
    -----------

    preprocessor : Pipeline
        The preprocessing pipeline.

    model : ModelBase
        The base machine learning model. Shoul

    postprocessor: PostprocessingPipeline, optional
        The postprocessing pipeline. Defaults to None.

    """

    def __init__(
        self,
        model: Union[ModelBase, Pipeline],
        preprocessor: Pipeline = None,
        postprocessor: PostprocessingPipeline = None,
    ):
        self.preprocessor = preprocessor
        self.model = model
        self.postprocessor = postprocessor
        self.predictive_model = Pipeline(
            [("preprocess", self.preprocessor), ("model", self.model)]
        )

    def fit(self, X, y, *args, **kwargs):
        return self.predictive_model.fit(X, y, *args, **kwargs)

    def predict(self, X, y=None, *args, **kwargs):
        return self.predictive_model.predict(X, y=y, *args, **kwargs)

    def fit_predict(self, X, y, *args, **kwargs):
        return self.predictive_model.fit_predict(X, y, *args, **kwargs)
