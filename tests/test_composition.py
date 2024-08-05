import pytest
from sklearn.pipeline import Pipeline
import xarray as xr

from src.step2_preprocessing.preprocessing import XArrayStandardScaler
from src.step3_modeling.ensemble import DefaultEnsemble
from src.step4_postprocessing.postprocessing import (
    PostprocessingPipeline,
    ExceedanceProbClosure,
)
from src.composition import ModelPipeline


@pytest.fixture
def preprocessor():
    return Pipeline(steps=[("scale", XArrayStandardScaler())])


@pytest.fixture
def postprocessor(lake_data):
    return PostprocessingPipeline(
        steps=[
            ("exceedance", ExceedanceProbClosure(lake_data.sel(variable="rnbs"))),
        ]
    )


def test_model_pipeline(snapshot, preprocessor, postprocessor):
    forecast_steps = 12
    model = DefaultEnsemble()
    full_pipeline = ModelPipeline(
        preprocessor=preprocessor, model=model, postprocessor=postprocessor
    )
    full_pipeline.fit(snapshot.train_x, snapshot.train_y)

    output_predictions = full_pipeline.predict(
        snapshot.test_x, forecast_steps=forecast_steps
    )
    non_processed_predictions = full_pipeline.predictive_model.predict(
        snapshot.test_x, forecast_steps=forecast_steps
    )

    assert non_processed_predictions.shape == output_predictions.shape
    assert not xr.testing.assert_allclose(non_processed_predictions, output_predictions)
