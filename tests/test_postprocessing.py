import numpy as np
import pytest
import xarray as xr

from src.modeling.ensemble import DefaultEnsemble
from src.postprocessing.postprocessing import (
    ExceedanceProbClosure,
    PostprocessingPipeline,
)


@pytest.fixture
def fake_data():

    fake_train = xr.DataArray(
        np.random.standard_normal(size=(250, 4)),
        dims=["Date", "lake"],
        coords={"Date": range(250), "lake": ["sup", "mic_hur", "eri", "ont"]},
    )

    fake_forecast = xr.DataArray(
        np.random.standard_normal(size=(10, 4)),
        dims=["Date", "lake"],
        coords={"Date": range(250, 260), "lake": ["sup", "mic_hur", "eri", "ont"]},
    )

    return fake_train, fake_forecast


def test_ecdf(fake_data):

    train, forecast = fake_data

    exceedence_fn = ExceedanceProbClosure(train)
    assert exceedence_fn(forecast).shape == forecast.shape


def test_label_return(fake_data):
    train, forecast = fake_data
    exceedence_fn = ExceedanceProbClosure(train, as_labels=True)

    output_labels = exceedence_fn(forecast)

    assert output_labels.isin(["Wet", "Normal", "Dry"]).all()


def test_postprocessing_pipeline(snapshot):
    model = DefaultEnsemble()
    model.fit(snapshot.train_x, snapshot.train_y)
    predictions = model.predict(snapshot.test_x)

    exceedance_fn = ExceedanceProbClosure(snapshot.train_y)

    postprocess = PostprocessingPipeline(steps=[("exceedance", exceedance_fn)])
    exceedance_probs = postprocess.transform(predictions)
    assert exceedance_probs.max() <= 1
    assert exceedance_probs.min() >= 0
