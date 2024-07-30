import numpy as np
import pandas as pd
import pytest

from src.step3_modeling import summarize


@pytest.fixture
def fake_results():

    true = np.random.standard_normal(size=100)
    mean = np.random.uniform(-2, 2, size=100)
    lower, upper = mean - 1, mean + 1
    std = 0.5
    return pd.DataFrame(
        {
            "mean": mean,
            "lower": lower,
            "upper": upper,
            "std": std,
            "true": true,
        }
    )


def test_summarize(fake_results):
    out = summarize(fake_results)
    assert isinstance(out, pd.Series)
