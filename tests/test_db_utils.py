import pytest

from src.step1_data_loading.db_utils import get_dsn
from src.step1_data_loading.data_loading import Runoff
from sqlalchemy import create_engine, text, select

import pandas as pd


@pytest.fixture
def engine(environ):
    engine = create_engine(get_dsn())
    return engine


def test_get_dsn(environ):
    dsn = get_dsn()
    print(dsn)
    assert dsn


def test_local_select(engine):

    with engine.connect() as conn:
        res = conn.execute(text("select 10")).fetchone()
    assert res[0] == 10


def test_simple_select(engine):
    # Example usage of a SQLAlchemy select
    with engine.connect() as conn:
        stmt = select(Runoff).limit(10)
        rows = pd.read_sql(stmt, con=conn, index_col="date")

    assert isinstance(rows, pd.DataFrame)
    assert rows.shape == (10, 4)
