import pytest

from src.step1_data_loading.db_utils import get_dsn
from src.step1_data_loading.data_loading import Runoff, superior
from sqlalchemy import create_engine, text, select

import pandas as pd


@pytest.fixture
def conn(environ):
    engine = create_engine(get_dsn())

    with engine.connect() as conn:
        yield conn


def test_get_dsn(environ):
    dsn = get_dsn()
    print(dsn)
    assert dsn


def test_local_select(conn):
    res = conn.execute(text("select 10")).fetchone()
    assert res[0] == 10


def test_simple_select(conn):
    # Example usage of a SQLAlchemy select
    stmt = select(Runoff).limit(10)
    rows = pd.read_sql(stmt, con=conn, index_col="date")

    assert isinstance(rows, pd.DataFrame)
    assert rows.shape == (10, 4)
