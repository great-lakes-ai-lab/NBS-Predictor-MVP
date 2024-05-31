from src.step1_data_loading.db_utils import get_dsn
from sqlalchemy import create_engine, text


def test_get_dsn(environ):
    dsn = get_dsn()
    print(dsn)
    assert dsn


def test_local_select(environ):

    engine = create_engine(get_dsn())
    with engine.connect() as conn:
        res = conn.execute(text("select 10")).fetchone()
    assert res[0] == 10
