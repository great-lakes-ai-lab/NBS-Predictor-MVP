import os
from sqlalchemy.engine import URL


def get_dsn(host=None, port=None, database=None, user=None, password=None):
    host = host or os.environ["DB_HOST"]
    port = port or os.environ["DB_PORT"]
    database = database or os.environ["DB_DATABASE"]
    user = user or os.environ["DB_USER"]

    try:
        password = password or os.environ["DB_PASSWORD"]
    # uses the .pgpass file if password is not available, which is the preferred behavior anyway
    except KeyError:
        pass

    return URL.create(
        drivername="postgresql+psycopg",
        username=user,
        password=password,
        host=host,
        port=port,
        database=database,
    )
