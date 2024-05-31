import os

import pytest


@pytest.fixture
def environ():
    old_environ = os.environ.copy()
    os.environ["DB_DATABASE"] = "noaa"
    os.environ["DB_HOST"] = "localhost"
    os.environ["DB_PORT"] = "5432"
    os.environ["DB_USER"] = "mcanearm"
    yield os.environ
    os.environ = old_environ
