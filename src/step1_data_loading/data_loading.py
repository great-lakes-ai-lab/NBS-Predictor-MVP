import datetime as dt

import pandas as pd
from sqlalchemy import create_engine, Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from src.step1_data_loading.db_utils import get_dsn


def load_data(file_path):
    """
    Load data from a CSV file and return a DataFrame.

    Args:
    - file_path (str): Path to the CSV file

    Returns:
    - pd.DataFrame: Loaded data
    """
    data = pd.read_csv(file_path)
    return data


class Base(DeclarativeBase):
    pass


class DFMixin(object):
    @classmethod
    def to_df(cls, conn: Engine, **kwargs) -> pd.DataFrame:
        df = pd.read_sql_table(
            table_name=cls.__tablename__,
            con=conn,
            schema=cls.__table_args__["schema"],
            **kwargs
        )
        return df


class RNBS(Base, DFMixin):
    __tablename__ = "rnbs_glcc"
    __table_args__ = {"schema": "ciglr"}

    date: Mapped[dt.date] = mapped_column(primary_key=True)
    sup: Mapped[float]
    mic_hur: Mapped[float]
    eri: Mapped[float]
    ont: Mapped[float]


class Runoff(Base, DFMixin):

    __tablename__ = "runoff_glerl_mic_hur_combined"
    __table_args__ = {"schema": "ciglr"}

    date: Mapped[dt.date] = mapped_column(primary_key=True)
    sup: Mapped[float]
    mic_hur: Mapped[float]
    eri: Mapped[float]
    ont: Mapped[float]


class Evaporation(Base, DFMixin):

    __tablename__ = "evap_glerl_lakes_mic_hur_combined"
    __table_args__ = {"schema": "ciglr"}

    date: Mapped[dt.date] = mapped_column(primary_key=True)
    sup: Mapped[float]
    mic_hur: Mapped[float]
    eri: Mapped[float]
    ont: Mapped[float]


class Precipitation(Base, DFMixin):

    __tablename__ = "pcp_glerl_lakes_mic_hur_combined"
    __table_args__ = {"schema": "ciglr"}

    date: Mapped[dt.date] = mapped_column(primary_key=True)
    sup: Mapped[float]
    mic_hur: Mapped[float]
    eri: Mapped[float]
    ont: Mapped[float]


class WaterLevel(Base, DFMixin):

    __tablename__ = "wl_glcc"
    __table_args__ = {"schema": "ciglr"}

    date: Mapped[dt.date] = mapped_column(primary_key=True)
    sup: Mapped[float]
    mic_hur: Mapped[float]
    eri: Mapped[float]
    ont: Mapped[float]


if __name__ == "__main__":
    dsn = get_dsn()

    engine = create_engine(dsn, echo=True)

    print(Base.metadata.create_all(engine))
