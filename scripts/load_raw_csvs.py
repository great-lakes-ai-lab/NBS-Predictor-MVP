from pathlib import Path

from sqlalchemy import create_engine

from src.data_loading import parse_file

if __name__ == "__main__":

    import os

    DB_PASSWORD = os.environ["DB_PASSWORD"]

    all_csvs = Path("../CSV/").glob("*.csv")
    dsn = f"postgresql+psycopg://mcanearm:{DB_PASSWORD}@localhost:5432/noaa"
    conn = create_engine(dsn)
    for csv in all_csvs:
        parse_file(csv, conn=conn, schema="raw_data")
