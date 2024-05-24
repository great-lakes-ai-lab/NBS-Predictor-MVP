from pathlib import Path

from sqlalchemy import create_engine

from src.data_loading import parse_file

if __name__ == "__main__":

    import os

    DB_PASSWORD = os.environ["DB_PASSWORD"]
    DB_HOST = os.environ["DB_HOST"]
    DB_DATABASE = os.environ["DB_DATABASE"]
    DB_USER = os.environ["DB_USER"]
    DB_PORT = os.environ["DB_PORT"]

    all_csvs = Path("../CSV/").glob("*.csv")
    dsn = f"postgresql+psycopg://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_DATABASE}"
    conn = create_engine(dsn)
    for csv in all_csvs:
        try:
            parse_file(csv, conn=conn, schema="raw_data", if_exists="fail")
        except ValueError:
            # If the table already exists, just keep going
            continue
