from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine

from src.step1_data_loading import RNBS, Runoff, Evaporation, Precipitation, WaterLevel
from src.step1_data_loading.db_utils import get_dsn

table_series_lookup = {
    "pcp": Precipitation,
    "evap": Evaporation,
    "runoff": Runoff,
    "rnbs": RNBS,
    "wl": WaterLevel,
}

if __name__ == "__main__":
    dsn = get_dsn()

    file_paths = Path(
        "../load"
    )  # default ignored directory for loading - put CSVs in there

    csv_files = file_paths.glob("*.csv")

    engine = create_engine(dsn)
    with engine.connect() as conn:
        for csv_file in csv_files:
            input_data = pd.read_csv(
                csv_file, index_col="Date", date_format="%Y%m%d"
            ).rename_axis(index="date")[["sup", "mic_hur", "eri", "ont"]]

            load_class = table_series_lookup[csv_file.name.split("_")[0]]
            input_data.to_sql(
                load_class.__tablename__,
                con=conn,
                schema=load_class.__table__.schema,
                if_exists="replace",
                index=True,
            )
