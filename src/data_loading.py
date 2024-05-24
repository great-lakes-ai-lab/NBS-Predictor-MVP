import datetime as dt
import re

import pandas as pd


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


lake_abbrev = {
    "S": "superior",
    "O": "ontario",
    "E": "erie",
    "M": "michigan",
    "H": "huron",
}
basin_abbrev = {"L": "land", "B": "basin", "W": "water"}


def clean_file_name(filename):
    _, var, lb, _ = filename.split("_")
    if lb == "Basin":
        return f"{var}_forecasts".lower()

    else:
        lake = lake_abbrev[lb[0]]
        basin = basin_abbrev[lb[1]]
        return f"{var}_{lake}_{basin}".lower()


def parse_type_1(filepath):
    """
    Type 1 files have data for all the basins combined in one file, starting with a column for the CFS modelling run
    initialization date in the format YYYYMMDDHH, a column for the CFS run forecast year in the format YYYY, and a
    column for the CFS forecast month in the format MM, and then a columns for each of the 15 Great Lakes Basins.

    Example 1: CFS_"variable_abbreviation"_Basin_Avgs.csv

    variable_abbreviation can be either APCP, EVAP, LHFX, or TMP
    APCP = Accumulated Precipitation (units: kg/m^2)
    EVAP = Evaporation (units: Kg/m^2)
    LHFX = Latent Heat Flux (units: W/M^2)
    TMP = Temperature (units: Kelvin)

    Filenames have the following convention
    YYYY=four digit year (i.e 2024)
    MM=two digit month (i.e 05 for May)
    DD=two digit day (i.e 23 for today's date)
    HH=two digit UTC time (i.e 00, 06, 12, or 18)

    :param filepath: The Path object of the target CSV file
    :return: The parsed data from the CSV and a name for the SQL table.
    """

    # read and fill in the 999 values with NaN
    df = pd.read_csv(filepath, index_col="cfsrun", na_values=999)
    forecast_dates = [dt.date(yr, mo, 1) for yr, mo in zip(df["year"], df["month"])]
    cfs_dates = pd.Series(
        [dt.datetime.strptime(str(x), "%Y%m%d%H") for x in df.index.values],
        index=df.index,
    )
    df = df.drop(["year", "month"], axis=1)
    current_columns = df.columns.to_list()
    col_order = ["cfs_runtime", "forecast_date"] + current_columns

    df = df.assign(cfs_runtime=cfs_dates, forecast_date=forecast_dates)[col_order]
    df.columns = [camel_to_snake(c) for c in df.columns]
    return df, clean_file_name(filepath.name)


def parse_type_2(filepath):
    """
    Type 2 files have the data for a single Great Lakes basin, starting with a column for the CFS modelling run
    initialization date in the format YYYYMMDDHH, and then a column for each of the nine months forecast by that
    particular CFS modelling run. A value of 999 in any of these columns indicates that the CFS run did not include
    that month in its simulation.

    Follows the conventions of Type 1 files

    Type 2: CFS_"variable_abbreviation"_"basin_abbreviation"_Avgs.csv

    basin_abbreviation key:

    Example for Lake Erie:

    EB = Erie full Basin
    EL = Erie Basin (land only)
    EW = Erie Basin (water only)

    :param filepath: The Path object of the target CSV file
    :return: The parsed data from the CSV and a name for the SQL table.
    """
    df = pd.read_csv(filepath, index_col="cfsrun", na_values=999)
    current_columns = list(df.columns.values)
    col_order = ["cfs_runtime"] + current_columns
    cfs_runtime = pd.Series(
        [dt.datetime.strptime(str(x), "%Y%m%d%H") for x in df.index.values],
        index=df.index,
    )
    df = df.assign(cfs_runtime=cfs_runtime)[col_order]
    df.columns = [camel_to_snake(c) for c in df.columns]
    return df, clean_file_name(filepath.name)


def camel_to_snake(name):
    name = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", name).lower()


def parse_file(filepath, conn, schema="ciglr"):
    """
    Wrapper to pass files to the appropriate parsing function based on the name of the file.

    :param filepath: filepath to the CSV
    :param conn: connection to a database for loading
    :param schema: which database schema to load
    :return:
    """
    if "Sizes" in filepath.name:
        df, tbl_name = pd.read_csv(filepath), "basin_sizes"
    elif "Basin" in filepath.name:
        df, tbl_name = parse_type_1(filepath)
    else:
        df, tbl_name = parse_type_2(filepath)

    df.to_sql(con=conn, name=tbl_name, schema=schema, if_exists="replace")
