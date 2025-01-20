# README: RNBS Forecasts CSV File

## Data Overview

The `RNBS_forecasts_2012-2022.csv` file contains the Residual Net Basin Supply (RNBS) forecasts for each of the Great Lakes for the period 2012–2022. The forecasts are provided in units of cubic meters per second [cms] and were generated using archived Climate Forecast System (CFS) data, processed through a Gaussian process model trained using the MVP methodology.

## Column Descriptions

| Column  | Description |
|---------|-------------|
| `cfsrun`| The Climate Forecast System (CFS) forecast runtime, denoted by a timestamp in the format YYYYMMDDHH. Here, YYYY represents the year, MM the month, DD the day, and HH the hour of forecast initialization. The forecast is re-initialized every 6 hours. |
| `month` & `year` | Each forecast run provides projections extending up to 9 months into the future. The columns represent each month and year for which the forecast is valid. The MM corresponds to the month, and YYYY corresponds to the year. |
| `sup` | The RNBS forecast value for Lake Superior, expressed in cubic meters per second [cms], for the specific month and year indicated in the corresponding columns. |
| `eri` | The RNBS forecast value for Lake Erie, expressed in cubic meters per second [cms], for the specific month and year indicated in the corresponding columns. |
| `ont` | The RNBS forecast value for Lake Ontario, expressed in cubic meters per second [cms], for the specific month and year indicated in the corresponding columns. |
| `mih` | The RNBS forecast value for Lake Michigan and Lake Huron as a single entity, expressed in cubic meters per second [cms], for the specific month and year indicated in the corresponding columns. Lake Michigan and Lake Huron are often modeled together due to their physical connection via the Straits of Mackinac, which allows water to flow freely between the two lakes. This interconnection results in highly interdependent hydrological behaviors, and modeling them as a combined system provides a more accurate representation of the water supply dynamics for both lakes. |

