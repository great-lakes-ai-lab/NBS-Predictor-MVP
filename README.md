# NBS-Predictor-MVP
A minimum viable product (MVP) for forecasting net basin supply (NBS) six months into the future using NOAA forecast data

- Target: Forecast monthly net basin supply (NBS) for all of the Great Lakes, for six months into the future
- Inputs: NOAA forecast data (e.g. Climate Forecast System, National Water Model)

(The targets and inputs will evolve over time. This repository represents an initial attempt at this modeling task.)

## Requirements
- Python 3.x
- (Eventually produce a requirements.txt file and environment setup files)

## Getting started

### Build Instructions

We use a Makefile to manage the project's development workflow. Check out the [Makefile Documentation](docs/MAKEFILE.md) for details on the available commands and how to use them.

## Project Organization
```
├── LICENSE
|
├── docs/
│   ├── MAKEFILE.md               # Detailed Makefile documentation
│   └── ...                       # Other documentation files
|
├── Makefile           <- Makefile with commands like `make init` or `make lint-requirements` (Coming soon)
|
├── README.md          <- The top-level README for developers using this project.
|
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
|   |                     the creator's initials, and a short `-` delimited description, e.g.
|   |                     `1.0_jqp_initial-data-exploration`.
|   |  
│   ├── exploratory    <- Notebooks for initial exploration.
|   |  
│   └── reports        <- Polished notebooks for presentations or intermediate results.
│
├── report             <- Generated analysis as HTML, PDF, LaTeX, etc.
|   |  
│   ├── figures        <- Generated graphics and figures to be used in reporting
|   |  
│   └── sections       <- LaTeX sections. The report folder can be linked to your overleaf
|                         report with github submodules.
│
├── requirements       <- Directory containing the requirement files.
│
├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
│
├── src                <- Source code for use in this project.
│   │
│   ├── __init__.py    <- Makes src a Python module
│   │
│   ├── data_loading   <- Scripts to download or generate data
│   │
│   ├── preprocessing  <- Scripts to turn raw data into clean data and features for modeling
|   |
│   ├── models         <- Scripts to train models and then use trained models to make
│   │                     predictions
│   │
│   └── tests          <- Scripts for unit tests of your functions
│
└── setup.cfg          <- setup configuration file for linting rules
```

