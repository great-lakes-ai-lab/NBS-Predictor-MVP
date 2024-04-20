Of course! Here's the README with the friendly introduction added back in:

---

# NBS-Predictor-MVP

Welcome to the NBS Predictor MVP project! We're excited to work towards a minimum viable product (MVP) designed to forecast net basin supply (NBS) six months into the future using NOAA forecast data.

- **Target**: Forecast monthly net basin supply (NBS) for all of the Great Lakes, for six months into the future
- **Inputs**: NOAA forecast data (e.g., Climate Forecast System, National Water Model)

Our project will evolve over time, adjusting its targets and inputs as needed. This repository represents our initial effort to tackle this prediciton task.

## Getting Started

### Prerequisites

Before diving in, make sure you have the following installed on your system:

- **Python 3.8** or later
- **pip**
- **virtualenv**

### Clone the Repository

Begin by cloning the repository to your local machine:

```bash
git clone https://github.com/CIGLR-ai-lab/NBS-Predictor-MVP.git
```

### Create a Virtual Environment

Next, create a virtual environment named `env` using `virtualenv`:

```bash
virtualenv env
```

### Activate the Virtual Environment

Activate your virtual environment:

- **Linux/Mac:**

  ```bash
  source ./env/bin/activate
  ```

- **Windows:**

  ```bash
  .\env\Scripts\activate
  ```

### Install Required Packages

Install the necessary packages using `pip`:

```bash
pip install -r requirements.txt
```

### Build Instructions (Optional)

We provide a Makefile to streamline the project's development workflow, which is optional. You can use the Makefile to simplify the setup process and automate routine tasks. If you prefer using the Makefile, follow these steps:

1. **Run Makefile Command**

   Run the `create_environment` target to set up the virtual environment:

   ```bash
   make create_environment
   ```

2. **Install Required Packages**

   Install the required packages using `pip`:

   ```bash
   pip install -r requirements.txt
   ```

3. **Activate the Virtual Environment**

   Activate the virtual environment as described in the "Activate the Virtual Environment" section above.

4. **Run the Project**

   Run the project using the appropriate command for your project:

   ```bash
   python your_script.py
   ```

For more details about the available Makefile commands and how to use them, check out the [Makefile Documentation](docs/MAKEFILE.md).

## Project Organization

```
├── LICENSE
|
├── docs/
│   ├── MAKEFILE.md               # Detailed Makefile documentation
│   └── ...                       # Other documentation files
|
├── Makefile           <- Makefile with commands like `make init` or `make lint-requirements` (Optional)
|
├── README.md          <- The top-level README for developers using this project.
|
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
|   |                     the creator's initials, and a short `-` delimited description, e.g.
|   |                     `1.0_initial-data-exploration`.
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
