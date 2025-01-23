# NBS-Predictor-MVP

Welcome to NBS-Predictor-MVP! We are excited to present a Minimum Viable Product (MVP) designed to forecast net basin supply (NBS) for the Laurentian Great Lakes six months into the future using NOAA forecast data.

- **Target**: Forecast monthly net basin supply (NBS) for all Laurentian Great Lakes six months into the future.
- **Inputs**: NOAA forecast data from the Climate Forecast System (CFS).

## Getting Started

### Prerequisites

Before getting started, make sure you have the following installed on your system:
- **Conda** (Anaconda or Miniconda)
- **Python 3.9** or later

## First Usage

#### Option 1: Clone the Repository with Git

Begin by cloning the repository to your local machine and navigate into the project directory:

```bash
git clone https://github.com/CIGLR-ai-lab/NBS-Predictor-MVP.git
cd NBS-Predictor-MVP
```

#### Option 2: Download the Repository as a ZIP File

Alternatively, you can download the repository as a ZIP file and extract it:

1. Click on the "Code" button at the top right of the repository page.
2. Click on "Download ZIP".
3. Extract the ZIP file to your desired location.
4. Navigate into the extracted directory:

```bash
cd NBS-Predictor-MVP
```

### Step 2: Create a Conda Environment

Create a Conda environment using the `environment.yml` file located in the `requirements` folder:

```bash
conda env create -f requirements/environment.yml
```

Activate the Conda environment:

```bash
conda activate cross_platform_env
```

### Step 3: Setting Up Jupyter Kernel

After setting up the Conda environment, you need to register a Jupyter kernel for the environment so that you can run the notebooks in Jupyter.

1. **Install `ipykernel`** (if not already installed):

```bash
conda install ipykernel
```

Create a new Jupyter kernel:

```bash
python -m ipykernel install --user --name cross_platform_env --display-name "Python (cross_platform_env)
```

This command registers the Conda environment as a kernel in Jupyter, so it can be selected when running notebooks.

### Step 4: Set Up the Data Directory

The MVP assumes that you have a `Data` directory with a necessary input mask file. From the `NBS-Predictor-MVP` directory, run these commands to set it up:

```bash
mkdir ../Data/Input/
cp notebooks/production/GL_mask.nc ../Data/Input/
```

Once you run the download scripts, you will also see downloaded forecast files and CSV summary files in this directory.

### Step 5: Manually Set Paths

Currently, paths to data directories are treated as user inputs. Users will need to manually set the full path to their local data directory.

In each notebook, there is a section titled 'User Input' where these paths should be specified.

For example, the variable for the path to the Data directory is named dir. You need to replace this variable with the full path to your data folder on your machine.

Here is a sample code snippet from the 'User Input' section:

```bash
# Example of "User Input" section in a notebook
# Path to download data to
dir = 'C:/Users/username/Desktop/Data/'
```

Make sure to update this variable in each notebook where it is required.

## Subsequent Usage

### Step 1: Activate the Conda Environment
Before starting your work, ensure you activate the Conda environment:

```bash
conda activate cross_platform_env
```

### Step 2: Running Jupyter Notebook or Jupyter Lab
Start the Jupyter Lab server or Jupyter Notebook:

```bash
jupyter lab
```

### Step 3: Set Paths and Run Notebooks
Navigate to the notebook you want to run (e.g., `notebooks/production/2_LEF_forecast_model.ipynb`) and open it. Ensure that the path variables in the 'User Input' section are correctly set.

### Step 4: Deactivate the Conda Environment
When you have finished working with the project, or if you need to return to your global Python environment, you can deactivate the Conda environment by running the following command:

```bash
conda deactivate
```

Note: If you need to update the Conda environment with new dependencies, update requirements/environment.yml, and then run:

```bash
conda env update -f requirements/environment.yml
```

## Quick Usage Example

Here's a quick example to get you started:

Start the Jupyter Lab server:

```bash
jupyter lab
```

- Navigate to the `notebooks/production/2_LEF_forecast_model.ipynb` notebook and open it.
- Set the `dir` variable in the 'User Input' section to the path of your Data directory.
- Run through the cells to generate the forecast.


## Project Organization 

#### Main Project Directory

Below is the structure of the main project directory, as reflected in this repository: 

```
NBS-Predictor-MVP/ # Main project directory
├── LICENSE # Licensing information
│
├── docs/ # Documentation (NOT CURRENTLY IN USE)
│ ├── MAKEFILE.md # Detailed documentation Makefile 
│ └── ... # Other documentation files
│
├── README.md # The top-level README for developers and users.
│
├── notebooks # Jupyter notebooks. Naming convention is a number (for ordering),
│ │ # the creator's initials, and a short - delimited description, e.g.
│ │ # 1.0_MMM_initial-data-exploration.
│ │
│ ├── exploratory # Notebooks for initial exploration.
│ │
│ └── production # Production-ready notebooks (DEMO)
│
├── requirements # Directory containing the requirement files.
│ └── environment.yml # Conda environment configuration file
│
├── src # Source code (NOT CURRENTLY IN USE)
│ ├── README.md # Documentation about the src directory
│ ├── composition.py # File containing composition-related functions/classes
│ ├── constants.py # File storing project constants
│ ├── data_loading # Directory for data loading scripts
│ ├── modeling # Directory for modeling-related scripts
│ ├── postprocessing # Directory for postprocessing scripts
│ ├── preprocessing # Directory for preprocessing scripts
│ ├── utils.py # Utility functions
│ └── tests # Scripts for unit tests of functions
```

#### Data Directory

The `Data` directory can sit at the same level as `NBS-Predictor-MVP` and contains the input data needed for the project:

```
Data/ # Parent directory for input data
├── Input/ # Input data directory
│ └── GL_mask.nc # Mask file, needs to be copied from notebooks/production
│
│   # Forecast files below will be created by a script/notebook if they don't already exist
├── CFS_EVAP_forecasts_Sums_CMS.csv # Evaporation forecasts
├── CFS_PCP_forecasts_Sums_CMS.csv  # Precipitation forecasts
├── CFS_TMP_forecasts_Avgs_K.csv    # Air temperature forecasts
```

Note that the `Data` directory can be placed anywhere, as long as the paths are updated appropriately. 

## Contributing

We welcome contributions to NBS-Predictor-MVP! Whether it's improving the codebase, addressing issues, or enhancing documentation, your help is appreciated.

To contribute:

1. **Clone the repository**: Click on the 'Code' button and follow instructions to clone the repository locally.
2. **Create your feature branch**: From your local repository, checkout a new branch for your feature or fix.
3. **Commit your changes**: Make sure your commits are clear and understandable.
4. **Push to the branch**: Push your changes to GitHub.
5. **Create a new Pull Request**: Submit a pull request to our `main` branch with a clear list of what you've done. Please follow the pull request template provided.

Before submitting a pull request, please make sure to review the [contributing guidelines](CONTRIBUTING.md) for detailed information on how to contribute.

Issues can be reported using the GitHub issue tracker. We ask that you please look for any related issues before submitting a new one to avoid duplicates.

For significant changes, please start a thread in the Discussions tab to chat about what you would like to contribute with the community.

New to Git/GitHub? Check out this resource from [Software Carpentry](https://swcarpentry.github.io/git-novice/)

## Code of Conduct

This project and everyone participating in it is governed by our Code of Conduct. By participating, you are expected to uphold this code and behave appropriately and respectfully toward other contributors.

Please read the [code of conduct](CODE_OF_CONDUCT.md) file to learn more about our standards.

## License

This project is licensed under the terms of the GNU Affero General Public License Version 3.0.

## References

A description of this tool, along with some initial results, was presented at the AGU Fall Meeting in 2024: 

Lindsay Fitzpatrick, Dani C Jones, Matt Mcanear, et al. Improving Subseasonal to Annual Water Level Forecasts in the North American Great Lakes Using Machine Learning. Authorea. January 22, 2025. [DOI:10.22541/essoar.173758147.79259133/v1](https://doi.org/10.22541/essoar.173758147.79259133/v1)
