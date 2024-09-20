# NBS-Predictor-MVP

Welcome to NBS-Predictor-MVP! We are excited to present a minimum viable product (MVP) designed to forecast net basin supply (NBS) for the Laurentian Great Lakes six months into the future using NOAA forecast data.

- **Target**: Forecast monthly net basin supply (NBS) for all of the Laurentian Great Lakes, for six months into the future.
- **Inputs**: NOAA forecast data from the Climate Forecast System)(CFS).

## Getting Started

### Prerequisites

Before getting started, make sure you have the following installed on your system:
- **Conda** (Anaconda or Miniconda)
- **Python 3.9** or later

### Download the Repository

You have two options to get the repository:

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

### Create a Conda Environment

Create a Conda environment using the `environment.yml` file located in the `requirements` folder:

```bash
conda env create -f requirements/environment.yml
```

Activate the Conda environment:

```bash
conda activate cross_platform_env
```

### Setting Up Jupyter Kernel

After setting up the Conda environment, you need to register a Jupyter kernel for the environment so that you can run the notebooks in Jupyter.

1. **Install `ipykernel`** (if not already installed):

```bash
conda install ipykernel
```

2. **Create a new Jupyter kernel**:

```bash
python -m ipykernel install --user --name cross_platform_env --display-name "Python (cross_platform_env)"
```

This command registers the Conda environment as a kernel in Jupyter, so it can be selected when running notebooks.

3. **Running the Jupyter Notebook**: 

Start the Jupyter Notebook server. If you are using jupyter lab, use this command:  

```
jupyter lab
```

Navigate to the notebook you want to run (e.g. `notebooks/production/2_LEF_forecast_model.ipynb`) and open it. 

### Manually Set Paths

Currently, paths to data directories are treated as user inputs. Users will need to manually set the full path to their local data directory. 

In each notebook, there is a section titled 'User Input' where these paths should be specified.

For example, the variable for the path to the `Data` directory is named `dir`. You need to replace this variable with the full path to your data folder on your machine.

Here is a sample code snippet from the 'User Input' section:

```python
# Example of "User Input" section in a notebook
# Path to download data to
dir = 'C:/Users/username/Desktop/Data/'
```

Make sure to update this variable in each notebook where it is required.

### Managing the Conda Environment

If you need to update the Conda environment with new dependencies, update `requirements/environment.yml`, and then run:

```bash
conda env update -f requirements/environment.yml
```

### Deactivate the Conda Environment

When you have finished working with the project, or if you need to return to your global Python environment, you can deactivate the Conda environment by running the following command:

```bash
conda deactivate
```

### Project Organization 

```
├── LICENSE
│
├── docs/
│   ├── MAKEFILE.md               # Detailed documentation Makefile (NOT CURRENTLY IN USE)
│   └── ...                       # Other documentation files
│
├── Makefile                      # Makefile with commands like `make init` or `make lint-requirements` (Optional)
│
├── README.md                     # The top-level README for developers and users.
│
├── notebooks                     # Jupyter notebooks. Naming convention is a number (for ordering),
│   │                             # the creator's initials, and a short `-` delimited description, e.g.
│   │                             # `1.0_MMM_initial-data-exploration`.
│   │  
│   ├── exploratory               # Notebooks for initial exploration.
│   │  
│   └── production                # Production-ready notebooks (USE THESE FOR A DEMO!)
│
├── requirements                  # Directory containing the requirement files.
│   └── environment.yml           # Conda environment configuration file
│
├── src                           # Source code (NOT CURRENTLY IN USE)
│   ├── README.md                 # Documentation about the src directory
│   ├── composition.py            # File containing composition-related functions/classes
│   ├── constants.py              # File storing project constants
│   ├── data_loading              # Directory for data loading scripts
│   ├── modeling                  # Directory for modeling-related scripts
│   ├── postprocessing            # Directory for postprocessing scripts
│   ├── preprocessing             # Directory for preprocessing scripts
│   ├── utils.py                  # Utility functions
│   └── tests                     # Scripts for unit tests of functions

```

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
