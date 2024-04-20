# Requirements

## Overview
This directory contains the environment setup to reproduce all code in this repository. By explicitly listing all requirements, we ensure a consistent environment across different setups and make it easier to track the packages needed for each stage of development and deployment.

### Structure
The repository includes several requirements files to cater to different stages and aspects of development:

- **`requirements.txt`**: This file contains the pip requirements for both development and deployment using `pip` and `virtualenv`.

- **`dev-requirements.txt`**: This file contains additional development-specific packages for linting, formatting, and other utilities.

- **`test-requirements.txt`**: This file lists the pip requirements needed for continuous integration, including linting and unit testing packages.

### Recommended Workflow
To maintain a reproducible and consistent environment with `pip` and `virtualenv`, follow this workflow:

1. **Install Required Packages**: Use `pip install` to install the packages needed for your analysis or development.
  
2. **Freeze Requirements**: After installing the required packages, run `pip freeze > requirements.txt` to capture the exact package versions used.
  
3. **Version Control**: If you add or update packages, remember to update the `requirements.txt` file by running `pip freeze > requirements.txt` again. Commit these changes to your version control system to keep track of the environment changes.

### Conda Support (Future)
While the project currently supports `pip` and `virtualenv`, we plan to add Conda support with an `environment.yml` file in the future to provide an alternative environment setup for those who prefer Conda.

By adhering to this workflow and maintaining separate requirements files for development, testing, and deployment, we ensure a streamlined and consistent development process while facilitating easy reproducibility and collaboration.
