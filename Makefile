# Makefile for Python Project Development Workflow

# Define phony targets to ensure commands are run even if files with these names exist.
.PHONY: clean lint format create_environment install_jupyter_tools help

# Global variables used across the Makefile.
PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PROJECT_NAME = my_python_project
PYTHON_INTERPRETER = python3

# COMMANDS
#################################################################################

## clean: Remove all compiled Python files and cache directories.
clean:
    find . -type f -name "*.py[co]" -delete
    find . -type d -name "__pycache__" -delete

## lint: Lint Python source code using flake8 for coding standards and quality checks.
lint:
    flake8 src

## format: Format Python source code automatically using black for consistent styling.
format: 
    black src

## create_environment: Set up the Python virtual environment using virtualenv.
create_environment:
    @echo "Setting up the virtual environment."
    # Install virtualenv if not already present.
    $(PYTHON_INTERPRETER) -m pip install --user virtualenv
    # Create a virtual environment for the project.
    virtualenv ./env
    @echo "Virtual environment created. Activate it with:\nsource ./env/bin/activate"

## install_jupyter_tools: Install Jupyter notebook tools and extensions to enhance notebook capabilities.
install_jupyter_tools:
    sh ./.setup_scripts/jupyter_tools.sh

## help: Display callable targets and their descriptions.
help:
    @echo "Makefile for managing Python project development workflow."
    @echo "The following commands are available:"
    @echo ""
    @grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "    \033[36m%-20s\033[0m %s\n", $$1, $$2}'