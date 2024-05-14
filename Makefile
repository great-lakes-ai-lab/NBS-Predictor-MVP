# Makefile for NBS-Predictor-MVP

# Define phony targets to ensure commands are run even if files with these names exist.
.PHONY: clean lint format create_environment install_jupyter_tools help

# Global variables used across the Makefile.
PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PROJECT_NAME = NBS-Predictor-MVP
PYTHON_INTERPRETER = python3

# COMMANDS
#################################################################################

## clean: Remove all compiled Python files and cache directories.
clean:
	@echo "Cleaning up *.pyc files and __pycache__ directory."
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## lint: Lint Python source code using flake8 for coding standards and quality checks.
lint:
	@echo "Linting Python source code."
	flake8 src

## format: Format Python source code automatically using black for consistent styling.
format:
	@echo "Formatting Python source code."
	black src

## create_environment: Set up the Python virtual environment using venv.
create_environment:
	@echo "Setting up the Python virtual environment using venv."
	test -d env || $(PYTHON_INTERPRETER) -m venv env
	@echo "Activate the virtual environment with 'source env/bin/activate' on Unix-like systems, or 'env\\Scripts\\activate' on Windows."

## install_jupyter_tools: Install Jupyter notebook tools and extensions to enhance notebook capabilities.
install_jupyter_tools:
	@echo "Installing Jupyter tools and extensions."
	sh ./.setup_scripts/jupyter_tools.sh

## help: Display callable targets and their descriptions.
help:
	@echo "Available commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
