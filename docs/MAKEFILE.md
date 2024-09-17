# Project Development Workflow with Makefile

Our Makefile is designed to streamline the process of setting up and maintaining a consistent development environment for your Python project. It includes a collection of commands that automate daily routine tasks, thus increasing efficiency and reducing the potential for errors.

## Makefile Commands

Here's what each target in the Makefile does and how to use it:

### `clean`

Remove all compiled Python files and cache directories. Run this command to ensure that your tests and runs are using the most recent code changes.

```bash
make clean
```

### `lint`

Run `flake8` to perform linting on Python source files. Use this command to check your code for potential errors and to ensure it aligns with our coding standards.

```bash
make lint
```

### `format`

Format your Python source code automatically using `black` for consistent styling across the entire codebase.

```bash
make format
```

### `create_environment`

Setup a Python virtual environment using `virtualenv`.

```bash
make create_environment
```

Activate the new virtual environment from the project directory using:

```bash
source ./env/bin/activate  # Linux/Mac
```
or
```bash
.\env\Scripts\activate     # Windows
```

### `install_jupyter_tools`

Install and configure Jupyter notebook tools and extensions. This enhances the capabilities and user experience of Jupyter notebooks used in the project.

```bash
make install_jupyter_tools
```

### `help`

Display a list of available Makefile commands along with descriptions. Run this if you need a quick reminder of the Makefile's capabilities.

```bash
make help
```

## Getting Started

To get started with the Makefile, ensure you have `make` installed on your system. Then, from your project's root directory, simply run any of the above commands prefaced by `make`, such as `make clean` or `make lint`. Take advantage of these commands to facilitate and automate tedious tasks, enabling you to focus on the development and innovation aspects of the project.
