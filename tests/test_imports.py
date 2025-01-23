import sys
import platform
import os
import pytest

# Print system info
print(f"Python version: {sys.version}")
print(f"Platform: {platform.system()} {platform.release()}")
print(f"Machine: {platform.machine()}")
print(f"Environment: {os.getenv('CONDA_DEFAULT_ENV')}")

# List of packages to test
packages = [
    "scipy",
    "sklearn",
    "boto3",
    "botocore",
    "cfgrib",
    "pandas",
    "netCDF4",
    "numpy",
    "jupyter",
    "jupyterlab",
    "ipykernel",
    "matplotlib",
    "PIL",
    "zlib",
    "zmq",
    "yaml",
    "xarray",
    "joblib",
    "absl",
    "astunparse",
    "flatbuffers",
    "gast",
    "pasta",
    "grpc",
    "h5py",
    "keras",
    "clang",
    "markdown",
    "markdown_it",
    "mdurl",
    "ml_dtypes",
    "opt_einsum",
    "optree",
    "google.protobuf",
    "rich",
    "tensorboard",
    "tensorboard_data_server",
    "tensorflow",
    "termcolor",
    "werkzeug",
    "wrapt"
]

# Function to test importing each package
@pytest.mark.parametrize("pkg", packages)
def test_imports(pkg):
    assert __import__(pkg), f"Could not import {pkg}"
