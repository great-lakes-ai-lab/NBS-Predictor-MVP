# tests/test_imports.py
import sys

# List of packages to test
packages = [
    "scipy",
    "sklearn", # scikit-learn is imported as sklearn
    "boto3",
    "botocore",
    "cfgrib",
    "pandas",
    "netCDF4", # netcdf4 is imported as netCDF4
    "numpy",
    "jupyter",
    "jupyterlab",
    "ipykernel",
    "matplotlib",
    "PIL",  # Pillow is imported as PIL
    "zlib",
    "zmq",  # zeromq is imported as zmq (pyzmq)
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
def test_imports(packages):
    failed_imports = []
    for pkg in packages:
        try:
            __import__(pkg)
            print(f"Successfully imported {pkg}")
        except ImportError as e:
            print(f"Failed to import {pkg}: {e}")
            failed_imports.append(pkg)
    return failed_imports

if __name__ == "__main__":
    failed = test_imports(packages)
    if failed:
        sys.exit(f"Failed to import the following packages: {', '.join(failed)}")
    else:
        print("All packages imported successfully!")