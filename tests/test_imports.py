import sys
import platform
import os

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
def test_imports(packages):
    failed_imports = []
    for pkg in packages:
        print(f"Testing import for package: {pkg}")
        try:
            __import__(pkg)
            print(f"Successfully imported {pkg}")
        except ImportError as e:
            print(f"Failed to import {pkg}: {e}")
            failed_imports.append(pkg)
        except Exception as e:
            print(f"An error occurred while importing {pkg}: {e}")
            failed_imports.append(pkg)
    return failed_imports

if __name__ == "__main__":
    failed = test_imports(packages)
    if failed:
        sys.exit(f"Failed to import the following packages: {', '.join(failed)}")
    else:
        print("All packages imported successfully!")