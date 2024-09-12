# tests/test_imports.py
import sys

# List of packages to test
packages = [
    "scipy",
    "scikit-learn",
    "boto3",
    "botocore",
    "cfgrib",
    "pandas",
    "netcdf4",
    "numpy",
    "jupyter",
    "jupyterlab",
    "ipykernel",
    "matplotlib",
    "PIL",  # Pillow is imported as PIL
    "zlib",
    "zeromq",
    "yaml",
    "xarray",
    "joblib",
    "absl",
    "astunparse",
    "flatbuffers",
    "gast",
    "google_pasta",
    "grpc",
    "h5py",
    "keras",
    "clang",
    "markdown",
    "markdown_it_py",
    "mdurl",
    "mld_hypes",
    "opt_einsum",
    "optree",
    "protobuf",
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