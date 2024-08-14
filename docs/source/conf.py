import os
import sys
import pdb
from pathlib import Path

# Insert source directory into sys.path
sys.path.insert(0, os.path.abspath('../../src'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'NBS-Predictor-MVP'
copyright = '2024, U-M CIGLR and NOAA GLERL'
author = 'U-M CIGLR and NOAA GLERL'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',       # Automatic documentation
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',      # For Google style or NumPy style docstrings
    'sphinx.ext.autosummary',
]

autosummary_generate = True  # Turn on sphinx.ext.autosummary

templates_path = ['_templates']
exclude_patterns = []

# The master toctree document.
master_doc = 'index'

# The suffix of source filenames.
source_suffix = {'.rst': 'restructuredtext'}

# The encoding of source files.
source_encoding = 'utf-8-sig'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
#html_static_path = ['_static']