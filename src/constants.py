# Place all your constants here
import os
from pathlib import Path

# Note: constants should be UPPER_CASE
constants_path = Path(os.path.realpath(__file__))
SRC_PATH = Path(os.path.dirname(constants_path))
PROJECT_PATH = Path(os.path.dirname(SRC_PATH))

DATA_DIR = PROJECT_PATH / "data"
lake_order = ["sup", "mic_hur", "eri", "ont"]
