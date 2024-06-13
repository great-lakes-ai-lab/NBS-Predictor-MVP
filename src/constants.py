# Place all your constants here
import os
from pathlib import Path

# Note: constants should be UPPER_CASE
constants_path = Path(os.path.realpath(__file__))
SRC_PATH = Path(os.path.dirname(constants_path))
PROJECT_PATH = Path(os.path.dirname(SRC_PATH))

DATA_DIR = PROJECT_PATH / "data"
runoff_path = DATA_DIR / "CIGLR" / "runoff_glerl_mic_hur_combined.csv"
rnbs_path = DATA_DIR / "CIGLR" / "rnbs_glcc.csv"
pcp_lake_path = DATA_DIR / "CIGLR" / "pcp_glerl_lakes_mic_hur_combined.csv"
evap_lake_path = DATA_DIR / "CIGLR" / "evap_glerl_lakes_mic_hur_combined.csv"
water_level_path = DATA_DIR / "CIGLR" / "wl_glcc.csv"
temp_path = DATA_DIR / "CFSR" / "CFSR_TMP_Basin_Avgs.csv"
