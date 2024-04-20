# Import functions/classes from individual steps
from .step1_data_loading.data_loading import load_data
from .step2_preprocessing.preprocessing import handle_missing_values, scale_features
from .step3_modeling.modeling import split_data, train_model, evaluate_model
from .step4_postprocessing.postprocessing import convert_to_labels, save_model
from .step5_testing.testing import run_pipeline

# Define __all__ to specify which modules/functions to import when using "from my_project import *"
__all__ = [
    'load_data',
    'handle_missing_values',
    'scale_features',
    'split_data',
    'train_model',
    'evaluate_model',
    'convert_to_labels',
    'save_model',
    'run_pipeline'
]
