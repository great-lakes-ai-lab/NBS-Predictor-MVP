# Import functions/classes from individual steps
from .data_loading import load_data
from .preprocessing import handle_missing_values, scale_features
from .modeling import split_data, train_model, evaluate_model
from .postprocessing import convert_to_labels, save_model

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
]
