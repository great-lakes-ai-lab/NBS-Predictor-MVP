import numpy as np
import xarray

__all__ = ["convert_to_labels", "save_model", "get_exceedance_prob"]


def convert_to_labels(probabilities, threshold=0.5):
    """
    Convert predicted probabilities to class labels based on a threshold.

    Args:
    - probabilities (np.ndarray): Predicted probabilities
    - threshold (float): Threshold for classifying as positive

    Returns:
    - np.ndarray: Predicted class labels
    """
    return np.where(probabilities > threshold, 1, 0)


def save_model(model, file_path):
    """
    Save a trained model to a file.

    Args:
    - model: Trained model
    - file_path (str): Path to save the model
    """
    import joblib

    joblib.dump(model, file_path)


def get_exceedance_prob(prob, hist_rnbs: xarray.DataArray):
    pass
