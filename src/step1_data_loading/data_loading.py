import pandas as pd

def load_data(file_path):
    """
    Load data from a CSV file and return a DataFrame.
    
    Args:
    - file_path (str): Path to the CSV file
    
    Returns:
    - pd.DataFrame: Loaded data
    """
    data = pd.read_csv(file_path)
    return data
