from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import pandas as pd

def handle_missing_values(data):
    """
    Handle missing values in the DataFrame.
    
    Args:
    - data (pd.DataFrame): Input data
    
    Returns:
    - pd.DataFrame: Data with missing values handled
    """
    imputer = SimpleImputer(strategy='mean')
    data_filled = imputer.fit_transform(data)
    return pd.DataFrame(data_filled, columns=data.columns)

def scale_features(data):
    """
    Scale numerical features in the DataFrame.
    
    Args:
    - data (pd.DataFrame): Input data
    
    Returns:
    - pd.DataFrame: Data with scaled features
    """
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data.select_dtypes(include=['float64', 'int64']))
    return pd.DataFrame(data_scaled, columns=data.select_dtypes(include=['float64', 'int64']).columns)
