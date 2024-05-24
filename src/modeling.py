from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def split_data(data, target_column):
    """
    Split data into train and test sets.
    
    Args:
    - data (pd.DataFrame): Input data
    - target_column (str): Name of the target column
    
    Returns:
    - tuple: (X_train, X_test, y_train, y_test)
    """
    X = data.drop(columns=[target_column])
    y = data[target_column]
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_model(X_train, y_train):
    """
    Train a Random Forest Classifier.
    
    Args:
    - X_train (pd.DataFrame): Training features
    - y_train (pd.Series): Training target
    
    Returns:
    - RandomForestClassifier: Trained model
    """
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)
    return clf

def evaluate_model(model, X_test, y_test):
    """
    Evaluate a model using accuracy score.
    
    Args:
    - model: Trained model
    - X_test (pd.DataFrame): Test features
    - y_test (pd.Series): Test target
    
    Returns:
    - float: Accuracy score
    """
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)
