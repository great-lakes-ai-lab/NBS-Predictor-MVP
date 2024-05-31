from src.step1_data_loading import data_loading
from src.step2_preprocessing import preprocessing
from src.step3_modeling import modeling
from src.step4_postprocessing import postprocessing


def run_pipeline(data_path, target_column):
    """
    Run the entire pipeline.

    Args:
    - data_path (str): Path to the data file
    - target_column (str): Name of the target column

    Returns:
    - float: Accuracy score
    """
    # Step 1: Data Loading
    data = data_loading.load_data(data_path)

    # Step 2: Preprocessing
    data = preprocessing.handle_missing_values(data)
    data = preprocessing.scale_features(data)

    # Step 3: Modeling
    X_train, X_test, y_train, y_test = modeling.split_data(data, target_column)
    model = modeling.train_model(X_train, y_train)

    # Step 4: Postprocessing
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    y_pred = postprocessing.convert_to_labels(y_pred_prob)

    # Step 5: Testing
    accuracy = modeling.evaluate_model(model, X_test, y_test)

    return accuracy
