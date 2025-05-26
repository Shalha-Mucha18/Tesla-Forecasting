import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from keras.models import load_model

def evaluate_model(model_or_path, X_test_scaled, y_test_scaled):
    # Load from file if path is provided
    if isinstance(model_or_path, str):
        model = joblib.load(model_or_path)
    else:
        model = model_or_path  # already a model object

    y_pred = model.predict(X_test_scaled)

    mae = mean_absolute_error(y_test_scaled, y_pred)
    mse = mean_squared_error(y_test_scaled, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_scaled, y_pred)

    return {
        'model': model,
        'y_pred': y_pred,
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r2': r2
    }


def evaluate_keras_model(model_or_path, X_test_scaled, y_test_scaled):
    # Load from file if path is provided
    if isinstance(model_or_path, str):
        model = load_model(model_or_path)
    else:
        model = model_or_path

    y_pred = model.predict(X_test_scaled).flatten()  # ensure shape matches

    mae = mean_absolute_error(y_test_scaled, y_pred)
    mse = mean_squared_error(y_test_scaled, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_scaled, y_pred)

    return {
        'model': model,
        'y_pred': y_pred,
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r2': r2
    }
