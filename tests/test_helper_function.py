import numpy as np
import pandas as pd
import joblib
import tempfile
import pytest
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import os


from utils.helper_function import (
    create_evaluation_table,
    predict_next_day_from_input,
    predict_next_close_lstm
)


def test_create_evaluation_table():
    dummy_results = {
        "ModelA": {"mae": 1.2, "mse": 2.3, "rmse": 1.52, "r2": 0.85},
        "ModelB": {"mae": 1.0, "mse": 2.0, "rmse": 1.41, "r2": 0.90}
    }

    df = create_evaluation_table(dummy_results)

    assert isinstance(df, pd.DataFrame)
    assert df.shape == (2, 4)
    assert "ModelA" in df.index
    assert "MAE" in df.columns or "mae" in df.columns or "Rmse" in df.columns


def test_predict_next_day_from_input(tmp_path):
    # Create dummy training data
    X_train = np.random.rand(100, 5)
    y_train = np.random.rand(100)

    # Fit scaler and model
    scaler = StandardScaler()
    scaler.fit(X_train)

    model = LinearRegression()
    model.fit(X_train, y_train)

    # Save model
    model_path = tmp_path / "lin_model.pkl"
    joblib.dump(model, model_path)

    input_data = {
        "Open": 1.0,
        "High": 1.0,
        "Low": 1.0,
        "Close": 1.0,
        "Volume": 1.0
    }

    results = predict_next_day_from_input(
        input_data,
        scaler,
        {"LinearModel": str(model_path)}
    )

    assert "original predictions" in results
    assert isinstance(results["original predictions"], dict)
    assert "LinearModel" in results["original predictions"]


import tempfile
import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import StandardScaler
import numbers

def test_predict_next_close_lstm():
    # Create dummy time series data
    df = pd.DataFrame(np.random.rand(10, 5), columns=["Open", "High", "Low", "Close", "Volume"])

    # Fit scalers
    feature_scaler = StandardScaler().fit(df)
    target_scaler = StandardScaler().fit(df[["Close"]])

    # Create dummy LSTM model
    model = Sequential([
        LSTM(10, input_shape=(5, 5)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')

    # Dummy train to initialize weights
    X_dummy = np.random.rand(1, 5, 5)
    y_dummy = np.random.rand(1, 1)
    model.fit(X_dummy, y_dummy, epochs=1, verbose=0)

    # Create temp file with .keras suffix
    fd, model_path = tempfile.mkstemp(suffix=".keras")
    os.close(fd)
    try:
        model.save(model_path)  # Save in the modern Keras format

        actual, predicted = predict_next_close_lstm(
            test_df=df,
            feature_scaler_lstm=feature_scaler,
            target_scaler_lstm=target_scaler,
            model_path=model_path,
            i=0
        )

        assert isinstance(actual, float)
        assert isinstance(predicted, numbers.Real)  # Accepts np.float32 and float

    finally:
        os.remove(model_path)

