import pytest
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from model.evaluate_model import evaluate_model, evaluate_keras_model


def test_evaluate_model_in_memory():
    # Dummy regression data
    X = np.random.rand(20, 3)
    y = X @ np.array([1.5, -2.0, 1.0]) + 0.5

    # Scaling
    scaler_X = StandardScaler().fit(X)
    scaler_y = StandardScaler().fit(y.reshape(-1, 1))

    X_scaled = scaler_X.transform(X)
    y_scaled = scaler_y.transform(y.reshape(-1, 1)).ravel()

    # Train in-memory model
    model = LinearRegression()
    model.fit(X_scaled, y_scaled)

    # Directly evaluate in memory
    results = evaluate_model(model, X_scaled, y_scaled)

    assert 'mae' in results and isinstance(results['mae'], float)
    assert 'mse' in results and isinstance(results['mse'], float)
    assert 'rmse' in results and isinstance(results['rmse'], float)
    assert 'r2' in results and isinstance(results['r2'], float)
    assert results['y_pred'].shape == y_scaled.shape


def test_evaluate_keras_model_in_memory():
    # Dummy regression data
    X = np.random.rand(20, 4)
    y = X[:, 0] * 2 + X[:, 1] * -3 + 1

    scaler_X = StandardScaler().fit(X)
    scaler_y = StandardScaler().fit(y.reshape(-1, 1))

    X_scaled = scaler_X.transform(X)
    y_scaled = scaler_y.transform(y.reshape(-1, 1))

    # Simple Keras model
    model = Sequential([
        Dense(5, input_shape=(4,), activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_scaled, y_scaled, epochs=2, verbose=0)

    # Directly evaluate in memory
    results = evaluate_keras_model(model, X_scaled, y_scaled)

    assert 'mae' in results and isinstance(results['mae'], float)
    assert 'mse' in results and isinstance(results['mse'], float)
    assert 'rmse' in results and isinstance(results['rmse'], float)
    assert 'r2' in results and isinstance(results['r2'], float)
    assert results['y_pred'].shape[0] == y_scaled.shape[0]
