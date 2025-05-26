import numpy as np
import pytest
import tempfile
import shutil
from sklearn.datasets import make_regression
from tensorflow.keras.models import load_model
import os
import sys
from model.train_model import train_models_only, build_sequence_model, train_sequence_model


@pytest.fixture(scope="module")
def dummy_data(): 
    X, y = make_regression(n_samples=100, n_features=10, noise=0.1)
    return X, y


@pytest.fixture(scope="module")
def dummy_sequence_data():
    X = np.random.rand(100, 10, 5)  # 100 samples, 10 time steps, 5 features
    y = np.random.rand(100)
    return X, y


@pytest.fixture
def temp_dir():
    path = tempfile.mkdtemp()
    yield path
    shutil.rmtree(path)


def test_train_models_only(dummy_data, temp_dir):
    X, y = dummy_data
    lr, dt, rf = train_models_only(X, y, temp_dir)

    # Check that models are saved
    assert os.path.exists(os.path.join(temp_dir, "model_lr.pkl"))
    assert os.path.exists(os.path.join(temp_dir, "model_dt.pkl"))
    assert os.path.exists(os.path.join(temp_dir, "model_rf.pkl"))

    # Check models are returned
    assert lr is not None and dt is not None and rf is not None


@pytest.mark.parametrize("model_type", ["lstm", "tcn"])
def test_build_sequence_model(model_type):
    input_shape = (10, 5)
    model = build_sequence_model(model_type, input_shape)
    assert model is not None
    assert isinstance(model.input_shape, tuple)
    assert model.output_shape[-1] == 1


def test_build_sequence_model_invalid_type():
    with pytest.raises(ValueError, match="Invalid model_type"):
        build_sequence_model("invalid_type", (10, 5))


def test_train_sequence_model(dummy_sequence_data, temp_dir):
    X, y = dummy_sequence_data
    model = build_sequence_model("lstm", input_shape=(10, 5))
    
    history = train_sequence_model(
        model=model,
        model_name="lstm",
        X_train=X[:80], y_train=y[:80],
        X_val=X[80:], y_val=y[80:],
        save_path=temp_dir,
        epochs=1,
        batch_size=8
    )

    assert history is not None
    saved_path = os.path.join(temp_dir, "model_lstm.keras")
    assert os.path.exists(saved_path)

    loaded_model = load_model(saved_path)
    assert loaded_model is not None
