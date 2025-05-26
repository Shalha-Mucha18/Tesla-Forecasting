import pytest
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from data.preprocess import (
    preprocess_stock_data, split_stock_data,
    preprocess_data_for_ml_model, preprocess_for_lstm
)

# ---------- Fixtures ----------

@pytest.fixture
def sample_df():
    dates = pd.date_range(start='2021-01-01', periods=10, freq='D')
    data = {
        'Date': dates,
        'Open': np.random.rand(10) * 100,
        'High': np.random.rand(10) * 100,
        'Low': np.random.rand(10) * 100,
        'Close': np.linspace(100, 110, 10),  
        'Volume': np.random.randint(1000, 5000, 10)
    }
    df = pd.DataFrame(data)
    df['Target'] = df['Close'].shift(-1)
    df.dropna(inplace=True)
    return df

@pytest.fixture
def features_and_target():
    return (['Open', 'High', 'Low', 'Close', 'Volume'], 'Target')

# ---------- Tests ----------

def test_split_stock_data_lengths(sample_df):
    train_df, test_df = split_stock_data(sample_df, train_ratio=0.7, tail_extend=2)
    assert len(train_df) >= int(0.7 * len(sample_df))
    assert len(test_df) == len(sample_df) - int(0.7 * len(sample_df))
    assert train_df.index.is_monotonic_increasing

@pytest.mark.parametrize("tail_extend", [0, 1, 5])
def test_split_stock_data_tail_extend(sample_df, tail_extend):
    train_df, _ = split_stock_data(sample_df, tail_extend=tail_extend)
    assert train_df.shape[0] >= int(0.8 * len(sample_df))
    if tail_extend > 0:
        assert (train_df.tail(tail_extend).index == train_df.iloc[-tail_extend:].index).all()

def test_preprocess_data_for_ml_model_shapes(sample_df, features_and_target):
    features, target = features_and_target
    train_df, test_df = split_stock_data(sample_df)
    X_train, y_train, X_test, y_test, _, _ = preprocess_data_for_ml_model(train_df, test_df, features, target)

    assert X_train.shape[0] == train_df.shape[0]
    assert y_train.shape[0] == train_df.shape[0]
    assert X_test.shape[0] == test_df.shape[0]
    assert y_test.shape[0] == test_df.shape[0]

    assert X_train.shape[1] == len(features)
    assert X_test.shape[1] == len(features)

def test_preprocess_for_lstm_shapes(sample_df, features_and_target):
    features, target = features_and_target
    train_df, test_df = split_stock_data(sample_df)
    X_train, y_train, X_val, y_val, X_test, y_test, _, _ = preprocess_for_lstm(train_df, test_df, features, target, time_steps=2)

    assert X_train.shape[0] == y_train.shape[0]
    assert X_val.shape[0] == y_val.shape[0]
    assert X_test.shape[0] == y_test.shape[0]
    assert X_train.shape[1:] == (2, len(features))  # 2 time steps

def test_invalid_feature_column(sample_df):
    with pytest.raises(KeyError):
        preprocess_data_for_ml_model(sample_df, sample_df, ['Invalid'], 'Target')

def test_invalid_target_column(sample_df, features_and_target):
    features, _ = features_and_target
    with pytest.raises(KeyError):
        preprocess_data_for_ml_model(sample_df, sample_df, features, 'Invalid')


