import pytest
import pandas as pd
import numpy as np
import os
import sys

# Make the 'src' folder accessible for importing 'features.build_features'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from features.build_features import add_features

# -------- Fixtures --------

@pytest.fixture
def sample_df_for_features():
    dates = pd.date_range(start='2023-01-01', periods=30, freq='D')
    data = {
        'Close': np.linspace(100, 130, 30) + np.random.normal(0, 1, 30)
    }
    df = pd.DataFrame(data, index=dates)
    return df

# -------- Tests --------

def test_add_features_output_shape(sample_df_for_features):
    result = add_features(sample_df_for_features)

    # Expect fewer rows due to NaNs from rolling operations
    assert result.shape[0] < sample_df_for_features.shape[0]

    expected_columns = ['Monthly_Return', 'MA5', 'MA10', 'MA20', 'Volatility_5', 'Volatility_10']
    assert all(col in result.columns for col in expected_columns)

def test_no_nans_in_result(sample_df_for_features):
    result = add_features(sample_df_for_features)
    assert not result.isnull().values.any()

def test_index_is_preserved(sample_df_for_features):
    result = add_features(sample_df_for_features)
    assert isinstance(result.index, pd.DatetimeIndex)
    assert result.index.is_monotonic_increasing

def test_monthly_return_type_and_values(sample_df_for_features):
    result = add_features(sample_df_for_features)
    assert 'Monthly_Return' in result.columns
    assert pd.api.types.is_float_dtype(result['Monthly_Return'])
