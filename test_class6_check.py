# test_class6_check.py
import pandas as pd
import pytest

from class6_check import IndustrialDataset


@pytest.fixture
def normal_df():
    return pd.DataFrame({
        "temperature": [20, 30, 40, 50, 60],
        "pressure": [100, 200, 150, 300, 250],
        "vibration": [0.1, 0.2, 0.3, 0.4, 0.5],
        "failure": [0, 0, 1, 0, 1]
    })

@pytest.fixture
def df_with_nan():
    return pd.DataFrame({
        "temperature": [20, None, 40, 50, 60],
        "pressure": [100, 200, 150, None, 250],
        "vibration": [0.1, 0.2, 0.3, 0.4, 0.5],
        "failure": [0, 0, 1, 0, 1]
    })

def test_basic_get_shape(normal_df):
    ds = IndustrialDataset(normal_df, target_column="failure")
    assert ds.get_shape() == (5, 4)

def test_get_features_columns(normal_df):
    ds = IndustrialDataset(normal_df, target_column="failure")
    feat_cols = ds.get_features().columns.tolist()
    assert feat_cols == ['temperature', 'pressure', 'vibration']

def test_get_target_values(normal_df):
    ds = IndustrialDataset(normal_df, target_column="failure")
    assert ds.get_target().tolist() == [0, 0, 1, 0, 1]

def test_class_distribution(normal_df):
    ds = IndustrialDataset(normal_df, target_column="failure")
    dist = ds.get_class_distribution().to_dict()
    assert dist == {0:3, 1:2}

def test_split_features_target(normal_df):
    ds = IndustrialDataset(normal_df, target_column="failure")
    X, y = ds.split_features_target()
    assert X.shape == (5,3)
    assert y.shape == (5,)
    assert X.columns.tolist() == ['temperature', 'pressure', 'vibration']

def test_missing_summary(df_with_nan):
    ds = IndustrialDataset(df_with_nan, target_column="failure")
    miss = ds.get_missing_summary()
    assert miss["temperature"] == 1
    assert miss["pressure"] == 1
    assert miss["vibration"] == 0
    assert miss["failure"] == 0

def test_target_column_not_exist_raise(normal_df):
    with pytest.raises(Exception):  # noqa: B017
        IndustrialDataset(normal_df, target_column="wrong_col")

def test_original_data_not_modified(normal_df):
    original = normal_df.copy()
    ds = IndustrialDataset(normal_df, target_column="failure")
    X, _y = ds.split_features_target()
    X.iloc[0,0] = 9999
    assert normal_df.equals(original)
