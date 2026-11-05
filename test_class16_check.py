# test_class16_check.py
import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from class16_check import (
    StandardScalerProcessor,
    MinMaxProcessor,
    MissingValueProcessor,
    ProcessorFactory,
    DataPipeline,
    MLPipeline
)

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "vibration": [1.0, 2.0, 3.0, np.nan],
        "temp": [25, 30, 35, 40]
    })

@pytest.fixture
def train_test_data():
    X_train = pd.DataFrame({"a": [1,2,3,4,5], "b":[10,20,30,40,50]})
    y_train = np.array([0,0,1,1,1])
    X_test = pd.DataFrame({"a": [1.5,4.5], "b":[15,45]})
    return X_train, y_train, X_test

def test_factory_correct_instance():
    p1 = ProcessorFactory.create_processor("standard")
    assert isinstance(p1, StandardScalerProcessor)
    p2 = ProcessorFactory.create_processor("minmax")
    assert isinstance(p2, MinMaxProcessor)
    p3 = ProcessorFactory.create_processor("missing")
    assert isinstance(p3, MissingValueProcessor)

def test_factory_invalid_type_raise():
    with pytest.raises(ValueError):
        ProcessorFactory.create_processor("abc")

def test_processor_interface(sample_df):
    for typ in ["standard", "minmax", "missing"]:
        proc = ProcessorFactory.create_processor(typ)
        ft = proc.fit_transform(sample_df)
        assert isinstance(ft, pd.DataFrame)
        t = proc.transform(sample_df)
        assert isinstance(t, pd.DataFrame)

def test_strategy_switch(sample_df):
    p_std = ProcessorFactory.create_processor("standard")
    p_mm = ProcessorFactory.create_processor("minmax")
    pipe = DataPipeline(p_std)
    res1 = pipe.fit_transform(sample_df)
    pipe.set_processor(p_mm)
    res2 = pipe.fit_transform(sample_df)
    assert not np.allclose(res1.values, res2.values)

def test_mlpipeline_fit_predict(train_test_data):
    X_train, y_train, X_test = train_test_data
    proc = ProcessorFactory.create_processor("standard")
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    ml_pipe = MLPipeline(processor=proc, model=model)
    ml_pipe.fit(X_train, y_train)
    pred = ml_pipe.predict(X_test)
    assert len(pred) == len(X_test)
    assert set(pred).issubset({0,1})

def test_missing_fill_logic():
    df = pd.DataFrame({"x":[np.nan, 5]})
    p = MissingValueProcessor(fill_value=99)
    out = p.fit_transform(df)
    assert out["x"].iloc[0] == 99
