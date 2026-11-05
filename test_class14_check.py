import pytest
from class14_check import BaseModel, LoggerMixin, MetricsMixin, IndustrialModel

def test_basic_workflow():
    model = IndustrialModel("RandomForest")
    assert model.get_model_name() == "RandomForest"
    assert model.is_trained is False

    model.fit()
    assert model.is_trained is True

    model.log("Model training started")
    model.log("Model training completed")
    assert model.get_logs() == [
        "Model training started",
        "Model training completed"
    ]

    model.add_metric("accuracy", 0.95)
    model.add_metric("recall", 0.92)
    assert model.get_metrics() == {"accuracy": 0.95, "recall": 0.92}


def test_instance_state_isolation():
    model = IndustrialModel("RandomForest")
    model.log("Model training started")
    model.add_metric("acc", 0.9)

    model2 = IndustrialModel("LogisticRegression")
    model2.log("New experiment")
    model2.add_metric("acc", 0.8)

    # 日志互不干扰
    assert model.get_logs() == ["Model training started"]
    assert model2.get_logs() == ["New experiment"]
    # 指标互不干扰
    assert model.get_metrics() == {"acc":0.9}
    assert model2.get_metrics() == {"acc":0.8}


def test_mro_can_inspect():
    mro = IndustrialModel.__mro__
    # 顺序必须：IndustrialModel → BaseModel → LoggerMixin → MetricsMixin → object
    assert mro[0] == IndustrialModel
    assert mro[1] == BaseModel
    assert mro[2] == LoggerMixin
    assert mro[3] == MetricsMixin


def test_empty_log_and_metric_at_begin():
    m = IndustrialModel("TestModel")
    assert m.get_logs() == []
    assert m.get_metrics() == {}
    assert m.is_trained is False


def test_fit_only_toggle_train_flag():
    m = IndustrialModel("ABC")
    m.fit()
    assert m.is_trained is True
