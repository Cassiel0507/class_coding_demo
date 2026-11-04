# test_param10_check.py
import numpy as np
import pytest

# 确保导入的模块名与你本地保存 MLExperiment 类的文件名完全一致
from param10_check import MLExperiment, RandomForestModel


@pytest.fixture
def sample_data():
    np.random.seed(42)
    X = np.random.rand(200, 8)
    y = (np.random.rand(200) > 0.5).astype(int)
    return X, y


def test_full_flow(sample_data):
    X, y = sample_data
    exp = MLExperiment(experiment_name="Industrial Failure Prediction", random_state=42)
    exp.set_dataset(X, y)
    exp.split_data(test_size=0.2)
    rf = RandomForestModel(n_estimators=100, max_depth=5, random_state=42)
    exp.set_model(rf)
    exp.run()
    exp.evaluate()
    res = exp.get_results()

    assert res["experiment_name"] == "Industrial Failure Prediction"
    for k in ("accuracy", "precision", "recall", "f1_score"):
        assert 0.0 <= res[k] <= 1.0


def test_run_before_set_dataset():
    exp = MLExperiment("test", 42)
    with pytest.raises(ValueError, match="Dataset not set"):
        exp.run()


def test_run_before_split(sample_data):
    X, y = sample_data
    exp = MLExperiment("test", 42)
    exp.set_dataset(X, y)
    with pytest.raises(ValueError, match="Data not splitted"):
        exp.run()


def test_run_before_set_model(sample_data):
    X, y = sample_data
    exp = MLExperiment("test", 42)
    exp.set_dataset(X, y)
    exp.split_data()
    with pytest.raises(ValueError, match="Model not set"):
        exp.run()


def test_evaluate_before_run(sample_data):
    X, y = sample_data
    exp = MLExperiment("test", 42)
    exp.set_dataset(X, y)
    exp.split_data()
    rf = RandomForestModel(10, 3, 42)
    exp.set_model(rf)
    with pytest.raises(ValueError, match="Model not trained"):
        exp.evaluate()


def test_get_info(sample_data):
    X, y = sample_data
    exp = MLExperiment("info-test", 99)
    info = exp.get_experiment_info()
    assert info["has_dataset"] is False
    exp.set_dataset(X, y)
    assert exp.get_experiment_info()["has_dataset"] is True


# =====================================================================
# 以下是为你修复的引发 Ruff 报错的特征重要性测试部分（确保你在源文件中实现了该方法）
# =====================================================================
@pytest.fixture
def blank_rf():
    """提供一个未跑通完整流程的实验对象用于异常测试"""
    return MLExperiment("test_rf", 42)


def test_feature_importance_before_fit_raise(blank_rf):
    # 修复核心：将泛洪的 Exception 替换为特定的 ValueError，并指定异常匹配信息
    with pytest.raises(ValueError, match="Model not trained"):
        blank_rf.get_feature_importance()
