# test_class9_check.py
import numpy as np
import pytest

from class9_check import RandomForestModel


@pytest.fixture
def sample_data():
    np.random.seed(42)
    X = np.random.rand(100, 5)
    y = (X[:, 0] + X[:, 1] > 1).astype(int)
    split = int(len(X) * 0.8)
    return X[:split], y[:split], X[split:], y[split:]


@pytest.fixture
def temp_paths(tmp_path):
    model_pkl = tmp_path / "rf.pkl"
    cfg_json = tmp_path / "config.json"
    return str(model_pkl), str(cfg_json)


def test_save_before_train_raise_error():
    m = RandomForestModel(n_estimators=10)
    with pytest.raises(ValueError, match="Model not trained"):
        m.save("dummy.pkl")


def test_predict_before_train_raise_error():
    m = RandomForestModel(n_estimators=10)
    X = np.random.rand(10, 5)
    with pytest.raises(ValueError, match="Model not trained"):
        m.predict(X)


def test_load_nonexist_model_raise_file_not_found():
    m = RandomForestModel(n_estimators=10)
    with pytest.raises(FileNotFoundError):
        m.load("not_exist_xxx.pkl")


def test_load_config_nonexist_json_raise(sample_data, temp_paths):
    m = RandomForestModel(n_estimators=10)
    _, cfg_path = temp_paths
    with pytest.raises(FileNotFoundError):
        m.load_config(cfg_path)


def test_config_save_load_only_hyperparams(sample_data, temp_paths):
    X_train, y_train, _, _ = sample_data
    m = RandomForestModel(n_estimators=20, max_depth=4, random_state=123)
    m.fit(X_train, y_train)
    _, cfg_path = temp_paths
    m.save_config(cfg_path)

    m2 = RandomForestModel(n_estimators=5)
    m2.load_config(cfg_path)
    info = m2.get_model_info()
    assert info["params"]["n_estimators"] == 20
    assert info["params"]["max_depth"] == 4
    assert info["is_trained"] is False


def test_model_save_load_prediction_consistent(sample_data, temp_paths):
    X_train, y_train, X_test, _ = sample_data
    m1 = RandomForestModel(n_estimators=50, max_depth=5, random_state=42)
    m1.fit(X_train, y_train)
    model_path, _ = temp_paths
    m1.save(model_path)
    pred_origin = m1.predict(X_test)

    # 新建实例，初始化参数不一样，然后 load
    m2 = RandomForestModel(n_estimators=5)
    m2.load(model_path)
    pred_loaded = m2.predict(X_test)
    np.testing.assert_array_equal(pred_origin, pred_loaded)

    info = m2.get_model_info()
    assert info["model_name"] == "RandomForestModel"
    assert info["is_trained"] is True


def test_get_model_info_structure(sample_data):
    X_train, y_train, _, _ = sample_data
    m = RandomForestModel(n_estimators=30, max_depth=3, random_state=99)
    info_before = m.get_model_info()
    assert info_before["is_trained"] is False

    m.fit(X_train, y_train)
    info_after = m.get_model_info()
    assert info_after["is_trained"] is True
    assert "params" in info_after
    assert info_after["params"]["n_estimators"] == 30


def test_fit_x_y_length_mismatch_raise():
    m = RandomForestModel(n_estimators=10)
    X = np.random.rand(20, 3)
    y = np.array([0, 1])
    with pytest.raises(ValueError):
        m.fit(X, y)
