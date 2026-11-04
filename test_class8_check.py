# test_class8_check.py
import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from class8_check import RandomForestModel


@pytest.fixture(scope="module")
def breast_cancer_split():
    data = load_breast_cancer()
    X = data.data
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test

@pytest.fixture
def blank_rf():
    return RandomForestModel(n_estimators=30, max_depth=3, random_state=42)

def test_init_params_stored(blank_rf):
    p = blank_rf.get_params()
    assert p["n_estimators"] == 30
    assert p["max_depth"] == 3
    assert p["random_state"] == 42

def test_predict_before_fit_raise(blank_rf, breast_cancer_split):
    _, X_test, _, _ = breast_cancer_split
    with pytest.raises(Exception):
        blank_rf.predict(X_test)

def test_predict_proba_before_fit_raise(blank_rf, breast_cancer_split):
    _, X_test, _, _ = breast_cancer_split
    with pytest.raises(Exception):
        blank_rf.predict_proba(X_test)

def test_evaluate_before_fit_raise(blank_rf, breast_cancer_split):
    _, X_test, _, y_test = breast_cancer_split
    with pytest.raises(Exception):
        blank_rf.evaluate(X_test, y_test)

def test_feature_importance_before_fit_raise(blank_rf):
    with pytest.raises(Exception):
        blank_rf.get_feature_importance()

def test_fit_predict_work(blank_rf, breast_cancer_split):
    X_train, X_test, y_train, _ = breast_cancer_split
    blank_rf.fit(X_train, y_train)
    y_pred = blank_rf.predict(X_test)
    assert y_pred.shape[0] == X_test.shape[0]
    assert set(np.unique(y_pred)).issubset({0,1})

def test_predict_proba_shape(blank_rf, breast_cancer_split):
    X_train, X_test, y_train, _ = breast_cancer_split
    blank_rf.fit(X_train, y_train)
    proba = blank_rf.predict_proba(X_test)
    assert proba.shape == (X_test.shape[0], 2)

def test_evaluate_metrics_range(blank_rf, breast_cancer_split):
    X_train, X_test, y_train, y_test = breast_cancer_split
    blank_rf.fit(X_train, y_train)
    met = blank_rf.evaluate(X_test, y_test)
    for v in met.values():
        assert 0 <= v <= 1
    assert {"accuracy","precision","recall","f1"} == set(met.keys())

def test_importance_sort(blank_rf, breast_cancer_split):
    X_train, _, y_train, _ = breast_cancer_split
    blank_rf.fit(X_train, y_train)
    df_imp = blank_rf.get_feature_importance()
    assert "importance" in df_imp.columns
    # 验证降序
    vals = df_imp["importance"].values
    assert all(vals[i] >= vals[i+1] for i in range(len(vals)-1))

def test_refit_overwrite(blank_rf, breast_cancer_split):
    X_train, X_test, y_train, y_test = breast_cancer_split
    blank_rf.fit(X_train, y_train)
    m1 = blank_rf.evaluate(X_test, y_test)
    # 再次 fit，覆盖模型
    blank_rf.fit(X_train, y_train)
    m2 = blank_rf.evaluate(X_test, y_test)
    # seed 固定，指标应一致
    np.testing.assert_almost_equal(m1["accuracy"], m2["accuracy"])
