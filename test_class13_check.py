# test_class13_check.py
import pytest
from class13_check import BaseMLModel, RandomForestModel, LinearRegressionModel


def test_get_model_type():
    rf = RandomForestModel()
    lr = LinearRegressionModel()
    assert rf.get_model_type() == "RandomForestModel"
    assert lr.get_model_type() == "LinearRegressionModel"


def test_fit_predict_rf():
    rf = RandomForestModel()
    X_train = [[1], [2], [3]]
    y_train = [0, 1, 0]
    X_test = [[4], [5]]
    rf.fit(X_train, y_train)
    pred = rf.predict(X_test)
    assert isinstance(pred, list)


def test_fit_predict_lr():
    lr = LinearRegressionModel()
    X_train = [[1], [2], [3]]
    y_train = [1, 0, 1]
    X_test = [[4], [5]]
    lr.fit(X_train, y_train)
    pred = lr.predict(X_test)
    assert isinstance(pred, list)


def test_cannot_instantiate_base():
    with pytest.raises(TypeError):
        BaseMLModel()


def test_incomplete_model_raise():
    class IncompleteModel(BaseMLModel):
        pass

    with pytest.raises(TypeError):
        IncompleteModel()


def test_partial_abstract_raise():
    class PartialModel(BaseMLModel):
        def fit(self, X, y):
            pass
        # 不实现 predict

    with pytest.raises(TypeError):
        PartialModel()
