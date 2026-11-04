# test_class7_check.py
import pytest

from class7_check import ModelEvaluator


@pytest.fixture
def binary_data():
    y_true = [0, 0, 0, 1, 1, 1]
    y_pred = [0, 0, 1, 1, 1, 1]
    y_pred_other = [0, 1, 0, 1, 0, 1]
    return y_true, y_pred, y_pred_other

@pytest.fixture
def multi_class_data():
    y_true = [0, 1, 2, 0, 1, 2]
    y_pred = [0, 1, 2, 0, 1, 1]
    return y_true, y_pred

def test_accuracy_binary(binary_data):
    y_true, y_pred, _ = binary_data
    eva = ModelEvaluator(y_true, y_pred)
    assert abs(eva.accuracy() - 5/6) < 1e-6

def test_precision_recall_f1_binary(binary_data):
    y_true, y_pred, _ = binary_data
    eva = ModelEvaluator(y_true, y_pred)
    prec = eva.precision(average="binary")
    rec = eva.recall(average="binary")
    eva.f1_score(average="binary")
    assert abs(prec - 0.75) < 1e-6
    assert abs(rec - 1.0) < 1e-6

def test_get_all_metrics(binary_data):
    y_true, y_pred, _ = binary_data
    eva = ModelEvaluator(y_true, y_pred)
    res = eva.get_all_metrics(average="binary")
    assert isinstance(res, dict)
    assert {"accuracy", "precision", "recall", "f1"}.issubset(res.keys())

def test_compare_method(binary_data):
    y_true, y_pred, y_pred_other = binary_data
    eva = ModelEvaluator(y_true, y_pred)
    cmp_result = eva.compare(y_pred_other, average="binary")
    assert {"self", "other"}.issubset(cmp_result.keys())
    assert {"accuracy", "precision", "recall", "f1"}.issubset(cmp_result["self"].keys())

def test_length_mismatch_init_raise():
    with pytest.raises(Exception):
        ModelEvaluator([0,1,0], [0,1])

def test_compare_length_mismatch_raise(binary_data):
    y_true, y_pred, _ = binary_data
    eva = ModelEvaluator(y_true, y_pred)
    with pytest.raises(Exception):
        eva.compare([0,1])

def test_multi_class_macro(multi_class_data):
    y_true, y_pred = multi_class_data
    eva = ModelEvaluator(y_true, y_pred)
    acc = eva.accuracy()
    prec_macro = eva.precision(average="macro")
    assert 0 <= acc <= 1
    assert 0 <= prec_macro <= 1
