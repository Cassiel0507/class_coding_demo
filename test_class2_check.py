import pytest

from class2_check import Experiment
from class_check import Dataset


def test_experiment_basic():
    ds = Dataset(name="sensor", data=[1,2,3])
    exp = Experiment("batch_001", ds)
    assert exp.experiment_name == "batch_001"
    assert exp.dataset is ds
    assert exp.get_results() == []


def test_add_and_get_result():
    ds = Dataset(name="sensor", data=[1,2,3])
    exp = Experiment("batch_001", ds)
    exp.add_result(0.92)
    exp.add_result({"mae":0.05})
    res = exp.get_results()
    assert res == [0.92, {"mae":0.05}]


def test_clear_results():
    ds = Dataset(name="sensor", data=[1,2,3])
    exp = Experiment("batch_001", ds)
    exp.add_result(0.88)
    exp.clear_results()
    assert exp.get_results() == []


def test_two_experiment_isolation():
    ds = Dataset(name="sensor", data=[1,2,3])
    exp1 = Experiment("batch_A", ds)
    exp2 = Experiment("batch_B", ds)

    exp1.add_result(10)
    exp2.add_result(20)

    assert exp1.get_results() == [10]
    assert exp2.get_results() == [20]

    exp1.clear_results()
    assert exp1.get_results() == []
    assert exp2.get_results() == [20]


def test_init_name_not_string():
    ds = Dataset(name="sensor", data=[1,2,3])
    with pytest.raises(TypeError):
        Experiment(999, ds)


def test_init_dataset_not_dataset():
    with pytest.raises(TypeError):
        Experiment("batch_001", [1,2,3])
