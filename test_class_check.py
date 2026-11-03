import pytest
from class_check import Dataset


def test_dataset_normal_case():
    # 正常数据集
    ds = Dataset(name="sensor_01", data=[10, 20, 30], description="温度传感器数据")
    assert ds.get_size() == 3
    assert ds.get_mean() == 20.0
    summary = ds.get_summary()
    assert summary["name"] == "sensor_01"
    assert summary["description"] == "温度传感器数据"
    assert summary["size"] == 3
    assert summary["mean"] == 20.0


def test_add_sample_valid():
    ds = Dataset(name="test", data=[1, 2])
    ds.add_sample(3)
    assert ds.get_size() == 3
    assert ds.get_mean() == 2.0


def test_add_sample_invalid_type():
    ds = Dataset(name="test", data=[1, 2])
    with pytest.raises(TypeError, match="样本必须是数字"):
        ds.add_sample("not_a_number")


def test_empty_data_get_mean():
    ds = Dataset(name="empty", data=[])
    with pytest.raises(ValueError, match="数据集为空，无法计算均值"):
        ds.get_mean()


def test_data_not_list_on_init():
    with pytest.raises(TypeError):
        Dataset(name="bad", data=12345)


def test_default_description():
    ds = Dataset(name="simple", data=[5, 5])
    assert ds.description == ""
    assert ds.get_summary()["description"] == ""
