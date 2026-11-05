# test_class12_check.py
from class12_check import BasePredictiveModel, VibrationModel, TemperatureModel


def test_vibration_info_custom_args():
    vib = VibrationModel(machine_name="Motor A", sampling_rate=2000, frequency_range=800)
    info = vib.get_info()
    assert info["machine_name"] == "Motor A"
    assert info["model_name"] == "VibrationModel"
    assert info["is_trained"] is False
    assert info["sampling_rate"] == 2000
    assert info["frequency_range"] == 800


def test_temperature_info_custom_args():
    temp = TemperatureModel(machine_name="Machine B", threshold=90, window_size=20)
    info = temp.get_info()
    assert info["machine_name"] == "Machine B"
    assert info["model_name"] == "TemperatureModel"
    assert info["is_trained"] is False
    assert info["threshold"] == 90
    assert info["window_size"] == 20


def test_fit_change_train_status():
    vib = VibrationModel("Motor A", 2000, 800)
    vib.fit()
    assert vib.get_info()["is_trained"] is True

    temp = TemperatureModel("Machine B", 90, 20)
    temp.fit()
    assert temp.get_info()["is_trained"] is True


def test_default_parameters():
    vib = VibrationModel(machine_name="V1")
    info = vib.get_info()
    assert info["sampling_rate"] == 1000
    assert info["frequency_range"] == 500

    temp = TemperatureModel(machine_name="T1")
    info = temp.get_info()
    assert info["threshold"] == 80
    assert info["window_size"] == 10


def test_inheritance():
    vib = VibrationModel("Motor A")
    temp = TemperatureModel("Machine B")
    assert isinstance(vib, BasePredictiveModel)
    assert isinstance(temp, BasePredictiveModel)
