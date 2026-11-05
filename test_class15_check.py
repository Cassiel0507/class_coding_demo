import pytest
from class15_check import (
    ModelRegistryMeta,
    BaseModel,
    RandomForestModel,
    LogisticRegressionModel
)

def test_registry_content():
    reg = ModelRegistryMeta.registry
    # 两个子类必须在注册表
    assert "RandomForestModel" in reg
    assert reg["RandomForestModel"] is RandomForestModel
    assert "LogisticRegressionModel" in reg
    assert reg["LogisticRegressionModel"] is LogisticRegressionModel
    # BaseModel 不能注册进去
    assert "BaseModel" not in reg

def test_can_instantiate_from_registry():
    model_cls = ModelRegistryMeta.registry["RandomForestModel"]
    inst = model_cls()
    assert type(inst).__name__ == "RandomForestModel"

def test_new_subclass_auto_register():
    # 动态新建一个子类，验证自动注册
    class XGBoostModel(BaseModel):
        pass
    reg = ModelRegistryMeta.registry
    assert "XGBoostModel" in reg
    assert reg["XGBoostModel"] is XGBoostModel

def test_registry_is_class_attr_of_meta():
    assert hasattr(ModelRegistryMeta, "registry")
    assert isinstance(ModelRegistryMeta.registry, dict)
