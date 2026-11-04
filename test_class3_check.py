import pytest

from class3_check import ModelConfig


def test_normal_init_and_get_config():
    # 正常初始化，使用全部默认值
    cfg = ModelConfig(model_name="random_forest")
    assert cfg.get_config() == {
        "model_name": "random_forest",
        "n_estimators": 100,
        "max_depth": 5,
        "random_state": 42
    }

    # 自定义合法参数
    cfg2 = ModelConfig(
        model_name="lgb",
        n_estimators=200,
        max_depth=None,
        random_state=123
    )
    conf = cfg2.get_config()
    assert conf["model_name"] == "lgb"
    assert conf["max_depth"] is None


def test_validate_n_estimators():
    # n_estimators <=0 报错
    with pytest.raises(ValueError):
        cfg = ModelConfig("test", n_estimators=0)
        cfg.validate()
    with pytest.raises(ValueError):
        cfg = ModelConfig("test", n_estimators=-10)
        cfg.validate()
    # 类型错误
    with pytest.raises(TypeError):
        cfg = ModelConfig("test", n_estimators=100.5)
        cfg.validate()


def test_validate_max_depth():
    # max_depth 合法：None
    cfg = ModelConfig("test", max_depth=None)
    cfg.validate()
    # max_depth <=0 非法
    with pytest.raises(ValueError):
        cfg = ModelConfig("test", max_depth=0)
        cfg.validate()
    with pytest.raises(ValueError):
        cfg = ModelConfig("test", max_depth=-3)
        cfg.validate()
    # 类型错误（不是None也不是int）
    with pytest.raises(TypeError):
        cfg = ModelConfig("test", max_depth=3.5)
        cfg.validate()


def test_validate_random_state():
    # 负数random_state报错
    with pytest.raises(ValueError):
        cfg = ModelConfig("test", random_state=-1)
        cfg.validate()
    # 类型错误
    with pytest.raises(TypeError):
        cfg = ModelConfig("test", random_state=10.0)
        cfg.validate()


def test_model_name_type_check():
    with pytest.raises(TypeError):
        cfg = ModelConfig(model_name=12345)
        cfg.validate()


def test_update_method_normal():
    cfg = ModelConfig("rf")
    cfg.update(n_estimators=150, random_state=99)
    assert cfg.n_estimators == 150
    assert cfg.random_state == 99

    cfg.update(max_depth=None)
    assert cfg.max_depth is None


def test_update_unknown_key():
    cfg = ModelConfig("rf")
    with pytest.raises(ValueError):
        cfg.update(learning_rate=0.1)


def test_update_invalid_value_triggers_validate():
    cfg = ModelConfig("rf")
    with pytest.raises(ValueError):
        cfg.update(n_estimators=-5)
