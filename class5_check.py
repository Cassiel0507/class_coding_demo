"""
【题目5：理解 @staticmethod 和 @classmethod ｜ 中等，DS面试高频】
类名：ExperimentManager
字段（实例属性）：
    experiment_name: str 实验名称
    random_state: int 随机种子
    parameters: dict 实验参数字典

方法清单：
1. 实例方法 get_info()
    输入：无（self）
    输出：dict，包含当前实例全部三个属性：{"experiment_name":..., "random_state":..., "parameters":...}

2. 静态方法 @staticmethod calculate_cv(values)
    作用：计算变异系数 CV = 标准差σ / 均值μ
    输入：可迭代数值序列（list / np.array）
    规则：
        - 均值为0 → 抛出 ValueError("mean is zero, cannot compute CV")
        - 输入为空序列 → 抛出 ValueError("input values cannot be empty")
        - 只接受数字，非数字不做额外处理（交给计算报错）
    输出：float CV值

3. 类方法 @classmethod from_config(cls, config)
    输入：config 字典，必须包含key：experiment_name, random_state, parameters
    逻辑：用config里面的值，返回一个新的 ExperimentManager 实例
    校验：缺少任一必填key抛出 KeyError

4. 类方法 @classmethod from_default(cls)
    输入：无
    默认配置固定为：
        experiment_name = "default_exp"
        random_state = 42
        parameters = {"lr":0.01, "epochs":100}
    输出：返回使用默认配置新建的 ExperimentManager 实例

=== 验收标准 ===
1. from_default() 生成实例，get_info() 能拿到上面默认配置
2. from_config() 读取自定义字典生成独立实例，属性和传入config一致
3. calculate_cv 是静态方法：不需要实例/类也能直接调用 ExperimentManager.calculate_cv()
4. from_config / from_default 是类方法，第一个参数必须是 cls
5. 边界case：空数组、均值=0，必须抛出指定异常
6. 浮点数对比允许微小误差（pytest用 pytest.approx）
"""

import numpy as np

class ExperimentManager:
    def __init__(self, experiment_name: str, random_state: int, parameters: dict):
        self.experiment_name = experiment_name
        self.random_state = random_state
        self.parameters = parameters

    def get_info(self):
        return {
            "experiment_name": self.experiment_name,
            "random_state": self.random_state,
            "parameters": self.parameters,
        }

    @staticmethod
    def calculate_cv(values):
        values = np.asarray(values)
        if values.size == 0:
            raise ValueError("input values cannot be empty")

        mean = np.mean(values)
        if mean == 0:
            raise ValueError("mean is zero, cannot compute CV")
        # NumPy 默认使用总体标准差（ddof=0）。
        return float(np.std(values) / mean)

    @classmethod
    def from_config(cls, config):
        return cls(
            config["experiment_name"],
            config["random_state"],
            config["parameters"],
        )

    @classmethod
    def from_default(cls):
        return cls("default_exp", 42, {"lr": 0.01, "epochs": 100})
