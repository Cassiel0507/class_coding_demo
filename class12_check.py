"""
需求文档：题目12 使用 super() 扩展父类的初始化和方法
难度：简单～中等｜考点：super()、继承、方法重写、复用父类逻辑
背景：工业预测性维护，不同设备模型共享基础字段，子类追加自有参数，用 super 避免重复代码

【类与输入约定】
父类 BasePredictiveModel
    __init__(machine_name, model_name)
        machine_name: str 设备名称
        model_name: str 模型名称
        is_trained: bool，默认 False
    get_info() -> dict：返回 {"machine_name", "model_name", "is_trained"}
    fit() -> None：将 self.is_trained 设置为 True

子类 VibrationModel(BasePredictiveModel)
    __init__(machine_name, sampling_rate=1000, frequency_range=500)
        使用 super().__init__ 调用父类初始化
        model_name 固定字符串："VibrationModel"
        sampling_rate：采样率，默认 1000
        frequency_range：频率上限，默认 500
    get_info() -> dict：
        调用 super().get_info()拿到父类字典，再追加 sampling_rate、frequency_range

子类 TemperatureModel(BasePredictiveModel)
    __init__(machine_name, threshold=80, window_size=10)
        使用 super().__init__ 调用父类初始化
        model_name 固定字符串："TemperatureModel"
        threshold：温度报警阈值，默认 80
        window_size：滑动窗口大小，默认 10
    get_info() -> dict：
        调用 super().get_info()拿到父类字典，再追加 threshold、window_size

【输出约定】
- get_info 返回字典，包含父类全部键 + 子类自有参数键
- fit() 执行后，字典内 is_trained 变为 True
- 默认参数不传时，使用题目给定默认值

【异常约定】
本题不强制参数类型校验，重点考察 super() 调用、方法覆写。

【验收标准（全部必须通过）】
1. VibrationModel 实例 get_info：
    machine_name、model_name="VibrationModel"、is_trained=False、sampling_rate、frequency_range
2. TemperatureModel 实例 get_info：
    machine_name、model_name="TemperatureModel"、is_trained=False、threshold、window_size
3. 调用 fit() 之后，对应实例 get_info()["is_trained"] == True
4. 使用默认参数构造实例时，参数取题目默认值
5. 子类必须使用 super() 调用父类 __init__ 与 get_info，禁止手动重复赋值父类字段
"""

# 请在这里补全 BasePredictiveModel、VibrationModel、TemperatureModel
class BasePredictiveModel:
    def __init__(self, machine_name: str, model_name: str):
        self.machine_name = machine_name
        self.model_name = model_name
        self.is_trained = False

    def get_info(self) -> dict[str, object]:
        return {
            "machine_name": self.machine_name,
            "model_name": self.model_name,
            "is_trained": self.is_trained,
        }

    def fit(self) -> None:
        self.is_trained = True

class VibrationModel(BasePredictiveModel):
    def __init__(self, machine_name: str, sampling_rate: int = 1000, frequency_range: int = 500):
        super().__init__(machine_name, "VibrationModel")
        self.sampling_rate = sampling_rate
        self.frequency_range = frequency_range

    def get_info(self) -> dict[str, object]:
        # 复用父类信息，再添加振动模型参数。
        info = super().get_info()
        info.update({"sampling_rate": self.sampling_rate, "frequency_range": self.frequency_range})
        return info

class TemperatureModel(BasePredictiveModel):
    def __init__(self, machine_name: str, threshold: int = 80, window_size: int = 10):
        super().__init__(machine_name, "TemperatureModel")
        self.threshold = threshold
        self.window_size = window_size

    def get_info(self) -> dict[str, object]:
        info = super().get_info()
        info.update({"threshold": self.threshold, "window_size": self.window_size})
        return info
