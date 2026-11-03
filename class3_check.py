"""
题目 3：创建一个 ModelConfig 类｜DS 高频
难度：简单～中等

背景：
训练多个机器学习模型，每个模型有不同超参数；用类统一管理配置，避免硬编码。

【类名】ModelConfig
【初始化属性】
入参：
    model_name: str，无默认值，必须传入
    n_estimators: int，默认值 = 100
    max_depth: int | None，默认值 = 5
    random_state: int，默认值 = 42

【方法1：validate()】
功能：校验当前实例所有参数合法性，不合法抛出对应异常
校验规则：
1. n_estimators：必须 > 0；否则 ValueError
2. max_depth：>0 或者等于 None；否则 ValueError
3. random_state：必须是非负整数（>=0）；否则 ValueError
4. 类型校验：
   - model_name 必须是字符串，否则 TypeError
   - n_estimators 必须是整数，否则 TypeError
   - max_depth 如果不为 None，必须是整数，否则 TypeError
   - random_state 必须是整数，否则 TypeError

【方法2：update(**kwargs)】
功能：支持关键字传参修改属性，修改后自动执行 validate()
允许更新的key：model_name / n_estimators / max_depth / random_state
传入不支持的key：抛出 ValueError
修改后参数非法，由 validate 抛出异常

【方法3：get_config()】
功能：返回字典，key和属性名一致，value为当前属性值
返回示例：
{"model_name": "xgb", "n_estimators":100, "max_depth":5, "random_state":42}

【异常约定】
- 类型不对 → TypeError
- 数值规则不满足 → ValueError

【验收标准】
1. 正常初始化合法参数，不报错，get_config 返回正确字典
2. validate() 触发所有非法场景，抛出对应异常
3. update 合法参数更新成功；非法参数/未知key抛出异常
4. max_depth 允许 None（树模型常用，代表不限制深度）
"""

class ModelConfig:
    def __init__(
        self,
        model_name: str,
        n_estimators: int = 100,
        max_depth: int | None = 5,
        random_state: int = 42,
    ):
        self.model_name = model_name
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.validate()

    def validate(self):
        if not isinstance(self.model_name, str):
            raise TypeError("model_name 必须是字符串")
        if not isinstance(self.n_estimators, int) or isinstance(self.n_estimators, bool):
            raise TypeError("n_estimators 必须是整数")
        if self.max_depth is not None and (
            not isinstance(self.max_depth, int) or isinstance(self.max_depth, bool)
        ):
            raise TypeError("max_depth 必须是整数或 None")
        if not isinstance(self.random_state, int) or isinstance(self.random_state, bool):
            raise TypeError("random_state 必须是整数")

        if self.n_estimators <= 0:
            raise ValueError("n_estimators 必须大于 0")
        if self.max_depth is not None and self.max_depth <= 0:
            raise ValueError("max_depth 必须大于 0 或为 None")
        if self.random_state < 0:
            raise ValueError("random_state 必须是非负整数")

    def update(self, **kwargs):
        allowed = {"model_name", "n_estimators", "max_depth", "random_state"}
        unknown = kwargs.keys() - allowed
        if unknown:
            raise ValueError(f"不支持的配置项: {', '.join(sorted(unknown))}")

        for key, value in kwargs.items():
            setattr(self, key, value)
        self.validate()

    def get_config(self):
        return {
            "model_name": self.model_name,
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "random_state": self.random_state,
        }