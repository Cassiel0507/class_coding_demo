"""
需求文档：题目11 创建基础机器学习模型继承体系
难度：简单｜考点：类继承、父类/子类、方法复用、面向对象基础
背景：DS项目里不同模型共享公共逻辑，用继承消除重复代码

【输入约定】
BaseModel（父类）
    __init__(model_name)
        model_name: str，模型名字
        is_trained: bool，训练状态，默认 False
RandomForestModel(BaseModel)
    __init__(n_estimators, max_depth)
        n_estimators: int，树数量
        max_depth: int，树最大深度
        固定 model_name = "RandomForest"
LogisticRegressionModel(BaseModel)
    __init__(C, max_iter)
        C: float，正则系数
        max_iter: int，最大迭代次数
        固定 model_name = "LogisticRegression"

【方法定义】
BaseModel：
    get_model_name() -> str：返回模型名称
    get_status() -> bool：返回 is_trained
    set_trained() -> None：把 is_trained 设置为 True

RandomForestModel：
    fit() -> None：调用 set_trained()，标记为已训练
    predict() -> list[int]：固定返回 [0, 1, 0]

LogisticRegressionModel：
    fit() -> None：调用 set_trained()，标记为已训练
    predict() -> list[int]：固定返回 [1, 0, 1]

【输出约定】
- get_model_name：返回对应模型名字符串
- get_status：返回布尔值，表示是否训练完成
- predict：返回固定的模拟预测列表
- isinstance(rf, BaseModel)、isinstance(lr, BaseModel) 都为 True

【异常约定】
本题不强制参数校验；子类构造函数只需接收指定超参，不做额外报错。

【验收标准（全部必须通过）】
1. rf.get_model_name() == "RandomForest"
2. lr.get_model_name() == "LogisticRegression"
3. 刚实例化时 rf.get_status() 为 False
4. rf.fit() 之后 rf.get_status() 变为 True
5. rf.predict() == [0, 1, 0]
6. lr.predict() == [1, 0, 1]
7. isinstance(rf, BaseModel) is True
8. isinstance(lr, BaseModel) is True
"""

# 请在这里补全 BaseModel、RandomForestModel、LogisticRegressionModel
class BaseModel:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.is_trained = False

    def get_model_name(self) -> str:
        return self.model_name

    def get_status(self) -> bool:
        return self.is_trained

    def set_trained(self) -> None:
        self.is_trained = True

class RandomForestModel(BaseModel):
    def __init__(self, n_estimators: int, max_depth: int):
        super().__init__("RandomForest")
        self.n_estimators = n_estimators
        self.max_depth = max_depth

    def fit(self) -> None:
        # 训练完成后更新父类中的状态。
        self.set_trained()

    def predict(self) -> list[int]:
        return [0, 1, 0]

class LogisticRegressionModel(BaseModel):
    def __init__(self, C: float, max_iter: int):
        super().__init__("LogisticRegression")
        self.C = C
        self.max_iter = max_iter

    def fit(self) -> None:
        self.set_trained()

    def predict(self) -> list[int]:
        return [1, 0, 1]
