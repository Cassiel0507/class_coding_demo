"""
需求文档：题目13 使用 ABC 创建统一的机器学习模型接口
难度：中等｜考点：abc.ABC、@abstractmethod、抽象基类、接口约束
背景：模型管理系统，强制所有模型必须实现 fit、predict；未实现则无法实例化

【导入依赖】
需要 import abc，从 abc 导入 ABC, abstractmethod

【类与输入约定】
抽象父类 BaseMLModel(ABC)
    - 抽象方法 fit(X, y)：必须由子类实现，接收训练特征 X、标签 y，无返回要求（模拟训练）
    - 抽象方法 predict(X)：必须由子类实现，接收待预测数据 X，返回模拟预测结果
    - 普通方法 get_model_type()：返回当前类的类名字符串

子类 RandomForestModel(BaseMLModel)
    - 实现 fit(X, y)：模拟训练，无真实训练逻辑
    - 实现 predict(X)：返回简单模拟预测列表/数组
    - get_model_type() 继承父类，返回 "RandomForestModel"

子类 LinearRegressionModel(BaseMLModel)
    - 实现 fit(X, y)：模拟训练
    - 实现 predict(X)：返回另一组模拟预测结果
    - get_model_type() 继承父类，返回 "LinearRegressionModel"

抽象约束：
任何继承 BaseMLModel，但没有同时实现 fit 和 predict 的类，实例化时抛出 TypeError

【输出约定】
- get_model_type()：返回类名字符串
- fit：执行后无强制返回值
- predict：接收 X，返回模拟预测结果（列表即可）

【异常约定】
继承 BaseMLModel、未实现全部 @abstractmethod 的类，尝试实例化必须抛出 TypeError

【验收标准（全部必须通过）】
1. RandomForestModel 实例 get_model_type() == "RandomForestModel"
2. LinearRegressionModel 实例 get_model_type() == "LinearRegressionModel"
3. RandomForestModel 可以正常调用 fit(X,y)、predict(X)，不报错，predict 返回模拟结果
4. LinearRegressionModel 可以正常调用 fit(X,y)、predict(X)，不报错，predict 返回模拟结果
5. 只继承 BaseMLModel、不实现 fit / predict 的 IncompleteModel，实例化触发 TypeError
6. BaseMLModel 本身不能直接实例化
"""

# 请在这里补全 BaseMLModel、RandomForestModel、LinearRegressionModel
from abc import ABC, abstractmethod


class BaseMLModel(ABC):
    @abstractmethod
    def fit(self, X, y):
        """模拟训练模型。"""

    @abstractmethod
    def predict(self, X):
        """返回模拟预测结果。"""

    def get_model_type(self):
        return type(self).__name__

class RandomForestModel(BaseMLModel):
    def fit(self, X, y):
        pass

    def predict(self, X):
        # 用 0 生成与输入样本数相同的预测结果。
        return [0] * len(X)

class LinearRegressionModel(BaseMLModel):
    def fit(self, X, y):
        pass

    def predict(self, X):
        return [1] * len(X)
