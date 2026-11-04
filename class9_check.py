# class9_check.py
"""
需求文档：创建可保存加载的 RandomForestModel 类（工程实践）
背景：在题目8基础上封装随机森林，支持模型序列化、配置独立导出，适合工业AI项目落盘。
类名：RandomForestModel
依赖：sklearn.ensemble.RandomForestClassifier / Regressor，joblib，json，os

【输入定义】
初始化入参：
    n_estimators: int，树数量
    max_depth: int | None，单树最大深度
    random_state: int，随机种子
    **kwargs：传给底层随机森林的其他参数
方法入参：
    fit(X, y): X 二维特征数组(DataFrame/np.ndarray)，y 标签数组
    save(path: str): 模型二进制保存路径（pkl，joblib）
    load(path: str): 模型二进制读取路径
    save_config(path: str): json 配置文件路径，只存超参，不存训练权重
    load_config(path: str): 从 json 读取超参，更新实例参数（不覆盖已经训练好的模型权重）
    get_model_info(): 无入参
    predict(X): 特征矩阵

【输出定义】
fit：无返回，标记实例已训练，保存底层 sklearn 模型对象
save / save_config：无返回，写入文件
load / load_config：无返回，修改实例内部状态
get_model_info：dict，包含：
    "model_name": str
    "params": dict 超参
    "is_trained": bool 是否完成训练
predict：预测结果数组，和 sklearn 原生 predict 输出对齐

【异常约定（必须抛出）】
1. save()：如果模型未训练，抛出 ValueError("Model not trained, cannot save.")
2. load()：path 文件不存在 → FileNotFoundError；文件损坏/不是合法模型 → 抛出合理异常
3. load_config()：配置文件不存在 → FileNotFoundError；json 格式非法 → json.JSONDecodeError
4. predict()：模型未训练时调用 predict → ValueError("Model not trained.")
5. fit 输入校验：X,y 长度不一致抛出 ValueError

【验收标准】
1. 训练后调用 save，再新建一个不同初始化参数的实例，调用 load，加载后直接 predict，不用 fit
2. 加载后的预测结果和原模型完全一致（相同输入输出相同）
3. save_config / load_config 只读写超参数，不会保存/覆盖模型权重
4. get_model_info 正确返回模型名、参数、训练标记
5. 所有禁止操作（未训练保存、未训练预测）触发对应异常
6. 文件不存在场景触发对应文件异常
7. 底层使用 joblib 存模型，json 存配置，二者分开

【使用示例】
model = RandomForestModel(n_estimators=100, max_depth=5, random_state=42)
model.fit(X_train, y_train)
model.save("random_forest.pkl")

new_model = RandomForestModel(n_estimators=10)
new_model.load("random_forest.pkl")
pred = new_model.predict(X_test)
assert pred.all() == model.predict(X_test).all()
"""
import json
import os

import joblib
from sklearn.ensemble import RandomForestClassifier


class RandomForestModel:
    def __init__(self, n_estimators: int, max_depth=None, random_state: int = 42, **kwargs):
        # 请自行实现
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.kwargs = kwargs
        self._model = None
        self._is_trained = False

    def fit(self, X, y):
        if len(X) != len(y):
            raise ValueError("X and y must have the same length")
        model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            **self.kwargs,
        )
        model.fit(X, y)
        self._model = model
        self._is_trained = True

    def save(self, path: str):
        if not self._is_trained:
            raise ValueError("Model not trained, cannot save.")
        joblib.dump({
            "model": self._model,
            "params": {
                "n_estimators": self.n_estimators,
                "max_depth": self.max_depth,
                "random_state": self.random_state,
                "kwargs": self.kwargs,
            },
        }, path)

    def load(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        saved = joblib.load(path)
        # 1. 严格的类型检查：如果不是字典，抛出标准的 TypeError
        if not isinstance(saved, dict):
            raise TypeError("Loaded object must be a dictionary")

        # 2. 内容合法性检查：如果类型对但里面的核心内容不对，抛出 ValueError
        if (not isinstance(saved.get("model"), RandomForestClassifier)
                or not isinstance(saved.get("params"), dict)):
            raise TypeError("Invalid RandomForestModel file structure: mismatched types")


        self._model = saved["model"]
        params = saved["params"]
        self.n_estimators = params["n_estimators"]
        self.max_depth = params["max_depth"]
        self.random_state = params["random_state"]
        self.kwargs = params["kwargs"]
        self._is_trained = True

    def save_config(self, path: str):
        with open(path, "w", encoding="utf-8") as config_file:
            json.dump({
                "n_estimators": self.n_estimators,
                "max_depth": self.max_depth,
                "random_state": self.random_state,
                "kwargs": self.kwargs,
            }, config_file)

    def load_config(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        with open(path, encoding="utf-8") as config_file:
            params = json.load(config_file)
        if not isinstance(params, dict) or not all(
            key in params for key in ("n_estimators", "max_depth", "random_state")
        ):
            raise ValueError("Invalid RandomForestModel config")
        # 只更新配置参数，不影响已训练模型。
        self.n_estimators = params["n_estimators"]
        self.max_depth = params["max_depth"]
        self.random_state = params["random_state"]
        self.kwargs = params.get("kwargs", {})

    def get_model_info(self) -> dict:
        return {
            "model_name": "RandomForestModel",
            "params": {
                "n_estimators": self.n_estimators,
                "max_depth": self.max_depth,
                "random_state": self.random_state,
                **self.kwargs,
            },
            "is_trained": self._is_trained,
        }

    def predict(self, X):
        if not self._is_trained:
            raise ValueError("Model not trained.")
        return self._model.predict(X)
