# class8_check.py
"""
题目8：创建一个 RandomForestModel 类
难度：中等～较难
场景：工业设备故障预测，随机森林分类封装，DS面试高频

【输入定义】
初始化：model = RandomForestModel(n_estimators=100, max_depth=5, random_state=42)
- n_estimators：树数量，int
- max_depth：单棵树最大深度，int / None
- random_state：随机种子，用于复现

【成员方法清单】
1. build_model()
   返回：RandomForestClassifier 实例，使用实例保存的参数构造
2. fit(X_train, y_train)
   训练模型；内部调用 build_model，更新实例内部模型状态
   返回：self（方便链式调用）
3. predict(X_test)
   返回：预测标签数组
   ⚠️ 如果模型尚未训练，抛出异常
4. predict_proba(X_test)
   返回：各类别概率数组
   ⚠️ 如果模型尚未训练，抛出异常
5. evaluate(X_test, y_test)
   返回 dict：{"accuracy", "precision", "recall", "f1"}，默认 average="weighted"
6. get_feature_importance()
   返回 pd.DataFrame，包含 feature_index / importance，按 importance 降序
7. get_params()
   返回 dict，当前模型使用的超参（n_estimators, max_depth, random_state）

【约束 & 异常约定】
1. __init__ 只保存超参，**不能直接训练模型**
2. 模型实例保存在内部，首次 fit 才真正构建 + 训练；多次 fit 会覆盖旧模型
3. 在 fit 之前调用 predict / predict_proba / evaluate / get_feature_importance → 抛异常
4. random_state 由实例统一管理，不允许在方法内硬编码
5. 评估指标使用 sklearn metrics，多分类默认 weighted
6. 不修改传入的原始 X、y

【验收参考测试流程】
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

data = load_breast_cancer()
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

rf = RandomForestModel(n_estimators=50, max_depth=4, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
proba = rf.predict_proba(X_test)
metrics = rf.evaluate(X_test, y_test)
imp_df = rf.get_feature_importance()
params = rf.get_params()

预期：
- metrics 四个指标都在 (0,1)
- imp_df 按重要性降序排列
- params 和初始化传入参数一致
- fit 前调用 predict 直接报错

"""
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


class RandomForestModel:
    def __init__(self, n_estimators=100, max_depth=5, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        # 首次 fit 前模型保持未训练状态。
        self.model = None

    def build_model(self):
        return RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
        )

    def fit(self, X_train, y_train):
        self.model = self.build_model()
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X_test):
        if self.model is None:
            raise RuntimeError("model has not been trained")
        return self.model.predict(X_test)

    def predict_proba(self, X_test):
        if self.model is None:
            raise RuntimeError("model has not been trained")
        return self.model.predict_proba(X_test)

    def evaluate(self, X_test, y_test, average="weighted"):
        y_pred = self.predict(X_test)
        return {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average=average),
            "recall": recall_score(y_test, y_pred, average=average),
            "f1": f1_score(y_test, y_pred, average=average),
        }

    def get_feature_importance(self):
        if self.model is None:
            raise RuntimeError("model has not been trained")
        return (
            pd.DataFrame({
                "feature_index": range(len(self.model.feature_importances_)),
                "importance": self.model.feature_importances_,
            })
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    def get_params(self):
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "random_state": self.random_state,
        }
