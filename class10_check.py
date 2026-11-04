# param10_check.py
"""
题目：创建一个完整的 MLExperiment 类
需求文档：
【输入定义】
1. __init__ 参数
   - experiment_name: str，实验名称
   - random_state: int，随机种子，保证可复现
2. set_dataset(X, y)
   - X: np.ndarray / pd.DataFrame，特征矩阵
   - y: np.ndarray / pd.Series，标签
3. split_data(test_size=0.2)
   - test_size: float，测试集占比 (0~1)
4. set_model(model)
   - model: 自定义模型对象（需要拥有 fit、predict 方法，如 RandomForestModel）
5. run()：无入参
6. evaluate()：无入参
7. get_results()：无入参
8. get_experiment_info()：无入参

【输出定义】
- get_results() 返回字典：包含实验名、accuracy、precision、recall、f1_score
- get_experiment_info() 返回字典：实验名、随机种子、是否设置数据集、是否划分、是否设置模型、是否训练完成
- split_data 内部使用 sklearn train_test_split，绑定实例 random_state
- run：在训练集 fit，在测试集预测，保存 y_pred
- evaluate：使用 ModelEvaluator 计算四个分类指标

【状态约定 & 异常规则】
状态标记：
  has_dataset: bool
  splitted: bool
  model_ready: bool
  trained: bool
异常：
  1. 未调用 set_dataset，直接 run() → 抛出 ValueError("Dataset not set, call set_dataset first.")
  2. 未 split_data 直接 run() → ValueError("Data not splitted, call split_data first.")
  3. 没有 set_model 直接 run() → ValueError("Model not set, call set_model first.")
  4. 没有 run()（未训练）直接 evaluate() → ValueError("Model not trained, call run() first.")

【依赖约定】
本类复用外部两个类（不在本文件实现，假定已经定义好）：
- RandomForestModel：具备 fit(X_train,y_train)、predict(X_test)
- ModelEvaluator：静态/实例方法 calculate_metrics(y_true, y_pred)，返回 {accuracy,precision,recall,f1_score}

【验收标准】
按照给定测试流程顺序调用，无报错；
get_results 返回格式匹配样例；
random_state 贯穿切分与模型，保证可复现；
职责分离：只做实验编排，不重新实现模型、评估器；
禁止 __init__ 里面自动训练；
状态信息由 get_experiment_info 对外暴露；
边界：test_size 在合法区间，不做额外校验（交给 train_test_split）。
"""
from sklearn.model_selection import train_test_split


class MLExperiment:
    def __init__(self, experiment_name: str, random_state: int):
        self.experiment_name = experiment_name
        self.random_state = random_state

        # 数据
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        # 模型
        self.model = None
        self.y_pred = None

        # 指标
        self.metrics = None

        # 状态标记
        self.has_dataset = False
        self.splitted = False
        self.model_ready = False
        self.trained = False

    def set_dataset(self, X, y):
        self.X = X
        self.y = y
        self.has_dataset = True
        self.X_train = self.X_test = self.y_train = self.y_test = None
        self.splitted = False
        self._clear_run_state()

    def _clear_run_state(self):
        self.y_pred = None
        self.metrics = None
        self.trained = False

    def split_data(self, test_size=0.2):
        if not self.has_dataset:
            raise ValueError("Dataset not set, call set_dataset first.")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y,
            test_size=test_size,
            random_state=self.random_state
        )
        self.splitted = True
        self._clear_run_state()

    def set_model(self, model):
        # model 需要支持 fit / predict
        self.model = model
        self.model_ready = True
        self._clear_run_state()

    def run(self):
        if not self.has_dataset:
            raise ValueError("Dataset not set, call set_dataset first.")
        if not self.splitted:
            raise ValueError("Data not splitted, call split_data first.")
        if not self.model_ready:
            raise ValueError("Model not set, call set_model first.")

        self._clear_run_state()
        self.model.fit(self.X_train, self.y_train)
        self.y_pred = self.model.predict(self.X_test)
        self.trained = True

    def evaluate(self):
        if not self.trained:
            raise ValueError("Model not trained, call run() first.")
        # 复用外部评估器
        self.metrics = ModelEvaluator.calculate_metrics(self.y_test, self.y_pred)

    def get_results(self):
        if self.metrics is None:
            raise ValueError("No evaluation result, call evaluate() first.")
        out = {
            "experiment_name": self.experiment_name,
            "accuracy": self.metrics["accuracy"],
            "precision": self.metrics["precision"],
            "recall": self.metrics["recall"],
            "f1_score": self.metrics["f1_score"]
        }
        return out

    def get_experiment_info(self):
        info = {
            "experiment_name": self.experiment_name,
            "random_state": self.random_state,
            "has_dataset": self.has_dataset,
            "splitted": self.splitted,
            "model_ready": self.model_ready,
            "trained": self.trained
        }
        return info


# ========== 下面是配套的支撑类（供你本地跑通整套测试，放在同一个文件也行） ==========
class RandomForestModel:
    def __init__(self, n_estimators, max_depth, random_state):
        from sklearn.ensemble import RandomForestClassifier
        self.clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        )

    def fit(self, X, y):
        self.clf.fit(X, y)

    def predict(self, X):
        return self.clf.predict(X)


class ModelEvaluator:
    @staticmethod
    def calculate_metrics(y_true, y_pred):
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            precision_score,
            recall_score,
        )
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1_score": float(f1_score(y_true, y_pred, zero_division=0))
        }


if __name__ == "__main__":
    from sklearn.datasets import make_classification
    
    print(" 1. 初始化实验...")
    # 实例化我们的实验管理类
    exp = MLExperiment(experiment_name="RandomForest_v1", random_state=42)
    
    print(" 2. 生成模拟数据并载入...")
    # 生成一个简单的二分类数据集 (1000个样本，20个特征)
    X_raw, y_raw = make_classification(n_samples=1000, n_features=20, random_state=42)
    exp.set_dataset(X_raw, y_raw)
    
    print(" 3. 切分数据集...")
    exp.split_data(test_size=0.3)
    
    print(" 4. 配置模型...")
    # 实例化配套的随机森林模型
    rf_model = RandomForestModel(n_estimators=100, max_depth=5, random_state=42)
    exp.set_model(rf_model)
    
    # 我们可以打印一下此时的实验状态
    print("当前实验状态:", exp.get_experiment_info())
    
    print(" 5. 运行训练与预测...")
    exp.run()
    
    print(" 6. 评估模型...")
    exp.evaluate()
    
    print("\n================ 实验最终结果 ================")
    import json
    # 用 json.dumps 让打印出来的字典更好看
    print(json.dumps(exp.get_results(), indent=4))
    print("==============================================")
