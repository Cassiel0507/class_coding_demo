# class7_check.py
"""
题目7：创建一个 ModelEvaluator 类
难度：中等
场景：机器学习分类模型评估，Data Scientist面试高频

【输入定义】
初始化：evaluator = ModelEvaluator(y_true=y_test, y_pred=y_pred)
- y_true：array-like，真实标签
- y_pred：array-like，模型预测标签

【成员方法清单】
1. accuracy()
   返回 float：准确率
2. precision(average="binary")
   返回 float：精确率，支持 average 参数：binary / macro / weighted
3. recall(average="binary")
   返回 float：召回率，支持 average 参数
4. f1_score(average="binary")
   返回 float：F1分数，支持 average 参数
5. get_all_metrics(average="binary")
   返回 dict：{"accuracy":xxx, "precision":xxx, "recall":xxx, "f1":xxx}
6. compare(other_y_pred, average="binary")
   other_y_pred：另一个模型的预测结果（和当前y_true共用同一套真实标签）
   返回 dict，包含当前模型、对比模型两套指标

【约束 & 异常约定】
1. y_true 和 y_pred 长度不一致 → 抛出异常
2. compare 方法中 other_y_pred 和 y_true 长度不一致 → 抛出异常
3. 底层使用 sklearn.metrics 对应函数
4. 支持二分类、多分类，由 average 参数控制
5. 不修改传入的原始数组

【验收标准 / 测试用例参考】
二分类测试数据：
y_true = [0, 0, 0, 1, 1, 1]
y_pred = [0, 0, 1, 1, 1, 1]
y_pred_other = [0, 1, 0, 1, 0, 1]

evaluator = ModelEvaluator(y_true, y_pred)
evaluator.accuracy()  # 预期 5/6 ≈0.8333
evaluator.precision(average="binary")
evaluator.recall(average="binary")
evaluator.f1_score(average="binary")
evaluator.get_all_metrics()
evaluator.compare(other_y_pred=y_pred_other)


"""
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


class ModelEvaluator:
    def __init__(self, y_true, y_pred):
        self.y_true = list(y_true)
        self.y_pred = list(y_pred)
        if len(self.y_true) != len(self.y_pred):
            raise ValueError("y_true and y_pred must have the same length")

    def accuracy(self):
        return float(accuracy_score(self.y_true, self.y_pred))

    def precision(self, average="binary"):
        return float(precision_score(self.y_true, self.y_pred, average=average))

    def recall(self, average="binary"):
        return float(recall_score(self.y_true, self.y_pred, average=average))

    def f1_score(self, average="binary"):
        return float(f1_score(self.y_true, self.y_pred, average=average))

    def get_all_metrics(self, average="binary"):
        return {
            "accuracy": self.accuracy(),
            "precision": self.precision(average),
            "recall": self.recall(average),
            "f1": self.f1_score(average),
        }

    def compare(self, other_y_pred, average="binary"):
        other_y_pred = list(other_y_pred)
        if len(other_y_pred) != len(self.y_true):
            raise ValueError("other_y_pred and y_true must have the same length")
        # 对比模型复用真实标签，当前模型和输入数据保持不变。
        other = ModelEvaluator(self.y_true, other_y_pred)
        return {
            "self": self.get_all_metrics(average),
            "other": other.get_all_metrics(average),
        }
