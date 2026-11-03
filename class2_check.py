"""
题目 2：管理多个实验的数据集｜难度：简单｜高频考点
背景：工业场景，同时维护多组不同机器/批次的实验，多实例隔离是核心考察点

【需求清单】
实现 Experiment 类
属性：
    experiment_name: str，实验名称
    dataset: Dataset 类型（就是上一题的 Dataset 对象）
    results: list，存放实验结果，实例初始化时默认是空列表 []
方法：
1. __init__(self, experiment_name: str, dataset: Dataset)
    参数：
        experiment_name：字符串实验名
        dataset：Dataset 实例（来自 class_check.py）
    初始化：self.results = []
    约束：
        - experiment_name 必须是 str，否则抛 TypeError
        - dataset 必须是 Dataset 实例，否则抛 TypeError
2. add_result(self, result)
    输入：result，支持任意类型结果（数值、字典都可以）
    行为：追加到 self.results
3. get_results(self)
    输入：无
    返回：list，返回当前实例的 results（返回副本防止外部原地篡改，可选加固点）
4. clear_results(self)
    输入：无
    行为：把 self.results 置为空列表

【核心验收重点（面试高频：实例隔离）】
✅ 新建两个独立 Experiment 对象 exp1、exp2
✅ exp1.add_result() 增加的结果，绝对不会出现在 exp2.results 里
✅ 各自 clear_results() 只清空自己的列表，互不干扰

【异常规则】
✅ __init__ 传入非字符串 experiment_name → TypeError
✅ __init__ 传入不是 Dataset 的对象 → TypeError
"""
# 下方写你的 Experiment 类代码
from class_check import Dataset


class Experiment:
    def __init__(self, experiment_name: str, dataset: Dataset):
        if not isinstance(experiment_name, str):
            raise TypeError("experiment_name 必须是字符串")
        if not isinstance(dataset, Dataset):
            raise TypeError("dataset 必须是 Dataset 实例")

        self.experiment_name = experiment_name
        self.dataset = dataset
        self.results = []  # 每个实验各自保存结果。

    def add_result(self, result):
        self.results.append(result)

    def get_results(self):
        return self.results.copy()

    def clear_results(self):
        self.results = []
