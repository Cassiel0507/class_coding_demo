# class6_check.py
"""
题目6：创建一个 IndustrialDataset 类
难度：中等
场景：工业DS传感器故障数据集封装

【输入定义】
初始化：IndustrialDataset(data=df, target_column="failure")
- data：pandas DataFrame，原始传感器数据表
- target_column：字符串，目标列名称（故障标签列）

【必须实现的方法清单】
1. get_features()
   返回：DataFrame，全部特征列，不含target列
2. get_target()
   返回：Series，目标标签列
3. get_shape()
   返回：元组 (n_rows, n_cols)，原始数据集行列数
4. get_missing_summary()
   返回：Series，每一列缺失值数量
5. get_class_distribution()
   返回：Series，目标分类变量频次统计（value_counts）
6. split_features_target()
   返回：(X, y)，X是特征DataFrame，y是目标Series

【约束 & 异常约定】
1. 传入的 target_column 在data中不存在 → 主动抛出异常
2. 所有操作**禁止修改原始传入的DataFrame**
3. X、y 应当是独立副本，避免链式赋值污染原始数据

【验收标准 / 预期结果（测试df）】
df = pd.DataFrame({
    "temperature": [20, 30, 40, 50, 60],
    "pressure": [100, 200, 150, 300, 250],
    "vibration": [0.1, 0.2, 0.3, 0.4, 0.5],
    "failure": [0, 0, 1, 0, 1]
})
dataset = IndustrialDataset(df, target_column="failure")
dataset.get_shape() == (5, 4)
dataset.get_features().columns.tolist() == ['temperature', 'pressure', 'vibration']
dataset.get_target().tolist() == [0, 0, 1, 0, 1]
dataset.get_class_distribution().to_dict() == {0:3, 1:2}


"""
import pandas as pd


class IndustrialDataset:
    def __init__(self, data: pd.DataFrame, target_column: str):
        if target_column not in data.columns:
            raise KeyError(target_column)
        self.data = data.copy()
        self.target_column = target_column

    def get_features(self):
        return self.data.drop(columns=self.target_column).copy()

    def get_target(self):
        return self.data[self.target_column].copy()

    def get_shape(self):
        return self.data.shape

    def get_missing_summary(self):
        return self.data.isna().sum()

    def get_class_distribution(self):
        return self.get_target().value_counts()

    def split_features_target(self):
        # 特征和目标分别返回副本，避免修改结果时影响数据集。
        return self.get_features(), self.get_target()