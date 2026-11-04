"""
题目 4：创建一个 DataProcessor 类
难度：中等
DS 高频｜工业数据预处理类
背景：工业数据通常存在缺失值、异常值和不同量纲的问题。需要创建数据预处理类处理 DataFrame。

【初始化】
processor = DataProcessor(fill_value=0, normalize=True)
参数：
    fill_value：填充缺失值的值
    normalize：bool，是否开启 Min-Max 归一化

【必须实现方法】
1. fill_missing(df)
    功能：使用实例属性 fill_value 填充 df 的缺失值
    返回：填充后的新 DataFrame，不修改原df
2. normalize_data(df)
    功能：对数值列做 Min-Max Normalization
    公式：X_norm = (X - X_min) / (X_max - X_min)
    注意：本方法不用于流水线；归一化统计量由 fit 统一记录
3. fit(df)
    功能：在训练集上学习，记录每个数值列的 min、max，保存到实例内部
    约束：fit 只学习，不返回归一化结果；只读取统计量，不修改输入df
4. transform(df)
    功能：使用 fit 保存好的 min/max 做归一化；先填充缺失值
    约束：
        - 禁止重新计算min/max，只能读取fit缓存的值
        - 如果没有先调用fit，直接transform必须主动抛出异常
        - 返回新DataFrame，绝不修改原始df
        - 某一列 max == min 时，避免除零错误（该列归一后全部填0）
5. fit_transform(df)
    功能：依次执行 fit(df) -> transform(df)，返回变换结果

【输入输出约定】
输入：pandas.DataFrame
输出：pandas.DataFrame（新对象，原地修改禁止）

【验收标准 / 测试点】
1. 基础样例1
df = pd.DataFrame({
    "temperature": [20, 30, 40, 50],
    "pressure": [100, 200, 300, 400]
})
processor = DataProcessor(fill_value=0, normalize=True)
result = processor.fit_transform(df)
预期输出：
   temperature  pressure
0     0.000000  0.000000
1     0.333333  0.333333
2     0.666667  0.666667
3     1.000000  1.000000

2. 基础样例2（transform新数据，复用fit得到的min/max）
new_df = pd.DataFrame({
    "temperature": [25, 45],
    "pressure": [150, 350]
})
result = processor.transform(new_df)
预期输出：
   temperature  pressure
0     0.166667  0.166667
1     0.833333  0.833333

3. 异常分支
- 未fit直接调用transform：抛出异常
- 存在缺失值：被 fill_value 填充
- 某数值列所有值相同（max=min）：归一化后该列为0，不除零报错
- 原始传入df不会被函数原地修改
4. 开关校验：normalize=False 时，transform只填充缺失值，不做归一化
"""


# ========== 在这里写你的 DataProcessor 类 ==========
class DataProcessor:
    def __init__(self, fill_value=0, normalize=True):
        self.fill_value = fill_value
        self.normalize = normalize
        self._stats = {}
        self._fitted = False

    def fill_missing(self, df):
        return df.fillna(self.fill_value).copy()

    def normalize_data(self, df):
        result = df.copy()
        for column in result.select_dtypes(include="number").columns:
            minimum = result[column].min()
            maximum = result[column].max()
            if maximum == minimum:
                result[column] = 0
            else:
                result[column] = (result[column] - minimum) / (maximum - minimum)
        return result

    def fit(self, df):
        self._stats = {
            column: (df[column].min(), df[column].max())
            for column in df.select_dtypes(include="number").columns
        }
        self._fitted = True

    def transform(self, df):
        if not self._fitted:
            raise RuntimeError("必须先调用 fit()")

        result = self.fill_missing(df)
        if self.normalize:
            for column, (minimum, maximum) in self._stats.items():
                if column not in result.columns:
                    continue
                # 常数列统一归一化为 0，避免除零。
                if maximum == minimum:
                    result[column] = 0
                else:
                    result[column] = (result[column] - minimum) / (maximum - minimum)
        return result

    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)
