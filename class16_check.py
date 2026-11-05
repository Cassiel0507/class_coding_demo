"""
【题目 16：Design Patterns——设计模式在 Data Science 中的应用】
难度：中等～较难
考点：工厂模式 Factory、策略模式 Strategy、组合 Composition、sklearn 预处理封装
背景：工业故障预测系统，不同信号（振动/温度/压力）对应不同预处理；消除大量 if-elif，方便扩展。

依赖：pandas, sklearn.preprocessing.StandardScaler, MinMaxScaler

Part A：工厂模式 Factory Pattern
1. 三个预处理器类，统一接口：fit_transform(df)、transform(df)
   - StandardScalerProcessor：封装 StandardScaler，对数值列标准化
   - MinMaxProcessor：封装 MinMaxScaler，对数值列 Min‑Max 归一化
   - MissingValueProcessor：pandas fillna，默认填充值 0
   约定：fit_transform 在内部 fit；transform 只做转换，不再 fit
2. ProcessorFactory
   - 静态方法 create_processor(processor_type: str)
   - 映射关系：
       "standard" → StandardScalerProcessor
       "minmax" → MinMaxProcessor
       "missing" → MissingValueProcessor
   - 传入未知类型：抛出 ValueError

Part B：策略模式 Strategy Pattern
DataPipeline
   - __init__(processor)：接收一个预处理器实例（策略）
   - set_processor(new_processor)：运行时替换预处理策略
   - fit_transform(df)：调用当前 processor.fit_transform(df)
   - transform(df)：调用当前 processor.transform(df)
   核心：DataPipeline 本身不写预处理逻辑，只委托给外部传入的处理器；新增预处理类不用改 DataPipeline

Part C：组合 Composition
MLPipeline
   - __init__(processor, model)：持有预处理器、模型（组合，不是继承）
   - fit(X_train, y_train)：
        ① processor.fit_transform(X_train)
        ② 使用转换后的数据训练 model
   - predict(X_test)：
        ① processor.transform(X_test)
        ② 在转换后数据上调用 model.predict，返回预测结果
   约定：model 需要具备 fit(X,y)、predict(X) 接口（兼容 sklearn 模型）

输入约定
- df / X_train / X_test：pandas DataFrame
- y_train：Series / ndarray
- processor_type：字符串，限定 "standard", "minmax", "missing"

输出约定
- fit_transform / transform 返回 DataFrame
- predict 返回预测数组/Series

异常约定
- create_processor 传入非法字符串 → ValueError
- 如果预处理器未 fit 就调用 transform，交由底层 sklearn/pandas 原生异常
- 不做额外的强类型校验，依赖接口契约

验收标准
Part A
1. ProcessorFactory.create_processor("standard") 返回 StandardScalerProcessor 实例
2. ProcessorFactory.create_processor("minmax") 返回 MinMaxProcessor 实例
3. ProcessorFactory.create_processor("missing") 返回 MissingValueProcessor 实例
4. 非法类型抛 ValueError

Part B
1. DataPipeline 使用初始 processor 执行 fit_transform
2. set_processor 之后，后续调用使用新处理器
3. 不需要修改 DataPipeline 源码即可切换策略

Part C
1. MLPipeline.fit：先拟合预处理，再拟合模型
2. MLPipeline.predict：仅 transform，不再拟合预处理
3. 预处理和模型作为组件注入，实现解耦

提示
- 所有处理器保持完全一致接口，这是工厂+策略模式的前提
- sklearn scaler 默认返回 ndarray，记得转回 DataFrame，保持接口统一
- MissingValueProcessor fillna 默认 fill_value=0
"""
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


# ========= Part A 处理器 + 工厂 =========
class StandardScalerProcessor:
    def __init__(self):
        self.scaler = StandardScaler()

    def fit_transform(self, df):
        values = self.scaler.fit_transform(df)
        return pd.DataFrame(values, index=df.index, columns=df.columns)

    def transform(self, df):
        values = self.scaler.transform(df)
        return pd.DataFrame(values, index=df.index, columns=df.columns)

class MinMaxProcessor:
    def __init__(self):
        self.scaler = MinMaxScaler()

    def fit_transform(self, df):
        values = self.scaler.fit_transform(df)
        return pd.DataFrame(values, index=df.index, columns=df.columns)

    def transform(self, df):
        values = self.scaler.transform(df)
        return pd.DataFrame(values, index=df.index, columns=df.columns)

class MissingValueProcessor:
    def __init__(self, fill_value=0):
        self.fill_value = fill_value

    def fit_transform(self, df):
        return df.fillna(self.fill_value)

    def transform(self, df):
        return df.fillna(self.fill_value)

class ProcessorFactory:
    @staticmethod
    def create_processor(processor_type):
        processors = {
            "standard": StandardScalerProcessor,
            "minmax": MinMaxProcessor,
            "missing": MissingValueProcessor,
        }
        try:
            return processors[processor_type]()
        except KeyError:
            raise ValueError(f"Unknown processor type: {processor_type}") from None

# ========= Part B 策略模式：DataPipeline =========
class DataPipeline:
    def __init__(self, processor):
        self.processor = processor

    def set_processor(self, new_processor):
        self.processor = new_processor

    def fit_transform(self, df):
        return self.processor.fit_transform(df)

    def transform(self, df):
        return self.processor.transform(df)

# ========= Part C 组合：MLPipeline =========
class MLPipeline:
    def __init__(self, processor, model):
        self.processor = processor
        self.model = model

    def fit(self, X_train, y_train):
        X_transformed = self.processor.fit_transform(X_train)
        self.model.fit(X_transformed, y_train)

    def predict(self, X_test):
        X_transformed = self.processor.transform(X_test)
        return self.model.predict(X_transformed)
