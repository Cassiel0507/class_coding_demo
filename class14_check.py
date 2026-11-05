"""
【题目 14：Multiple Inheritance——多重继承与 MRO】
难度：中等
考点：Python 多重继承、Mixin、MRO、super()、实例状态隔离

背景：
工业机器学习系统，把模型基础能力、日志、指标统计拆为独立类，通过多重继承组合。

类定义清单：
1. BaseModel
    属性
        model_name：str，模型名称
        is_trained：bool，训练标记，初始 False
    方法
        fit()：设置 self.is_trained = True
        get_model_name()：返回 self.model_name

2. LoggerMixin
    属性
        logs：list，保存日志字符串，每个实例独立
    方法
        log(message: str)：追加消息到 logs
        get_logs()：返回 logs 列表

3. MetricsMixin
    属性
        metrics：dict，key=指标名，value=指标值，每个实例独立
    方法
        add_metric(name: str, value)：写入指标
        get_metrics()：返回指标字典

4. IndustrialModel(BaseModel, LoggerMixin, MetricsMixin)
    - 需要实现 __init__，正确初始化全部父类需要的实例变量
    - 保证不同实例之间 logs、metrics 完全隔离，不能共享类变量

输入约定：
- IndustrialModel 构造入参：model_name（字符串）
- log 的 message：字符串
- add_metric：name 字符串，value 任意数值

输出约定：
- get_model_name()：str
- get_logs()：list[str]
- get_metrics()：dict
- fit() 修改 is_trained 为 True

异常约定：
- 不强制主动抛异常；如果传入非字符串日志/指标名，直接交由 Python 默认行为
- 重点风险：不要把 logs / metrics 定义在类层面（会多实例串数据）

验收标准（全部必须通过）
1. model = IndustrialModel("RandomForest")
   model.fit() → is_trained == True
   model.log("Model training started")
   model.log("Model training completed")
   model.add_metric("accuracy", 0.95)
   model.add_metric("recall", 0.92)
   get_model_name() == "RandomForest"
   get_logs() == ["Model training started", "Model training completed"]
   get_metrics() == {"accuracy":0.95, "recall":0.92}

2. 多实例隔离：
   model2 = IndustrialModel("LogisticRegression")
   model2.log("New experiment")
   model.get_logs() 不能包含 "New experiment"
   model2.get_logs() 只有 ["New experiment"]

3. 可以打印 IndustrialModel.__mro__ 观察方法解析顺序

提示：
Mixin 类一般用 super 配合；所有可变容器（list/dict）必须在实例 __init__ 里面初始化，不能放在类顶层。
"""

class BaseModel:
    def __init__(self, model_name):
        self.model_name = model_name
        self.is_trained = False

    def fit(self):
        self.is_trained = True

    def get_model_name(self):
        return self.model_name

class LoggerMixin:
    def __init__(self):
        self.logs = []

    def log(self, message):
        self.logs.append(message)

    def get_logs(self):
        return self.logs

class MetricsMixin:
    def __init__(self):
        self.metrics = {}

    def add_metric(self, name, value):
        self.metrics[name] = value

    def get_metrics(self):
        return self.metrics

class IndustrialModel(BaseModel, LoggerMixin, MetricsMixin):
    def __init__(self, model_name):
        # 各父类在实例中初始化自己的状态，避免可变数据被实例共享。
        BaseModel.__init__(self, model_name)
        LoggerMixin.__init__(self)
        MetricsMixin.__init__(self)
