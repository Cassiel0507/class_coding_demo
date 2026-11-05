"""
【题目 15：Metaclass——理解 Python 元类】
难度：较难
考点：元类、type、__new__、类创建钩子、类注册表
说明：元类属于 Python 高级特性，DS 面试一般不要求手写，但要理解用途。

背景：
模型注册系统，自动收集所有模型子类；每定义一个新模型类，自动登记到注册表，基类本身不注册。

任务清单
1. 元类 ModelRegistryMeta，继承 type
    - 类属性：registry，字典，key：类名字符串，value：类对象
    - 重写 __new__(mcs, name, bases, namespace)
    - 逻辑：创建类之后，只有当新类是 BaseModel 的子类，并且不等于 BaseModel 本身，才加入 registry
2. BaseModel，指定 metaclass=ModelRegistryMeta
3. RandomForestModel、LogisticRegressionModel：继承 BaseModel，不需要额外写方法

输入约定
- 定义子类时由 Python 自动触发元类 __new__，不需要手动传参
- registry 挂载在元类 ModelRegistryMeta 上

输出约定
- ModelRegistryMeta.registry：{类名: 类对象}
- BaseModel 本身不出现在注册表里面
- 通过注册表取出类，可以正常实例化

异常约定
- 不主动抛异常；允许重复定义同名类（后定义覆盖前面）
- 不处理动态删除类的场景

验收标准（全部必须通过）
1. 定义 BaseModel(metaclass=ModelRegistryMeta)
2. 定义 RandomForestModel(BaseModel)、LogisticRegressionModel(BaseModel)
3. ModelRegistryMeta.registry 等于：
   {
       "RandomForestModel": RandomForestModel,
       "LogisticRegressionModel": LogisticRegressionModel
   }
4. BaseModel 不在 registry 的 key 里面
5. 从注册表取出 RandomForestModel，实例化，type(model).__name__ == "RandomForestModel"
6. 新增更多 BaseModel 子类，会自动追加进 registry

提示
- __new__ 负责创建类对象；mcs 就是元类本身
- 判断逻辑：issubclass(新类, BaseModel) and name != "BaseModel"
- 不要把 registry 放在 BaseModel 上，要求放在元类上
"""
from typing import ClassVar


class ModelRegistryMeta(type):
    registry: ClassVar[dict] = {}  # 👈 Ruff 会自动修改这行，明确告诉工具这是“类变量”

    def __new__(mcs, name, bases, namespace):
        cls = super().__new__(mcs, name, bases, namespace)
        # 只登记 BaseModel 的子类；相同类名会由新类覆盖。
        if name != "BaseModel" and issubclass(cls, BaseModel):
            mcs.registry[name] = cls
        return cls

class BaseModel(metaclass=ModelRegistryMeta):
    pass

class RandomForestModel(BaseModel):
    pass

class LogisticRegressionModel(BaseModel):
    pass
