"""
题目 1：创建一个 Dataset 类｜难度：简单
背景：工业数据集管理类

【需求清单】
实现 Dataset 类
1. __init__(self, name, data, description="")
    参数：
        name: str，数据集名称
        data: list，数据集数值列表，列表内元素应为可求平均值的数字（int/float）
        description: str，数据集描述，可选，默认空字符串
    属性：绑定 name / data / description
    约束：
        - data 必须是 list 类型；如果不是 list，后续可以抛出 TypeError
        - data 内部元素允许为空列表（空数据集）
2. get_size(self)
    输入：无
    返回：int，data 里面样本数量（len(self.data)）
3. get_mean(self)
    输入：无
    返回：float，数据集平均值
    异常规则：如果 data 是空列表，抛出 ValueError("数据集为空，无法计算均值")
4. add_sample(self, value)
    输入：value（int/float）
    行为：把 value 追加到 self.data
    约束：value 必须是数字类型，否则抛 TypeError("样本必须是数字")
5. get_summary(self)
    输入：无
    返回：dict，结构固定：
        {
            "name": 数据集名称,
            "description": 描述文本,
            "size": 样本数量,
            "mean": 均值
        }

【验收标准】
✅ 正常数据集：可以正确返回 size、mean、summary
✅ 空数据集调用 get_mean() 抛出指定 ValueError
✅ add_sample 成功追加数字，加非数字抛 TypeError
✅ data 传非list类型初始化时抛 TypeError
✅ summary字典key固定不变
"""
# 下面在这里写你的 Dataset 类代码
class Dataset:
    def __init__(self, name, data, description=""):
        if not isinstance(data, list):
            raise TypeError("data 必须是 list 类型")
        if any(not isinstance(value, (int, float)) for value in data):
            raise TypeError("样本必须是数字")

        self.name = name
        self.data = data
        self.description = description

    def get_size(self):
        return len(self.data)

    def get_mean(self):
        # 空数据集没有可计算的均值。
        if not self.data:
            raise ValueError("数据集为空，无法计算均值")
        return sum(self.data) / len(self.data)

    def add_sample(self, value):
        if not isinstance(value, (int, float)):
            raise TypeError("样本必须是数字")
        self.data.append(value)

    def get_summary(self):
        return {
            "name": self.name,
            "description": self.description,
            "size": self.get_size(),
            "mean": self.get_mean(),
        }
