import pandas as pd
import pytest

from class4_check import DataProcessor


def test_fit_transform_basic_case():
    """测试样例1：基础fit_transform，匹配预期归一结果"""
    df = pd.DataFrame({
        "temperature": [20, 30, 40, 50],
        "pressure": [100, 200, 300, 400]
    })
    processor = DataProcessor(fill_value=0, normalize=True)
    res = processor.fit_transform(df)

    assert res.loc[0, "temperature"] == pytest.approx(0.0)
    assert res.loc[1, "temperature"] == pytest.approx(1/3)
    assert res.loc[2, "temperature"] == pytest.approx(2/3)
    assert res.loc[3, "temperature"] == pytest.approx(1.0)

    assert res.loc[0, "pressure"] == pytest.approx(0.0)
    assert res.loc[1, "pressure"] == pytest.approx(1/3)
    assert res.loc[2, "pressure"] == pytest.approx(2/3)
    assert res.loc[3, "pressure"] == pytest.approx(1.0)


def test_transform_after_fit_new_data():
    """测试样例2：fit一次，transform新数据，复用统计量"""
    df_train = pd.DataFrame({
        "temperature": [20, 30, 40, 50],
        "pressure": [100, 200, 300, 400]
    })
    df_new = pd.DataFrame({
        "temperature": [25, 45],
        "pressure": [150, 350]
    })
    processor = DataProcessor(fill_value=0, normalize=True)
    processor.fit(df_train)
    res = processor.transform(df_new)

    assert res.loc[0, "temperature"] == pytest.approx(0.16666666666666666)
    assert res.loc[1, "temperature"] == pytest.approx(0.8333333333333334)
    assert res.loc[0, "pressure"] == pytest.approx(0.16666666666666666)
    assert res.loc[1, "pressure"] == pytest.approx(0.8333333333333334)


def test_transform_before_fit_raise_error():
    """未fit直接transform，必须抛异常"""
    df = pd.DataFrame({"a": [1,2]})
    processor = DataProcessor()
    with pytest.raises(RuntimeError):
        processor.transform(df)


def test_fill_missing_works():
    """缺失值被fill_value填充"""
    df = pd.DataFrame({"a": [1, None, 3]})
    processor = DataProcessor(fill_value=-99, normalize=False)
    out = processor.fill_missing(df)
    assert out["a"].iloc[1] == -99
    # 原df不变
    assert pd.isna(df["a"].iloc[1])


def test_zero_range_column_avoid_div_zero():
    """列max=min，除零保护，归一结果为0"""
    df = pd.DataFrame({"fixed": [5,5,5], "var": [1,2,3]})
    processor = DataProcessor(normalize=True)
    res = processor.fit_transform(df)
    assert all(res["fixed"] == 0)


def test_normalize_switch_off():
    """normalize=False：只填充缺失值，不做归一"""
    df = pd.DataFrame({"x": [10,20,None]})
    processor = DataProcessor(fill_value=0, normalize=False)
    res = processor.fit_transform(df)
    assert res["x"].iloc[2] == 0
    assert res["x"].iloc[0] == 10
    assert res["x"].iloc[1] == 20


def test_original_dataframe_not_modified():
    """校验所有操作不修改原始df"""
    df = pd.DataFrame({"a": [1, None, 3]})
    df_copy = df.copy(deep=True)
    processor = DataProcessor()
    processor.fit_transform(df)
    pd.testing.assert_frame_equal(df, df_copy)
