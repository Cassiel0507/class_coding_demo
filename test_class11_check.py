from class11_check import BaseModel, RandomForestModel, LogisticRegressionModel

def test_rf_model_name():
    rf = RandomForestModel(n_estimators=100, max_depth=5)
    assert rf.get_model_name() == "RandomForest"

def test_lr_model_name():
    lr = LogisticRegressionModel(C=1.0, max_iter=200)
    assert lr.get_model_name() == "LogisticRegression"

def test_initial_train_status():
    rf = RandomForestModel(100, 5)
    lr = LogisticRegressionModel(1.0, 200)
    assert rf.get_status() is False
    assert lr.get_status() is False

def test_fit_set_trained():
    rf = RandomForestModel(100, 5)
    rf.fit()
    assert rf.get_status() is True

    lr = LogisticRegressionModel(1.0, 200)
    lr.fit()
    assert lr.get_status() is True

def test_predict_output():
    rf = RandomForestModel(100, 5)
    lr = LogisticRegressionModel(1.0, 200)
    assert rf.predict() == [0, 1, 0]
    assert lr.predict() == [1, 0, 1]

def test_inheritance_check():
    rf = RandomForestModel(100, 5)
    lr = LogisticRegressionModel(1.0, 200)
    assert isinstance(rf, BaseModel)
    assert isinstance(lr, BaseModel)

def test_base_set_trained():
    base = BaseModel("test_model")
    assert base.get_status() is False
    base.set_trained()
    assert base.get_status() is True
