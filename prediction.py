import joblib as jlb

def predict(sepal_L, sepal_W, petal_L, petal_W):
    clf = jlb.load("rf_model.sav")
    input_data = [[sepal_L, sepal_W, petal_L, petal_W]]
    prediction = clf.predict(input_data)
    print("joblib",jlb.__version__)
    return prediction
