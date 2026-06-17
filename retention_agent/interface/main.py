import joblib

MODEL_PATH = "notebooks/models/log_reg_pipeline_soodabeh.pkl"

model = joblib.load(MODEL_PATH)


def pred(X_pred):
    prediction = model.predict(X_pred)
    probability = model.predict_proba(X_pred)[:, 1]

    return prediction, probability
