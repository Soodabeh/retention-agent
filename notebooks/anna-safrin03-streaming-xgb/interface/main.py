import os
import joblib

model = joblib.load(os.environ["MODEL_PATH"])


def pred(X_pred):
    preprocessor = joblib.load("../models/xgb_churn_preproc_pipeline.pkl")
    X_pred = preprocessor.transform(X_pred)
    prediction = model.predict(X_pred)
    probability = model.predict_proba(X_pred)[:, 1]

    return prediction, probability


#model = joblib.load("models/model.pkl")

#def pred(X_pred):
    X_processed = preprocessor.transform(X_pred)

    prediction = model.predict(X_processed)
    probability = model.predict_proba(X_processed)[:, 1]

    return prediction, probability
