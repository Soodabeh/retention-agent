import os
import joblib

model = joblib.load(os.environ["MODEL_PATH"]+ os.environ["MODEL_NAME_XGB"])
preprocessor = joblib.load(os.environ["MODEL_PATH"]+os.environ["MODEL_NAME_PREPROCESSOR"])

def pred(X_pred):

    X_pred = preprocessor.transform(X_pred)
    prediction = model.predict(X_pred)
    probability = model.predict_proba(X_pred)[:, 1]

    return prediction, probability
