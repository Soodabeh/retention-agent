import joblib
import os
from notebooks.utils import load_path_preprocessor_and_model

preprocessor_path, model_path = load_path_preprocessor_and_model()
preprocessor = joblib.load(preprocessor_path)
model = joblib.load(model_path)

def pred(X_pred):
    X_pred = preprocessor.transform(X_pred)
    prediction = model.predict(X_pred)
    probability = model.predict_proba(X_pred)[:, 1]

    return prediction, probability
