import joblib
from notebooks.utils import load_path_preprocessor_and_model

preprocessor_path, model_path = load_path_preprocessor_and_model()

preprocessor = None
try:
    preprocessor = joblib.load(preprocessor_path)
except FileNotFoundError:
    pass

model = joblib.load(model_path)

def pred(X_pred):
    if preprocessor is not None:
        X_pred = preprocessor.transform(X_pred)

    prediction = model.predict(X_pred)
    probability = model.predict_proba(X_pred)[:, 1]

    return prediction, probability
