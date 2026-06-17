import joblib
from retention_agent.params import MODEL_PATH

model = joblib.load(MODEL_PATH)


def pred(X_pred):
    prediction = model.predict(X_pred)
    probability = model.predict_proba(X_pred)[:, 1]

    return prediction, probability


#preprocessor = joblib.load("models/preprocessor.pkl")
#model = joblib.load("models/model.pkl")

#def pred(X_pred):
    X_processed = preprocessor.transform(X_pred)

    prediction = model.predict(X_processed)
    probability = model.predict_proba(X_processed)[:, 1]

    return prediction, probability
