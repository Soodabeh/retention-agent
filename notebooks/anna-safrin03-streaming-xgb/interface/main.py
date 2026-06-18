import os
import joblib

model = joblib.load(os.environ["MODEL_PATH"]+ 'xgb_churn_model.pkl')
preprocessor = joblib.load(os.environ["MODEL_PATH"]+'xgb_churn_preproc_pipeline.pkl')

def pred(X_pred):

    X_pred = preprocessor.transform(X_pred)
    prediction = model.predict(X_pred)
    probability = model.predict_proba(X_pred)[:, 1]

    return prediction, probability
