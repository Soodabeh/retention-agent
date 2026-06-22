import joblib
from notebooks.utils import ( load_path_preprocessor_and_model,load_path_shap_background)
from retention_agent.interface.explain import (build_log_reg_explainer,explain_customer)

preprocessor_path, model_path = load_path_preprocessor_and_model()

preprocessor = None
try:
    preprocessor = joblib.load(preprocessor_path)
except FileNotFoundError:
    pass

model = joblib.load(model_path)

#for SHAP

shap_background_path = load_path_shap_background()
shap_background = joblib.load(shap_background_path)

shap_preprocessor, explainer, feature_names = build_log_reg_explainer(
    pipeline=model,
    X_background=shap_background,
)

def pred(X_pred):
    if preprocessor is not None:
        X_pred = preprocessor.transform(X_pred)

    prediction = model.predict(X_pred)
    probability = model.predict_proba(X_pred)[:, 1]

    return prediction, probability

def explain(X_pred, top_n=5):
    return explain_customer(
        pipeline=model,
        preprocessor=shap_preprocessor,
        explainer=explainer,
        feature_names=feature_names,
        X_customer=X_pred,
        top_n=top_n,
    )
