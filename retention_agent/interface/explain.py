import shap
import pandas as pd


def build_log_reg_explainer(pipeline, X_background, background_size=500):
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    X_background_t = preprocessor.transform(X_background)
    background_data = X_background_t[:background_size]

    explainer = shap.Explainer(model, background_data)
    feature_names = preprocessor.get_feature_names_out()

    return preprocessor, explainer, feature_names


def explain_customer(
    pipeline,
    preprocessor,
    explainer,
    feature_names,
    X_customer,
    top_n=5
):
    X_customer_t = preprocessor.transform(X_customer)

    shap_values = explainer(X_customer_t)
    customer_shap_values = shap_values.values[0]

    shap_df = pd.DataFrame({
        "feature": feature_names,
        "shap_value": customer_shap_values
    })

    shap_df["abs_shap"] = shap_df["shap_value"].abs()

    top_drivers = (
        shap_df
        .sort_values("abs_shap", ascending=False)
        .head(top_n)
        .drop(columns="abs_shap")
        .to_dict(orient="records")
    )

    churn_probability = pipeline.predict_proba(X_customer)[0, 1]
    churn_prediction = pipeline.predict(X_customer)[0]

    return {
        "churn_prediction": int(churn_prediction),
        "churn_probability": float(churn_probability),
        "top_drivers": top_drivers
    }
