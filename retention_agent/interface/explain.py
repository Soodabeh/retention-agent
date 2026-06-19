import pandas as pd
import shap


#
# Rule-Based Explanation Engine
# only for 10 features not all.

FEATURE_MESSAGES = {
    "num__AverageViewingDuration": {
        "positive": "Viewing session duration is contributing to higher churn risk.",
        "negative": "Viewing session duration is contributing to lower churn risk.",
    },
    "num__AccountAge": {
        "positive": "Account age is contributing to higher churn risk.",
        "negative": "Account age is contributing to lower churn risk.",
    },
    "num__ContentDownloadsPerMonth": {
        "positive": "Content download activity is contributing to higher churn risk.",
        "negative": "Content download activity is contributing to lower churn risk.",
    },
    "num__MonthlyCharges": {
        "positive": "Monthly charges are contributing to higher churn risk.",
        "negative": "Monthly charges are contributing to lower churn risk.",
    },
    "num__ViewingHoursPerWeek": {
        "positive": "Weekly viewing activity is contributing to higher churn risk.",
        "negative": "Weekly viewing activity is contributing to lower churn risk.",
    },
    "cat__SubscriptionType_Basic": {
        "positive": "Basic subscription plan is contributing to higher churn risk.",
        "negative": "Basic subscription plan is contributing to lower churn risk.",
    },
    "cat__SubscriptionType_Premium": {
        "positive": "Premium subscription plan is contributing to higher churn risk.",
        "negative": "Premium subscription plan is contributing to lower churn risk.",
    },
    "cat__PaymentMethod_Credit card": {
        "positive": "Credit card payment method is contributing to higher churn risk.",
        "negative": "Credit card payment method is contributing to lower churn risk.",
    },
    "cat__PaymentMethod_Electronic check": {
        "positive": "Electronic check payment method is contributing to higher churn risk.",
        "negative": "Electronic check payment method is contributing to lower churn risk.",
    },
    "cat__Contract_Month-to-month": {
        "positive": "Month-to-month contract is contributing to higher churn risk.",
        "negative": "Month-to-month contract is contributing to lower churn risk.",
    },
}


def add_rule_based_messages(top_drivers):
    explained_drivers = []

    for driver in top_drivers:
        feature = driver["feature"]
        shap_value = driver["shap_value"]

        direction = "positive" if shap_value > 0 else "negative"

        message = FEATURE_MESSAGES.get(
            feature,
            {
                "positive": f"{feature} is contributing to higher churn risk.",
                "negative": f"{feature} is contributing to lower churn risk.",
            },
        )[direction]

        explained_drivers.append(
            {
                **driver,
                "message": message,
            }
        )

    return explained_drivers



# SHAP Explainer


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
    top_n=5,
):
    X_customer_t = preprocessor.transform(X_customer)

    shap_values = explainer(X_customer_t)
    customer_shap_values = shap_values.values[0]

    shap_df = pd.DataFrame(
        {
            "feature": feature_names,
            "shap_value": customer_shap_values,
        }
    )

    shap_df["abs_shap"] = shap_df["shap_value"].abs()

    top_drivers = (
        shap_df
        .sort_values("abs_shap", ascending=False)
        .head(top_n)
        .drop(columns="abs_shap")
        .to_dict(orient="records")
    )

    top_drivers = add_rule_based_messages(top_drivers)

    churn_probability = pipeline.predict_proba(X_customer)[0, 1]
    churn_prediction = pipeline.predict(X_customer)[0]

    return {
        "churn_prediction": int(churn_prediction),
        "churn_probability": float(churn_probability),
        "top_drivers": top_drivers,
    }
