import re

import pandas as pd
import shap


# ============================================================
# Rule-Based Explanation Engine
# ============================================================
# SHAP calculates the numerical impact of each transformed feature.
# This dictionary converts those SHAP results into human-readable
# business messages.
#
# Important:
# - Positive SHAP value means the feature increases churn risk.
# - Negative SHAP value means the feature decreases churn risk.
# - CustomerID and Churn are excluded because they are not explanatory
#   input features for the customer explanation.
# ============================================================

FEATURE_MESSAGES = {
    "AccountAge": {
        "positive": "Account age is contributing to higher churn risk.",
        "negative": "Account age is contributing to lower churn risk.",
    },
    "MonthlyCharges": {
        "positive": "Monthly charges are contributing to higher churn risk.",
        "negative": "Monthly charges are contributing to lower churn risk.",
    },
    "TotalCharges": {
        "positive": "Total charges are contributing to higher churn risk.",
        "negative": "Total charges are contributing to lower churn risk.",
    },
    "SubscriptionType": {
        "positive": "Subscription type is contributing to higher churn risk.",
        "negative": "Subscription type is contributing to lower churn risk.",
    },
    "PaymentMethod": {
        "positive": "Payment method is contributing to higher churn risk.",
        "negative": "Payment method is contributing to lower churn risk.",
    },
    "PaperlessBilling": {
        "positive": "Paperless billing is contributing to higher churn risk.",
        "negative": "Paperless billing is contributing to lower churn risk.",
    },
    "ContentType": {
        "positive": "Preferred content type is contributing to higher churn risk.",
        "negative": "Preferred content type is contributing to lower churn risk.",
    },
    "MultiDeviceAccess": {
        "positive": "Multi-device access is contributing to higher churn risk.",
        "negative": "Multi-device access is contributing to lower churn risk.",
    },
    "DeviceRegistered": {
        "positive": "Registered device type is contributing to higher churn risk.",
        "negative": "Registered device type is contributing to lower churn risk.",
    },
    "ViewingHoursPerWeek": {
        "positive": "Weekly viewing activity is contributing to higher churn risk.",
        "negative": "Weekly viewing activity is contributing to lower churn risk.",
    },
    "AverageViewingDuration": {
        "positive": "Average viewing duration is contributing to higher churn risk.",
        "negative": "Average viewing duration is contributing to lower churn risk.",
    },
    "ContentDownloadsPerMonth": {
        "positive": "Content download activity is contributing to higher churn risk.",
        "negative": "Content download activity is contributing to lower churn risk.",
    },
    "GenrePreference": {
        "positive": "Genre preference is contributing to higher churn risk.",
        "negative": "Genre preference is contributing to lower churn risk.",
    },
    "UserRating": {
        "positive": "User rating is contributing to higher churn risk.",
        "negative": "User rating is contributing to lower churn risk.",
    },
    "SupportTicketsPerMonth": {
        "positive": "Support tickets are contributing to higher churn risk.",
        "negative": "Support tickets are contributing to lower churn risk.",
    },
    "Gender": {
        "positive": "Gender is contributing to higher churn risk.",
        "negative": "Gender is contributing to lower churn risk.",
    },
    "WatchlistSize": {
        "positive": "Watchlist size is contributing to higher churn risk.",
        "negative": "Watchlist size is contributing to lower churn risk.",
    },
    "ParentalControl": {
        "positive": "Parental control setting is contributing to higher churn risk.",
        "negative": "Parental control setting is contributing to lower churn risk.",
    },
    "SubtitlesEnabled": {
        "positive": "Subtitle setting is contributing to higher churn risk.",
        "negative": "Subtitle setting is contributing to lower churn risk.",
    },
}


def get_original_feature_name(feature: str) -> str:
    """
    Convert transformed feature names back to original dataset feature names.

    The preprocessor creates transformed feature names such as:
    - num__AverageViewingDuration
    - cat__SubscriptionType_Basic
    - cat__PaymentMethod_Electronic check

    For the frontend and explanation messages, we want to map them back to:
    - AverageViewingDuration
    - SubscriptionType
    - PaymentMethod
    """

    feature = feature.replace("num__", "").replace("cat__", "")

    for original_feature in FEATURE_MESSAGES.keys():
        if feature == original_feature or feature.startswith(f"{original_feature}_"):
            return original_feature

    return feature


def make_display_name(feature: str) -> str:
    """
    Convert technical feature names into clean display names.

    Example:
    - AverageViewingDuration -> Average Viewing Duration
    - SupportTicketsPerMonth -> Support Tickets Per Month
    """

    feature = get_original_feature_name(feature)
    feature = re.sub(r"(?<!^)(?=[A-Z])", " ", feature)

    return feature.strip()


def add_rule_based_messages(top_drivers):
    """
    Add human-readable explanation messages to SHAP top drivers.

    SHAP only tells us the numerical impact of each feature.
    This function adds business-friendly messages based on direction:

    - shap_value > 0: feature increases churn risk
    - shap_value < 0: feature decreases churn risk
    """

    explained_drivers = []

    for driver in top_drivers:
        raw_feature = driver["feature"]
        shap_value = driver["shap_value"]

        original_feature = get_original_feature_name(raw_feature)
        display_feature = make_display_name(original_feature)

        direction = "positive" if shap_value > 0 else "negative"

        message = FEATURE_MESSAGES.get(
            original_feature,
            {
                "positive": f"{display_feature} is contributing to higher churn risk.",
                "negative": f"{display_feature} is contributing to lower churn risk.",
            },
        )[direction]

        explained_drivers.append(
            {
                **driver,
               # "raw_feature": raw_feature,
                "feature": display_feature,
              #  "original_feature": original_feature,
                "message": message,
            }
        )

    return explained_drivers


# ============================================================
# SHAP Explainer
# ============================================================
# SHAP is not a separate model.
# It explains the existing Logistic Regression pipeline.
#
# The pipeline contains:
# - preprocessor: transforms raw customer data
# - model: Logistic Regression model
#
# SHAP needs transformed data, so we use the same preprocessor
# that was used during model training.
#
# The background dataset represents the baseline customer population.
# SHAP compares each new customer against this background to estimate
# which features push churn risk up or down.
# ============================================================


def build_log_reg_explainer(pipeline, X_background, background_size=500):
    """
    Build a SHAP explainer for the Logistic Regression model.

    Steps:
    1. Extract the preprocessor and model from the saved pipeline.
    2. Transform the background data using the same preprocessor.
    3. Use transformed background data as SHAP baseline.
    4. Return the explainer and transformed feature names.
    """

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
    """
    Generate prediction and SHAP explanation for one customer.

    Steps:
    1. Transform customer input using the saved preprocessor.
    2. Calculate SHAP values using the SHAP explainer.
    3. Rank features by absolute SHAP impact.
    4. Keep the top N most important drivers.
    5. Add human-readable rule-based messages.
    6. Return churn prediction, probability, and top drivers.
    """

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
        shap_df.sort_values("abs_shap", ascending=False)
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
