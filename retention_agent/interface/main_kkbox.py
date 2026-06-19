import json
import joblib
import numpy as np
import scipy.sparse as sp
import shap

from notebooks.utils import load_path_kkbox_artifacts


# === Module load: runs once at FastAPI startup ===

preprocessor_path, model_path, threshold_path = load_path_kkbox_artifacts()

preprocessor = joblib.load(preprocessor_path)
model = joblib.load(model_path)

# Loud failure if threshold file missing — by design (a wrong-threshold
# silent-default would degrade predictions invisibly).
with open(threshold_path) as f:
    THRESHOLD = float(json.load(f)["threshold"])

# Encoded feature names like "num__bd_clean", "cat__city_5" — the columns
# produced by ColumnTransformer at training time.
feature_names = preprocessor.get_feature_names_out()

# The 4 original categorical features, mirroring training notebook cell 29.
# Needed to map one-hot column names back to their original feature.
CATEGORICAL_FEATURES = [
    "city",
    "gender_filled",
    "registered_via",
    "latest_payment_method_id",
]


def _encoded_to_original(encoded_name):
    """Map one ColumnTransformer output column name to its original input feature."""
    if encoded_name.startswith("num__"):
        return encoded_name[len("num__"):]
    if encoded_name.startswith("cat__"):
        rest = encoded_name[len("cat__"):]
        for cat in CATEGORICAL_FEATURES:
            if rest.startswith(cat + "_") or rest == cat:
                return cat
    return encoded_name  # fallback — shouldn't trigger with this preprocessor


# Pre-compute the encoded → original map once.
ENCODED_TO_ORIGINAL = {name: _encoded_to_original(name) for name in feature_names}

# SHAP explainer — built at startup, NOT loaded from pkl.
# TreeExplainer is fast to construct on a fitted tree model, and pickling
# it would couple the artifact to a specific shap library version.
explainer = shap.TreeExplainer(model)


def pred_kkbox(X_pred):
    """
    Predict KKBox churn with SHAP-based per-customer driver explanation.

    X_pred: pandas.DataFrame with the 18 KKBox feature columns (one row
            expected from the current /predict_kkbox route).

    Returns (prediction, probability, top_drivers):
      - prediction: np.ndarray shape (n,), int (1 = will churn, 0 = will stay)
      - probability: np.ndarray shape (n,), float (positive-class probability)
      - top_drivers: list of 5 {"feature", "shap_value", "direction"} dicts,
                     sorted by absolute impact descending — computed for row 0.
    """
    # 1. Preprocess (output may be sparse).
    X_t = preprocessor.transform(X_pred)

    # 2. Probability + thresholded binary prediction.
    # We apply OUR tuned threshold, NOT model.predict() (which uses 0.5).
    probability = model.predict_proba(X_t)[:, 1]
    prediction = (probability >= THRESHOLD).astype(int)

    # 3. SHAP values for the positive class.
    shap_vals = explainer.shap_values(X_t)
    # LGBM + older shap returns [class0, class1]; newer returns a single array.
    if isinstance(shap_vals, list):
        shap_vals_pos = shap_vals[1]
    else:
        shap_vals_pos = shap_vals
    # Densify defensively (cell 75 of the training notebook hit this).
    if sp.issparse(shap_vals_pos):
        shap_vals_pos = shap_vals_pos.toarray()
    shap_row = np.asarray(shap_vals_pos[0]).ravel()

    # 4. Aggregate one-hot SHAP back to the original feature (signed sum).
    aggregated = {}
    for i, encoded_name in enumerate(feature_names):
        orig = ENCODED_TO_ORIGINAL[encoded_name]
        aggregated[orig] = aggregated.get(orig, 0.0) + float(shap_row[i])

    # 5. Sort by absolute impact, take top 5, format for the API response.
    sorted_features = sorted(
        aggregated.items(), key=lambda kv: abs(kv[1]), reverse=True
    )
    top_drivers = [
        {
            "feature": feat,
            "shap_value": shap_val,
            "direction": "increases" if shap_val > 0 else "decreases",
        }
        for feat, shap_val in sorted_features[:5]
    ]

    return prediction, probability, top_drivers