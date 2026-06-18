import pandas as pd
import os
import joblib

from sklearn.metrics import classification_report, roc_auc_score

def load_data(file_path: str):
    df = pd.read_csv(file_path)
    return df

def transform_columns(df: pd.DataFrame):
    clean_cols = df.columns.str.strip().str.replace(' ', '_')

    # Insert underscore between lowercase letters/digits and an uppercase letter (e.g., CustomerID -> Customer_ID)
    clean_cols = clean_cols.str.replace(r'([a-z0-9])([A-Z])', r'\1_\2', regex=True)

    # Insert underscore between consecutive uppercase letters followed by lowercase (e.g., USAUser -> USA_User)
    clean_cols = clean_cols.str.replace(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', regex=True)

    # Lowercase everything and collapse any sequential underscores (e.g., customer__id -> customer_id)
    df.columns = clean_cols.str.lower().str.replace(r'_+', '_', regex=True)
    return df

def load_raw_data():
    data_descriptions = f'''{os.path.dirname(os.path.dirname(__file__))}/raw_data/data_descriptions.csv'''
    data = f'''{os.path.dirname(os.path.dirname(__file__))}/raw_data/train.csv'''
    print(data_descriptions)
    print(data)
    return data_descriptions, data

def load_path_validation_data():
    X = f'''{os.path.dirname(__file__)}/data/X.csv'''
    y = f'''{os.path.dirname(__file__)}/data/y.csv'''
    return X,y

def pickle(preprocessor, model):
    # 4. Serialize and export the pipeline artifact to your Shared Volume
    os.makedirs(os.environ["MODEL_PATH"], exist_ok=True)
    preproc_path = f'''{os.path.dirname(__file__)}/models/xgb_churn_preprocessor.pkl'''
    model_path = f'''{os.path.dirname(__file__)}/models/xgb_churn_model.pkl'''

    print(preproc_path)
    print(model_path)

    joblib.dump(preprocessor, preproc_path)
    print(f"Success! Saved pipeline preprocessing artifact to: {preproc_path}")

    joblib.dump(model, model_path)
    print(f"Success! Saved model artifact to: {model_path}")


def evaluate(preprocessor, model, X_test, y_test):
    # 3. Evaluate the model on unseen test data
    X_test  = preprocessor.transform(X_test)
    y_pred = model.predict(X_test)          # Hard classification: 0 (Stay) or 1 (Churn)
    y_proba = model.predict_proba(X_test)[:, 1] # Risk probabilities (e.g., 0.84)

    print("\n================ MODEL PERFORMANCE ================")
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_proba):.4f}")
    print("============================================= =======")
    return y_pred, y_proba
