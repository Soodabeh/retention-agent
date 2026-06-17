Training & Serialization Pipeline

```mermaid
graph TD
    %% Base Styling
    classDef data fill:#e1f5fe,stroke:#0288d1,stroke-width:2px,color:#01579b;
    classDef process fill:#fff3e0,stroke:#f57c00,stroke-width:2px,color:#e65100;
    classDef model fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px,color:#4a148c;
    classDef storage fill:#efebe9,stroke:#5d4037,stroke-width:2px,color:#3e2723;

    %% Data Ingestion
    A[/"Raw Data:<br>customer_churn.csv"/]:::data --> B("1. transform_columns()<br>(Strip spaces, lowercase, snake_case)"):::process
    B --> C("2. Split X & y<br>(Drop 'churn' and 'customer_id')"):::process

    %% Splitting Subgraph
    subgraph Splitting ["3. Stratified 60/20/20 Split (Avoid Data Leakage)"]
        C --> D1["Train Set<br>(60%)"]:::data
        C --> D2["Validation Set<br>(20%)"]:::data
        C --> D3["Test Set<br>(20%)"]:::data
    end

    %% Preprocessing
    P_Def["Define ColumnTransformer<br>(Imputers, Scaler, OHE)"]:::process

    D1 & P_Def --> E1("4. preprocessor.fit_transform()"):::process
    E1 --> F1["X_train_processed"]:::data

    D2 & P_Def --> E2("5. preprocessor.transform()<br>(Transform Only)"):::process
    E2 --> F2["X_val_processed"]:::data

    D3 & P_Def --> E3("6. preprocessor.transform()<br>(Transform Only)"):::process
    E3 --> F3["X_test_processed"]:::data

    %% Model Training
    F1 --> G("7. XGBClassifier.fit()"):::model
    F2 -->|eval_set<br>Early Stopping| G
    y_train_label["y_train"]:::data --> G

    G --> H["Trained XGBoost Model"]:::model

    %% Serialization
    subgraph Serialization ["8. Save Artifacts to Docker Volume (../models/)"]
        H --> M1[("standalone_xgb_model.pkl")]:::storage
        E1 -->|Extract Fitted Preprocessor| M2[ Blitz Setup <br> 💾 xgb_churn_preproc_pipeline.pkl ]:::storage

        K("Bundle Preprocessor + Model<br>into Unified Pipeline"):::model --> M3[("xgb_churn_pipeline.pkl<br>(Production Grade)")]:::storage
        H & E1 --> K
    end


%%{init: {
  'theme': 'base',
  'themeVariables': {
    'background': '#ffffff',
    'primaryColor': '#e1f5fe',
    'primaryTextColor': '#1a1a1a',
    'lineColor': '#0288d1',
    'nodeBorder': '#0288d1',
    'tertiaryColor': '#ffffff',
    'mainBkg': '#ffffff',
    'edgeLabelBackground': '#ffffff'
  }
}}%%
```

Evaluation & Inference Pipeline
```mermaid
graph TD
    %% Base Styling
    classDef data fill:#e1f5fe,stroke:#0288d1,stroke-width:2px,color:#01579b;
    classDef process fill:#fff3e0,stroke:#f57c00,stroke-width:2px,color:#e65100;
    classDef storage fill:#efebe9,stroke:#5d4037,stroke-width:2px,color:#3e2723;
    classDef approachA fill:#e8f5e9,stroke:#388e3c,stroke-width:2px,color:#1b5e20;
    classDef approachB fill:#fffde7,stroke:#fbc02d,stroke-width:2px,color:#f57f17;

    %% Raw Input
    A[/"Fresh Evaluation Data:<br>validation_data.csv"/]:::data --> B("1. transform_columns()<br>(Must match Notebook 1 string modifications)"):::process
    B --> C("2. Extract X & y<br>(Drop 'churn' and 'customer_id')"):::process
    C --> X_raw["X_val_raw"]:::data
    C --> y_raw["y_val_raw<br>(Ground Truth Labels)"]:::data

    %% Branching Logic
    X_raw --> Choice{"Choose Processing<br>& Prediction Strategy"}

    %% Strategy A
    subgraph Strategy_A ["Approach A: The Production Way"]
        Choice -->|Pass Raw Columns| M3[("xgb_churn_pipeline.pkl")]:::storage
        M3 --> Run_A("pipeline.predict(X_val_raw)<br>(Internal transformation & scaling)"):::approachA
        Run_A --> Y_Pred_A["y_pred / y_prob"]:::data
    end

    %% Strategy B
    subgraph Strategy_B ["Approach B: The Modular Way (Debugging)"]
        Choice -->|Pass to Separated Picker| M2[("xgb_churn_preproc_pipeline.pkl")]:::storage
        M2 --> Run_B1("preprocessor.transform(X_val_raw)<br>(Strictly Transform Only!)"):::approachB
        Run_B1 --> X_proc["X_val_processed"]:::data

        X_proc & M1[("standalone_xgb_model.pkl")]:::storage --> Run_B2("model.predict(X_val_processed)"):::approachB
        Run_B2 --> Y_Pred_B["y_pred / y_prob"]:::data
    end

    %% Evaluation
    Y_Pred_A & y_raw --> Eval("classification_report()"):::process
    Y_Pred_B & y_raw --> Eval
    Eval --> Final_Output[\"Generate Metrics:<br>Precision, Recall, F1, AUC-ROC"/]:::data
    ```
