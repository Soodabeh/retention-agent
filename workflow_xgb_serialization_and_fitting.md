```mermaid
graph TD
    %% Base Styling
    classDef data fill:#e1f5fe,stroke:#0288d1,stroke-width:2px;
    classDef process fill:#fff3e0,stroke:#f57c00,stroke-width:2px;
    classDef model fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px;
    classDef storage fill:#efebe9,stroke:#5d4037,stroke-width:2px;

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
    ```
