### Data Pipeline & Architecture

# AI Customer Retention Intelligence

## Overview
The objective of this project is to develop a customer intelligence platform driven by artificial intelligence. This platform is designed to flag customers who are at risk of leaving and provide clear explanations for their potential churn. By combining machine learning, customer segmentation, and AI-driven insights, the solution predicts customer attrition and clarifies the core drivers behind it.


## Problem Statement
Subscription-based businesses face the daily challenge of customer loss. Product and business teams frequently struggle to determine:
* Which specific users are most likely to cancel their subscriptions?
* What distinct customer profiles or segments exist within the user base?
* What specific actions or behaviors signal an impending churn?
* Which users should the business prioritize when launching retention campaigns?


## Data Sources
This project is built on two primary datasets to model and analyze user behavior:

**1. Customer Churn Predictive Analytics Dataset**
* **Volume:** Roughly 243,000 user records.
* **Attributes:** 20 distinct features covering user profiles, billing and revenue, product engagement, and overall experience.
* **Target Variable:** A binary churn indicator (Yes / No) with an overall churn rate of 18%.
* **Data Integrity:** Clean dataset with no missing values.

**2. Olist Brazilian E-Commerce Public Dataset**
* **Volume:** 99,000 orders distributed across 9 relational tables.
* **Nature:** Authentic commercial transaction data capturing the complete customer lifecycle, from initial purchase to final review, spanning three years.
* **Target Variable:** A binary churn target needs to be engineered from the transactional history.
* **Application:** Highly suitable for cohort analysis, Customer Lifetime Value (CLV) calculations, segmentation, and churn modeling.

```mermaid
graph TD
    subgraph Data Sources
        A[Kaggle Churn Dataset<br/>~243,000 records<br/>20 features]
        B[Olist E-Commerce Dataset<br/>99,000 orders<br/>9 relational tables]
    end

    subgraph Data Processing & Engineering
        C[Data Cleaning & Merging<br/>Handling missing values & joins]
        D[Feature Engineering<br/>Profile, Revenue, Engagement, Experience]
    end

    subgraph Machine Learning Modeling
        E[Unsupervised Learning<br/>KMeans Clustering]
        F[Supervised Learning<br/>LogReg, Random Forest, XGBoost]
    end

    subgraph Outputs & AI Insights
        G[Customer Segments<br/>Power, Loyal, At-Risk, Low Engagement]
        H[Churn Predictions<br/>Probability %]
        I[AI Intelligence Agent<br/>Natural Language Explanations]
    end

    %% Define the data flow
    A --> C
    B --> C
    C --> D

    D --> E
    D --> F

    E --> G
    F --> H

    G --> I
    H --> I
```


## Methodology

### 1. Customer Segmentation (Unsupervised Learning)
* **Algorithm:** KMeans Clustering.
* **Objective:** Identify and group users based on shared behavioral traits.
* **Target Segments:** Power Users, Loyal Customers, Low Engagement Users, and At-Risk Customers.

### 2. Churn Prediction Pipeline (Supervised Learning)
* **Input Data:** Subscription details, behavioral logs, satisfaction scores, and revenue metrics.
* **Target:** Predicting the likelihood of churn.
* **Evaluated Models:** Logistic Regression, Random Forest, and XGBoost.

### 3. Explainability and AI Integration
* **Model Interpretation:** Extracting feature importance to understand the weight of different variables (with optional SHAP value integration).
* **AI Business Explanations:** The system translates model outputs into readable, natural language insights. For a given Customer ID, the output includes their assigned segment, the probability of them leaving, and a summary of risk factors (e.g., highlighting that low engagement or high support ticket volumes are the primary drivers).

## Business Value
This tool directly empowers business and product teams to make data-backed retention decisions. It shifts the approach from reactive to proactive by clearly identifying at-risk users, mapping out behavioral segments, and explaining the 'why' behind the churn risk.

## Future Roadmap
Planned enhancements following the initial development phase include:
* Development of an interactive Power BI Dashboard for stakeholder reporting.
* In-depth Cohort Analysis.
* Advanced Retention Analytics.
* Implementation of a Lightweight Analytics Agent.
