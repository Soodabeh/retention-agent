## Data Acquisition

The notebook uses the KKBox Churn Prediction dataset.

Kaggle:
https://www.kaggle.com/competitions/kkbox-churn-prediction-challenge/data

Download the following files from Kaggle:

- train.csv.7z
- members_v3.csv.7z
- transactions.csv.7z
- user_logs.csv.7z

Place the files in:

```text
raw_data/
```

Extract them with:

```bash
7z x train.csv.7z
7z x members_v3.csv.7z
7z x transactions.csv.7z
7z x user_logs.csv.7z
```

## Dataset Overview

### train.csv

Target labels.


### members_v3.csv

Customer demographics.


### transactions.csv

Subscription and payment history.


## Initial Exploration

Current observations:

- Churn labels already exist in train.csv
- Customer identifier: `msno` (ID)
- Transactions table contains multiple rows per customer
- Transaction data will need to be aggregated before modeling

Planned baseline:

train
- members
- aggregated transactions

Target:
is_churn


### notebooks/01_data_exploration.ipynb

Initial exploration of:

- dataset sizes
- churn distribution
- customer demographics
- subscription behavior
- transaction statistics
