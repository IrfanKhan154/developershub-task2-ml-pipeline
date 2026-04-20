# Task 2 - ML Pipeline

This project implements an end-to-end machine learning pipeline for Telco customer churn prediction using **Python** and **scikit-learn**.

## What it does
- Loads the Telco churn dataset from OpenML
- Preprocesses data with:
  - One-hot encoding for categorical features
  - Standard scaling for numeric features
  - Missing-value imputation
- Trains two models using `GridSearchCV`:
  - Logistic Regression
  - Random Forest
- Evaluates models using **accuracy**
- Saves the best model with `joblib` as `best_model.joblib`

## Files
- `pipeline.py` - Full end-to-end training pipeline
- `requirements.txt` - Python dependencies
- `README.md` - Project documentation

## Run locally
```bash
python -m pip install -r requirements.txt
python pipeline.py
```

After running, the best trained pipeline is saved to:
- `best_model.joblib`
