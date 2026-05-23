# Fraud Detection Using ML Models

A machine learning project that detects fraudulent financial transactions using two classification models, served through a Streamlit web app.

## What We Built

- An exploratory data analysis (EDA) of a 6.3M-row financial transactions dataset
- Two trained classification models: **Logistic Regression** and **Random Forest**
- A **Streamlit** web app (`fraud_detection.py`) that loads both models and predicts fraud from user input
- Saved model artifacts (`logistic_model.pkl`, `random_forest_model.pkl`)

## Dataset

- ~6.36M transactions, 11 columns
- Heavily imbalanced: only **0.13%** of transactions are fraud (8,213 out of 6,362,620)
- Fraud only occurs in **TRANSFER** and **CASH_OUT** transactions

> The CSV is not committed (470MB, exceeds GitHub's 100MB limit). Add it locally as `AIML Dataset.csv` in the repo root.

## Feature Engineering

Beyond the raw columns, we added two engineered features that turned out to be the most important signals:

- `balanceDiffOrig` = `oldbalanceOrg - newbalanceOrig` — how much the sender's balance actually changed
- `balanceDiffDest` = `newbalanceDest - oldbalanceDest` — how much the receiver's balance actually changed

We dropped non-predictive columns: `nameOrig`, `nameDest`, `isFlaggedFraud`, and `step`.

## Results

| Model | Accuracy | Fraud Precision | Fraud Recall | Fraud F1 |
|---|---|---|---|---|
| Logistic Regression | 0.95 | **0.02** | 0.94 | 0.04 |
| Random Forest | **1.00** | **0.96** | 0.81 | **0.88** |

The Random Forest is the production model. Logistic Regression is kept for comparison in the UI.

## Tech Stack

- Python 3.13
- pandas, numpy, matplotlib, seaborn
- scikit-learn (Logistic Regression, Random Forest, Pipeline, ColumnTransformer)
- joblib (model serialization)
- streamlit (web UI)

## How to Run

1. Install dependencies:
   ```bash
   py -3.13 -m pip install pandas numpy matplotlib seaborn scikit-learn streamlit joblib
   ```
2. Place `AIML Dataset.csv` in the repo root.
3. Run all cells in `analysis_model.ipynb` to train and save the models.
4. Launch the app:
   ```bash
   py -3.13 -m streamlit run fraud_detection.py
   ```

## What We Learned

**1. Accuracy is a misleading metric on imbalanced data.**
Predicting "not fraud" for every transaction would already give 99.87% accuracy. The model that hit 94.5% accuracy was actually worse than a constant predictor for catching fraud. Always look at **precision, recall, and F1 for the minority class**.

**2. Algorithm choice matters more than tuning when the problem is non-linear.**
Switching from Logistic Regression to Random Forest moved fraud precision from **2% to 96%** and F1 from **0.04 to 0.88**. No hyperparameter tweak on the linear model could have closed that gap — the decision boundary just isn't linear.

**3. Engineered features beat raw features.**
The single biggest signal for fraud was the *inconsistency* between transaction amount and the actual balance change. Adding `balanceDiffOrig` and `balanceDiffDest` gave the tree-based model the structure it needed.

**4. `class_weight="balanced"` is a cheap, effective fix for imbalance.**
We avoided heavier tools like SMOTE and still got strong results just by reweighting the loss function.

**5. Out-of-distribution inputs break models in revealing ways.**
When we tested the app with absurd values (a $1,000 transaction draining a $10B balance to $0), the Logistic Regression flagged it as fraud while the Random Forest didn't. The RF was correctly noting the pattern didn't match its learned fraud signature — but neither model can be trusted on inputs far outside the training distribution. Production systems need input validation, not just model predictions.

