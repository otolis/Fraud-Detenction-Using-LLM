import streamlit as st
import pandas as pd
import joblib

# Load both models
logistic_model = joblib.load("logistic_model.pkl")
rf_model = joblib.load("random_forest_model.pkl")

st.title("Fraud Detection Using ML Models")
st.markdown("Please enter the transaction details and use the predict button to see the results.")
st.divider()
transaction_type = st.selectbox("Transaction Type", ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CREDIT", "DEPOSIT"])
amount= st.number_input("Amount", min_value=0.0, value=1000.0)
oldbalanceOrg = st.number_input("Old Balance (Sender)", min_value=0.0, value=1000.0)
newbalanceOrig = st.number_input("New Balance (Sender)", min_value=0.0, value=9000.0)
oldbalanceDest = st.number_input("Old Balance (Receiver)", min_value=0.0, value=0.0)
newbalanceDest = st.number_input("New Balance (Receiver)", min_value=0.0, value=0.0)

if st.button("Predict"):
    input_data = pd.DataFrame({
        "type": [transaction_type],
        "amount": [amount],
        "oldbalanceOrg": [oldbalanceOrg],
        "newbalanceOrig": [newbalanceOrig],
        "oldbalanceDest": [oldbalanceDest],
        "newbalanceDest": [newbalanceDest],
        "balanceDiffOrig": [oldbalanceOrg - newbalanceOrig],
        "balanceDiffDest": [newbalanceDest - oldbalanceDest]
    })
    
    logistic_prediction = logistic_model.predict(input_data)[0]
    rf_prediction = rf_model.predict(input_data)[0]
    
    st.write(f"Logistic Regression Prediction: {'Fraud' if logistic_prediction == 1 else 'Not Fraud'}")
    st.write(f"Random Forest Prediction: {'Fraud' if rf_prediction == 1 else 'Not Fraud'}")
    