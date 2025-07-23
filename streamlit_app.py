import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

@st.cache_resource
def load_model():
  model_path = {"Random Forest Classifier": "RandomForest_Churn_Model2025-07-19__0054.joblib",
                "Logistic Regression": "LogisticRegression_Churn_Model_2025-07-23__1432.joblib"}
  models = {}

  for model_name, model in model_path.items():
    if not os.path.exists(model):
      st.error(f"{model_name} cannot be found at {model}")
      models[model_name] = None
    else:
      models[model_name] = joblib.load(model)
  
  return models       

models = load_model()


st.title("Customer Churn Prediction App")

st.markdown("Enter key customer details below to predict churn possibility")

model_choice = st.selectbox("Choose a Prediction Model", ["Random Forest Classifier", "Logistic Regression"])
model = models.get(model_choice)

with st.form("churn_form"):
  gender = st.selectbox("Gender", ["Female", "Male"])
  Dependents = st.selectbox("Are you a Dependent?", ["No", "Yes"])
  tenure = st.slider("Tenure (months)", 0, 72, 12)
  InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
  OnlineSecurity = st.selectbox("Online Security", ["No", "No internet service", "Yes"])
  OnlineBackup = st.selectbox("Online Backup", ["No", "No internet service", "Yes"])
  TechSupport = st.selectbox("Tech Support", ["No", "No internet service", "Yes"])
  Contract = st.selectbox("How long is your contract?", ["Month-to-month", "one year", "two years"])
  PaymentMethod = st.selectbox("Payment Method", ["Bank Transfer (Automatic)", "Credit Card (Automatic)", "Electronic Check", "Mailed Check"])
  MonthlyCharges = st.number_input("Charges per Month", min_value=0.0)
  TotalCharges = st.number_input("Total Charges", min_value=0.0)

  submitted = st.form_submit_button("Predict")


if submitted and model is not None:
  input_data = pd.DataFrame([{"customerID": 534,
                             "gender": {"Female": 0, "Male": 1}[gender],
                             "SeniorCitizen": 1,
                             "Partner": 0,
                             "Dependents": {"No": 0, "Yes": 1}[Dependents],
                             "tenure": tenure,
                             "PhoneService": 1,
                             "MultipleLines": 2,
                             "InternetService": {"DSL": 0, "Fiber optic": 1, "No": 2}[InternetService],
                             "OnlineSecurity": {"No": 0, "No internet service": 1, "Yes": 2}[OnlineSecurity],
                             "OnlineBackup": {"No": 0, "No internet service": 1, "Yes": 2}[OnlineBackup],
                             "DeviceProtection": 2,
                             "TechSupport": {"No": 0, "No internet service": 1, "Yes": 2}[TechSupport],
                             "StreamingTV": 1,
                            "StreamingMovies": 2,
                             "Contract": {"Month-to-month": 0, "one year": 1, "two years": 2}[Contract],
                           "PaperlessBilling": 1,
                            "PaymentMethod": {"Bank Transfer (Automatic)": 0, "Credit Card (Automatic)": 1, "Electronic Check": 2, "Mailed Check": 3}	[PaymentMethod],
                             "MonthlyCharges": MonthlyCharges,
                            "TotalCharges": TotalCharges}])


  prediction = model.predict(input_data)
  probability = model.predict_proba(input_data)[0][1]
  prob_percent = round(probability * 100, 2)

    
  if prediction[0] == 0:
    st.success(f"✅ The customer is **not likely to churn**. \n\n Retention Probability: {100 - prob_percent}%")
  else:
    st.warning(f"⚠️ The customer is **likely to churn**. \n\n Churn Probability: {prob_percent}%")
    
  
          
        
        
        
     
        
    

