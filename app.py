import streamlit as st
import pandas as pd
import joblib

# ---- Load model, scaler, and column structure ----
model = joblib.load("model/churn_model.pkl")
scaler = joblib.load("model/scaler.pkl")
model_columns = joblib.load("model/model_columns.pkl")

st.title("📊 Customer Churn Prediction")
st.write("Enter customer details to predict the likelihood of churn.")

# ---- Input fields ----
gender = st.selectbox("Gender", ["Male", "Female"])
senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
partner = st.selectbox("Partner", ["No", "Yes"])
dependents = st.selectbox("Dependents", ["No", "Yes"])
tenure = st.slider("Tenure (months)", 0, 72, 12)
phone_service = st.selectbox("Phone Service", ["No", "Yes"])
multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
payment_method = st.selectbox("Payment Method", [
    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
])
monthly_charges = st.number_input("Monthly Charges", min_value=0.0, max_value=2000.0, value=70.0)
total_charges = st.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=1000.0)

# ---- Predict button ----
if st.button("Predict Churn"):

    # Step 1: Build a single-row dataframe matching raw column names
    input_dict = {
        "gender": gender,
        "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }

    input_df = pd.DataFrame([input_dict])

    # Step 2: One-hot encode the same way training data was encoded
    input_encoded = pd.get_dummies(input_df)

    # Step 3: Align columns with training columns (fill missing with 0)
    input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)

    # Step 4: Scale the numeric columns (same scaler used in training)
    num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    input_encoded[num_cols] = scaler.transform(input_encoded[num_cols])

    # Step 5: Predict
    prediction = model.predict(input_encoded)[0]
    probability = model.predict_proba(input_encoded)[0][1]

    # ---- Display result ----
    if prediction == 1:
        st.error(f"⚠️ This customer is likely to CHURN (probability: {probability:.2%})")
    else:
        st.success(f"✅ This customer is likely to STAY (probability of churn: {probability:.2%})")