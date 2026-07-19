import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

model = joblib.load("loan_model.pkl")
accuracy = joblib.load("accuracy.pkl")
data = pd.read_csv("dataset/loan_approval_dataset.csv")

# Load trained model
model = joblib.load("loan_model.pkl")

st.set_page_config(page_title="Loan Approval Prediction", layout="centered")

st.title("🏦 Loan Approval Prediction System")
st.metric(
    "Model Accuracy",
    f"{accuracy*100:.2f}%"
)
st.write("Enter the applicant details below.")
st.subheader("Dataset Preview")

st.dataframe(data.head())

st.subheader("Loan Approval Status Distribution")

fig, ax = plt.subplots()

sns.countplot(
    x=" loan_status",
    data=data,
    ax=ax
)

st.pyplot(fig)

st.subheader("Education Distribution")

fig, ax = plt.subplots()

sns.countplot(
    x=" education",
    data=data,
    ax=ax
)

st.pyplot(fig)

# User Inputs

loan_id = st.number_input("Loan ID", min_value=1, step=1)
no_of_dependents = st.number_input("Number of Dependents", min_value=0, step=1)

education = st.selectbox(
    "Education",
    ["Graduate", "Not Graduate"]
)

self_employed = st.selectbox(
    "Self Employed",
    ["Yes", "No"]
)

income_annum = st.number_input("Annual Income")

loan_amount = st.number_input("Loan Amount")

loan_term = st.number_input("Loan Term (Years)", min_value=1)

cibil_score = st.number_input("CIBIL Score", min_value=300, max_value=900)

residential_assets_value = st.number_input("Residential Assets Value")

commercial_assets_value = st.number_input("Commercial Assets Value")

luxury_assets_value = st.number_input("Luxury Assets Value")

bank_asset_value = st.number_input("Bank Asset Value")

if st.button("Predict Loan Approval"):

    # Convert text values into numbers
    education_value = 1 if education == "Graduate" else 0
    self_employed_value = 1 if self_employed == "Yes" else 0


    # Create input data
    input_data = [[
    loan_id,
    no_of_dependents,
    education_value,
    self_employed_value,
    income_annum,
    loan_amount,
    loan_term,
    cibil_score,
    residential_assets_value,
    commercial_assets_value,
    luxury_assets_value,
    bank_asset_value
]]

    # Prediction
    prediction = model.predict(input_data)


    # Display result
    if prediction[0] == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")