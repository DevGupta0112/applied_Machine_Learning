import streamlit as st
import pandas as pd
import joblib
import base64

# -------- Background image setup --------
def set_bg(url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("{url}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        .main-container {{
            background-color: rgba(255, 255, 255, 0.9);
            padding: 2rem;
            border-radius: 10px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Background URL (credit card fraud image)
background_url = "https://cdn.prod.website-files.com/64035b5f27be9bd2bfb64a75/66a2616fd5b7d9bab6484f65_Enhance%20Your%20Risk%20Insurance%20Strategy%20(5).jpg"
set_bg(background_url)

# -------- Load ML models --------
models = {
    "Random Forest": joblib.load("models/random_forest.pkl"),
    "Support Vector Machine": joblib.load("models/svm.pkl"),
    "Logistic Regression": joblib.load("models/logistic_model_b.pkl"),
    "Gradient Boosting": joblib.load("models/gradient_boosting.pkl")
}

# -------- App UI --------
st.markdown("<div class='main-container'>", unsafe_allow_html=True)
st.title("💳 Credit Card Fraud Detection App")

st.markdown("Select a machine learning model and input transaction details to check if it's fraudulent.")

# Model selection
model_choice = st.selectbox("🔍 Select Prediction Model", list(models.keys()))

# Input features (simplified - match your dataset)
st.subheader("🧾 Transaction Details")
v1 = st.number_input("V1", -100.0, 100.0, 0.0)
v2 = st.number_input("V2", -100.0, 100.0, 0.0)
v3 = st.number_input("V3", -100.0, 100.0, 0.0)
v4 = st.number_input("V4", -100.0, 100.0, 0.0)
amount = st.number_input("Transaction Amount ($)", 0.0, 100000.0, 100.0)

# Predict button
if st.button("🧠 Predict Fraud"):
    input_data = pd.DataFrame([[v1, v2, v3, v4, amount]],
                              columns=["V1", "V2", "V3", "V4", "Amount"])
    
    model = models[model_choice]
    prediction = model.predict(input_data)[0]
    result = "⚠️ Fraudulent Transaction!" if prediction == 1 else "✅ Legitimate Transaction."

    st.subheader("🔎 Prediction Result")
    st.success(result) if prediction == 0 else st.error(result)

st.markdown("</div>", unsafe_allow_html=True)
