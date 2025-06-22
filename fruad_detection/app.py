import streamlit as st
import pandas as pd
import joblib
import os

# -------- Background Image Setup --------
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
            max-width: 800px;
            margin: auto;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# -------- Set Background --------
background_url = "https://cdn.prod.website-files.com/64035b5f27be9bd2bfb64a75/66a2616fd5b7d9bab6484f65_Enhance%20Your%20Risk%20Insurance%20Strategy%20(5).jpg"
set_bg(background_url)

# -------- Load ML Models --------
model_dir = "models"
required_files = [
    "random_forest.pkl",
    "svm.pkl",
    "logistic_model_b.pkl",
    "gradient_boosting.pkl"
]

missing_files = [f for f in required_files if not os.path.exists(os.path.join(model_dir, f))]
if missing_files:
    st.error(f"❌ Missing model files: {', '.join(missing_files)}.\n\nPlease check the 'models' folder.")
    st.stop()

# Load models
models = {
    "Random Forest": joblib.load(os.path.join(model_dir, "random_forest.pkl")),
    "Support Vector Machine": joblib.load(os.path.join(model_dir, "svm.pkl")),
    "Logistic Regression": joblib.load(os.path.join(model_dir, "logistic_model_b.pkl")),
    "Gradient Boosting": joblib.load(os.path.join(model_dir, "gradient_boosting.pkl"))
}

# -------- App UI --------
st.markdown("<div class='main-container'>", unsafe_allow_html=True)
st.title("💳 Credit Card Fraud Detection")

st.markdown("🔍 Use different ML models to analyze transaction data and detect fraud risk.")

# Model selection
model_choice = st.selectbox("Choose a Prediction Model", list(models.keys()))

# Input Features
st.subheader("📋 Transaction Input")
v1 = st.number_input("V1", min_value=-100.0, max_value=100.0, value=0.0)
v2 = st.number_input("V2", min_value=-100.0, max_value=100.0, value=0.0)
v3 = st.number_input("V3", min_value=-100.0, max_value=100.0, value=0.0)
v4 = st.number_input("V4", min_value=-100.0, max_value=100.0, value=0.0)
amount = st.number_input("Transaction Amount ($)", min_value=0.0, max_value=100000.0, value=100.0)

# Predict
if st.button("🚀 Predict Fraud"):
    input_data = pd.DataFrame([[v1, v2, v3, v4, amount]], columns=["V1", "V2", "V3", "V4", "Amount"])
    model = models[model_choice]
    prediction = model.predict(input_data)[0]
    result = "⚠️ Fraudulent Transaction!" if prediction == 1 else "✅ Legitimate Transaction."
    
    st.subheader("🔎 Prediction Result")
    st.success(result) if prediction == 0 else st.error(result)

st.markdown("</div>", unsafe_allow_html=True)
