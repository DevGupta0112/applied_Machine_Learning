import streamlit as st
import pandas as pd
import joblib

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
            background-color: rgba(255, 255, 255, 0.93);
            padding: 2rem;
            border-radius: 12px;
            max-width: 900px;
            margin: auto;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Set background image
background_url = "https://images.unsplash.com/photo-1605902711622-cfb43c4437d5?ixlib=rb-4.0.3&auto=format&fit=crop&w=1470&q=80"
set_bg(background_url)

# -------- Load ML models --------
model_dir = "fruad_detection"  # Correct path for deployed models

try:
    models = {
        "Random Forest": joblib.load(f"{model_dir}/random_forest.pkl"),
        "Support Vector Machine": joblib.load(f"{model_dir}/svm.pkl"),
        "Logistic Regression": joblib.load(f"{model_dir}/logistic_model_b.pkl"),
        "Gradient Boosting": joblib.load(f"{model_dir}/gradient_boosting.pkl")
    }
except FileNotFoundError as e:
    st.error(f"❌ Missing model files: {e}")
    st.stop()

# -------- UI --------
st.markdown("<div class='main-container'>", unsafe_allow_html=True)
st.title("💳 Credit Card Fraud Detection")
st.markdown("Select a machine learning model and input transaction details to detect fraud:")

# Model selection
model_choice = st.selectbox("🔍 Choose a Model", list(models.keys()))

# Input section for V1 to V28 + Amount
st.subheader("🧾 Transaction Input")
input_features = {}

# Layout 2 columns for better UX
col1, col2 = st.columns(2)
with col1:
    for i in range(1, 15):
        input_features[f"V{i}"] = st.number_input(f"V{i}", -100.0, 100.0, 0.0)
with col2:
    for i in range(15, 29):
        input_features[f"V{i}"] = st.number_input(f"V{i}", -100.0, 100.0, 0.0)

amount = st.number_input("Transaction Amount ($)", 0.0, 100000.0, 100.0)
input_features["Amount"] = amount

# Predict
if st.button("🧠 Predict"):
    input_df = pd.DataFrame([input_features])
    model = models[model_choice]
    prediction = model.predict(input_df)[0]
    result = "⚠️ Fraudulent Transaction!" if prediction == 1 else "✅ Legitimate Transaction."

    st.subheader("📊 Prediction Result")
    st.success(result) if prediction == 0 else st.error(result)

st.markdown("</div>", unsafe_allow_html=True)
