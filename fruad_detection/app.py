import streamlit as st
import pandas as pd
import joblib

# -------- Custom Styling --------
st.markdown(
    """
    <style>
    .main-container {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 12px;
        max-width: 700px;
        margin: auto;
    }
    .dark-title {
        color: #111111;
        font-size: 2.3rem;
        font-weight: 800;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -------- Load ML Models --------
model_dir = "fruad_detection"  # Update as per your project folder
try:
    models = {
        "Logistic Regression": joblib.load(f"{model_dir}/logistic_model_b.pkl"),
        "Support Vector Machine": joblib.load(f"{model_dir}/svm.pkl")
    }
except FileNotFoundError as e:
    st.error(f"❌ Missing model files: {e}")
    st.stop()

# -------- UI --------
st.markdown("<div class='main-container'>", unsafe_allow_html=True)
st.markdown("<div class='dark-title'>💳 Credit Card Fraud Detection</div>", unsafe_allow_html=True)
st.markdown("Select a model and input transaction features to detect fraud:")

# Model selection
model_choice = st.selectbox("🔍 Choose a Model", list(models.keys()))

# Input section
st.subheader("🧾 Transaction Input (30 Features)")
input_features = {}
input_features["Time"] = st.number_input("Time", 0.0, 200000.0, 10000.0)

for i in range(1, 29):
    input_features[f"V{i}"] = st.number_input(f"V{i}", -100.0, 100.0, 0.0)

input_features["Amount"] = st.number_input("Amount ($)", 0.0, 100000.0, 100.0)

# Predict
if st.button("🧠 Predict"):
    input_df = pd.DataFrame([input_features])
    model = models[model_choice]
    prediction = model.predict(input_df)[0]
    result = "⚠️ Fraudulent Transaction!" if prediction == 1 else "✅ Legitimate Transaction."

    st.subheader("📊 Prediction Result")
    if prediction == 0:
        st.success(result)
    else:
        st.error(result)

st.markdown("</div>", unsafe_allow_html=True)
