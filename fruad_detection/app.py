import streamlit as st
import pandas as pd
import joblib

# -------- Set Background Image --------
def set_background(url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("{url}");
            background-attachment: fixed;
            background-size: cover;
            background-repeat: no-repeat;
            background-position: center;
        }}
        .main-container {{
            background-color: rgba(255, 255, 255, 0.93);
            padding: 2rem;
            border-radius: 12px;
            max-width: 700px;
            margin: auto;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# ✅ Use working image URL
bg_url = "https://images.unsplash.com/photo-1605902711622-cfb43c4437d5?ixlib=rb-4.0.3&auto=format&fit=crop&w=1470&q=80"
set_background(bg_url)

# -------- Load ML Models --------
model_dir = "fruad_detection"  # Make sure this folder exists and has models

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
st.title("💳 Credit Card Fraud Detection")
st.markdown("Select a model and input transaction features to detect fraud:")

# Model selection
model_choice = st.selectbox("🔍 Choose a Model", list(models.keys()))

# Input section
st.subheader("🧾 Transaction Input (30 Features)")

# Input features: Time, V1 to V28, Amount
input_features = {}
input_features["Time"] = st.number_input("Time", 0.0, 200000.0, 10000.0)

for i in range(1, 29):
    input_features[f"V{i}"] = st.number_input(f"V{i}", -100.0, 100.0, 0.0)

input_features["Amount"] = st.number_input("Amount ($)", 0.0, 100000.0, 100.0)

# Predict button
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
