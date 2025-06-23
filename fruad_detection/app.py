import streamlit as st
import pandas as pd
import joblib
st.set_page_config(page_title="Credit Card Fraud Detection", layout="centered")

# Load models
model_dir = "fruad_detection"
try:
    models = {
        "Support Vector Machine": joblib.load(f"{model_dir}/svm.pkl"),
        "Logistic Regression": joblib.load(f"{model_dir}/logistic_model_b.pkl"),
    }
except FileNotFoundError as e:
    st.error(f"Missing model file: {e}")
    st.stop()

# Custom CSS for background
bg_image = "https://images.unsplash.com/photo-1605902711622-cfb43c4437d5?ixlib=rb-4.0.3&auto=format&fit=crop&w=1470&q=80"
st.markdown(f"""
    <style>
    .stApp {{
      background: url("{bg_image}") no-repeat center center fixed;
      background-size: cover;
    }}
    .main-container {{
      background-color: rgba(255,255,255,0.95);
      padding: 2rem; border-radius: 15px;
      box-shadow: 0px 0px 15px rgba(0,0,0,0.2);
      max-width: 750px; margin: 3rem auto;
    }}
    </style>
""", unsafe_allow_html=True)

# UI layout
st.markdown("<div class='main-container'>", unsafe_allow_html=True)
st.title("💳 Credit Card Fraud Detection")
st.markdown("**Choose a model and input 30 features to predict fraud.**")

model_choice = st.selectbox("Select model", list(models.keys()))
st.subheader("Transaction Features")
features = {"Time": st.number_input("Time", 0.0, 200000.0, 10000.0)}
for i in range(1,29):
    features[f"V{i}"] = st.number_input(f"V{i}", -100.0, 100.0, 0.0)
features["Amount"] = st.number_input("Amount", 0.0, 100000.0, 100.0)

if st.button("Predict"):
    df = pd.DataFrame([features])
    pred = models[model_choice].predict(df)[0]
    msg = "✅ Legitimate" if pred == 0 else "⚠️ Fraudulent"
    st.subheader("Result")
    st.success(msg) if pred == 0 else st.error(msg)

st.markdown("</div>", unsafe_allow_html=True)
