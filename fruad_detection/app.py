import streamlit as st
import pandas as pd
import joblib
import os
from tensorflow.keras.models import load_model
import numpy as np

# --- PAGE CONFIG ---
st.set_page_config(page_title="Fraud Detection", layout="centered")

# --- CUSTOM BACKGROUND IMAGE ---
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(rgba(0, 0, 0, 0.65), rgba(0, 0, 0, 0.65)), 
                    url('https://stowellcrayk.com/wp-content/uploads/2024/01/credit-card-fraud.webp');
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        color: white;
    }

    .block-container {
        padding-top: 2rem;
    }

    .stSelectbox > div > div {
        color: black !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# --- PAGE TITLE ---
st.title("💳 Credit Card Fraud Detection")

# --- MODEL FILES ---
model_files = {
    "Random Forest": "random_forest.pkl",
    "Gradient Boosting": "gradient_boosting.pkl",
    "SVM": "svm.pkl",
    "Logistic Regression": "logistic_model_b.pkl",
    "Shallow Neural Network": "shallow_nn_b.keras"
}

# --- MODEL SELECTION ---
model_choice = st.selectbox("🔍 Select a model for prediction", list(model_files.keys()))

# --- FILE UPLOAD ---
uploaded_file = st.file_uploader("📁 Upload a CSV file with input features", type=["csv"])

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)

        if data.empty:
            st.warning("⚠️ The uploaded file is empty.")
        else:
            st.write("📄 Uploaded Data Preview", data.head())

            if st.button("🚀 Run Prediction"):
                model_path = os.path.join("fruad_detection", "models", model_files[model_choice])

                if not os.path.exists(model_path):
                    st.error(f"❌ Model file not found: {model_path}")
                else:
                    st.info(f"✅ Loaded model: {model_choice}")

                    if model_files[model_choice].endswith(".keras"):
                        model = load_model(model_path)
                        predictions = model.predict(data)
                        predictions = [1 if p >= 0.5 else 0 for p in predictions.flatten()]
                    else:
                        model = joblib.load(model_path)

                        if hasattr(model, "n_features_in_") and data.shape[1] != model.n_features_in_:
                            st.error(f"❌ Expected {model.n_features_in_} features, got {data.shape[1]}")
                            st.stop()

                        predictions = model.predict(data)

                    data["Prediction"] = predictions
                    data["Prediction"] = data["Prediction"].map({0: "Not Fraud", 1: "Fraud"})

                    st.success("✅ Prediction Complete!")
                    st.dataframe(data)

                    csv = data.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="📥 Download Predictions",
                        data=csv,
                        file_name="fraud_predictions.csv",
                        mime="text/csv",
                    )

    except Exception as e:
        st.error(f"❌ Error occurred: {str(e)}")
