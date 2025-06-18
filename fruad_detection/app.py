import streamlit as st
import pandas as pd
import joblib
import os

# Page setup
st.set_page_config(page_title="Fraud Detection", layout="centered")
st.title("💳 Credit Card Fraud Detection")

# Load available model files (update names to match your folder)
model_files = {
    "Random Forest": "random_forest.pkl",
    "Gradient Boosting": "gradient_boosting.pkl",
    "SVM": "svm.pkl",
    "Logistic Regression": "logistic_model_b.pkl",
    "Shallow Neural Network": "shallow_nn_b.keras"  # optional, if handled differently
}

model_choice = st.selectbox("🔍 Select a model for prediction", list(model_files.keys()))

uploaded_file = st.file_uploader("📁 Upload a CSV file with input features", type=["csv"])

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
        st.write("📄 Uploaded Data Preview", data.head())

        if st.button("🚀 Run Prediction"):
            model_path = os.path.join("fruad_detection", "models", model_files[model_choice])

            # Handle Keras model differently (optional if you're using it)
            if model_files[model_choice].endswith(".keras"):
                from tensorflow.keras.models import load_model
                model = load_model(model_path)
                predictions = model.predict(data)
                predictions = [1 if p >= 0.5 else 0 for p in predictions.flatten()]
            else:
                model = joblib.load(model_path)
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
        st.error(f"❌ Error: {str(e)}")
