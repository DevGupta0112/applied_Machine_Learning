import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Fraud Detection", layout="centered")
st.title("💳 Credit Card Fraud Detection")

# Load models
model_files = {
    "Random Forest": "model_rf.pkl",
    "Gradient Boosting": "model_gb.pkl",
    "SVM": "model_svm.pkl",
    "Logistic Regression": "model_lr.pkl",
    "KNN": "model_knn.pkl"
}

model_choice = st.selectbox("Select Model", list(model_files.keys()))

uploaded_file = st.file_uploader("Upload CSV file with test data", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("📄 Uploaded Data", data.head())

    if st.button("Run Prediction"):
        model_path = os.path.join("models", model_files[model_choice])
        model = joblib.load(model_path)

        predictions = model.predict(data)

        data["Prediction"] = predictions
        data["Prediction"] = data["Prediction"].map({0: "Not Fraud", 1: "Fraud"})
        st.success("✅ Prediction Complete")
        st.dataframe(data)

        csv = data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Results",
            data=csv,
            file_name="fraud_predictions.csv",
            mime="text/csv",
        )
