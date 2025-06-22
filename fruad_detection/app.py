import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.models import load_model

st.title("💳 Credit Card Fraud Detection")
st.markdown("Choose a model and enter transaction details to check for fraud.")

model_choice = st.selectbox("Choose a model:", [
    "Logistic Regression", "Random Forest", "Gradient Boosting", "SVM", "Shallow Neural Network"
])

if model_choice == "Shallow Neural Network":
    model = load_model("models/shallow_nn_b.keras")
    is_keras = True
else:
    file_map = {
        "Logistic Regression": "logistic_model_b.pkl",
        "Random Forest": "random_forest.pkl",
        "Gradient Boosting": "gradient_boosting.pkl",
        "SVM": "svm.pkl"
    }
    with open(f"models/{file_map[model_choice]}", "rb") as f:
        model = pickle.load(f)
    is_keras = False

# Inputs
st.subheader("📝 Enter Transaction Info")
time = st.number_input("Time", min_value=0.0, value=10000.0)
amount = st.number_input("Amount", min_value=0.0, value=200.0)
features = [st.number_input(f"V{i}", value=0.0) for i in range(1, 29)]

input_data = np.array([[time, amount] + features])

# Predict
if st.button("Predict"):
    if is_keras:
        prediction = (model.predict(input_data) > 0.5).astype("int")
    else:
        prediction = model.predict(input_data)
    result = "🚨 Fraud Detected!" if prediction[0] == 1 else "✅ Legitimate Transaction"
    st.success(f"Prediction: {result}")
