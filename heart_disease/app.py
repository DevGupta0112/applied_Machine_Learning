import streamlit as st
import numpy as np
import joblib
import os
# Load model and scaler
APP_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(APP_DIR, 'heart_model.pkl'))
scaler = joblib.load(os.path.join(APP_DIR, 'scaler.pkl'))

# Set page config
st.set_page_config(page_title="Heart Disease Prediction", page_icon="🫀", layout="centered")

# Inject custom CSS for background and styling
st.markdown("""
    <style>
    /* Background */
    .stApp {
        background-image: url('https://images.unsplash.com/photo-1588776814546-ec7e6d20b0df?auto=format&fit=crop&w=1350&q=80');
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        font-family: 'Segoe UI', sans-serif;
    }

    /* Card-style inputs */
    .stNumberInput, .stSelectbox {
        background-color: rgba(255, 255, 255, 0.8) !important;
        border-radius: 12px;
        padding: 10px;
    }

    /* Main content styling */
    .block-container {
        background-color: rgba(255, 255, 255, 0.9);
        padding: 2rem;
        border-radius: 20px;
        max-width: 700px;
        margin: auto;
    }

    /* Buttons */
    button {
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 10px;
    }

    /* Result styling */
    .result {
        font-size: 24px;
        font-weight: bold;
        padding: 1rem;
        border-radius: 15px;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# App title and description
st.image("https://www.google.com/url?sa=i&url=https%3A%2F%2Fwww.alibabacloud.com%2Fblog%2Fzhpredicting-heart-diseases-with-machine-learning_218458&psig=AOvVaw1UorH2SlOWDlPNfjR4VY7i&ust=1749142068936000&source=images&cd=vfe&opi=89978449&ved=0CBQQjRxqFwoTCLj_k5qc2I0DFQAAAAAdAAAAABAE", width=100)
st.title("🫀 Heart Disease Prediction App")
st.markdown("### Enter the following health data:")

# Input form
with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=50)
        sex = st.selectbox("Sex", ["Male", "Female"])
        cp = st.selectbox("Chest Pain Type", ["Typical Angina (0)", "Atypical Angina (1)", "Non-anginal (2)", "Asymptomatic (3)"])
        trestbps = st.number_input("Resting Blood Pressure", min_value=80, max_value=200, value=120)
        chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["Yes (1)", "No (0)"])

    with col2:
        restecg = st.selectbox("Resting ECG Results", ["Normal (0)", "ST-T Abnormality (1)", "LV Hypertrophy (2)"])
        thalach = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=220, value=150)
        exang = st.selectbox("Exercise Induced Angina", ["Yes (1)", "No (0)"])
        oldpeak = st.number_input("Oldpeak (ST Depression)", min_value=0.0, max_value=6.0, value=1.0, step=0.1)
        slope = st.selectbox("Slope of ST Segment", ["Upsloping (0)", "Flat (1)", "Downsloping (2)"])
        ca = st.selectbox("Major Vessels Colored (0–3)", [0, 1, 2, 3])
        thal = st.selectbox("Thalassemia", ["Normal (0)", "Fixed Defect (1)", "Reversible Defect (2)"])

    submitted = st.form_submit_button("🔍 Predict")

# Mapping for categorical options
if submitted:
    input_data = np.array([[age,
                            1 if sex == "Male" else 0,
                            int(cp[cp.find("(")+1]),
                            trestbps,
                            chol,
                            1 if fbs.startswith("Yes") else 0,
                            int(restecg[restecg.find("(")+1]),
                            thalach,
                            1 if exang.startswith("Yes") else 0,
                            oldpeak,
                            int(slope[slope.find("(")+1]),
                            ca,
                            int(thal[thal.find("(")+1])]])

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.markdown(
            '<div class="result" style="background-color:#ffdddd; color:#990000;">⚠️ You may have a heart disease. Please consult a doctor.</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="result" style="background-color:#ddffdd; color:#006600;">✅ You are unlikely to have heart disease. Keep taking care of your health!</div>',
            unsafe_allow_html=True
        )
