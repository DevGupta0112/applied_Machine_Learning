import streamlit as st
import numpy as np
import joblib
import os
# Load model and scaler
APP_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(APP_DIR, 'heart_model.pkl'))
scaler = joblib.load(os.path.join(APP_DIR, 'scaler.pkl'))

# Page config
st.set_page_config(page_title="Heart Disease Detection", page_icon="🫀", layout="centered")

# Stylish CSS with glassmorphism and more
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;700&family=Roboto&display=swap');

    html, body, [data-testid="stAppViewContainer"] {
        background: 
            linear-gradient(rgba(0, 0, 0, 0.4), rgba(0, 0, 0, 0.4)),
            url('https://images.unsplash.com/photo-1588776814546-ec7e6d20b0df?auto=format&fit=crop&w=1350&q=80') no-repeat center center fixed;
        background-size: cover;
        font-family: 'Montserrat', 'Roboto', sans-serif;
        color: #f0f0f0;
        height: 100%;
    }

    [data-testid="stSidebar"] {
        background: rgba(30, 30, 30, 0.85);
        color: #ddd;
        font-weight: 600;
    }

    .main > div {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(12px);
        border-radius: 25px;
        padding: 2.5rem 3rem;
        margin: 2rem auto;
        max-width: 720px;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        border: 1px solid rgba(255, 255, 255, 0.18);
    }

    .css-18e3th9 {
        font-size: 3rem !important;
        font-weight: 800 !important;
        letter-spacing: 1.2px;
        margin-bottom: 0.2rem !important;
        color: #ff4b4b !important;
        text-shadow: 1px 1px 4px rgba(0,0,0,0.7);
    }

    .css-1d391kg {
        color: #f0f0f0 !important;
        font-size: 1.2rem !important;
        margin-bottom: 2rem !important;
        font-weight: 500;
    }

    input, select, textarea {
        background: rgba(255, 255, 255, 0.15) !important;
        color: #f0f0f0 !important;
        border: none !important;
        border-radius: 15px !important;
        padding: 14px 18px !important;
        font-size: 1.1rem !important;
        box-shadow: inset 2px 2px 8px rgba(0,0,0,0.2);
        transition: background 0.3s ease;
        font-family: 'Roboto', sans-serif !important;
    }
    input:focus, select:focus, textarea:focus {
        background: rgba(255, 255, 255, 0.3) !important;
        outline: none !important;
        box-shadow: 0 0 8px 2px #ff4b4b;
    }

    .css-1lcbmhc.e1fqkh3o3 {
        gap: 1.6rem !important;
    }

    button[kind="primary"] {
        background: linear-gradient(90deg, #ff4b4b, #ff1f1f) !important;
        color: white !important;
        font-weight: 700 !important;
        border-radius: 20px !important;
        padding: 16px 30px !important;
        font-size: 1.3rem !important;
        box-shadow: 0 4px 14px rgba(255, 75, 75, 0.5);
        transition: box-shadow 0.3s ease;
        cursor: pointer;
    }
    button[kind="primary"]:hover {
        box-shadow: 0 8px 20px rgba(255, 75, 75, 0.9);
    }

    .result {
        font-size: 1.5rem;
        font-weight: 700;
        padding: 1.4rem 2rem;
        border-radius: 30px;
        text-align: center;
        margin-top: 2rem;
        box-shadow: 0 6px 20px rgba(0,0,0,0.3);
        font-family: 'Montserrat', sans-serif;
        user-select: none;
    }

    .result.positive {
        background: rgba(0, 150, 0, 0.7);
        color: #e0ffe0;
        text-shadow: 0 1px 3px rgba(0, 50, 0, 0.7);
    }

    .result.negative {
        background: rgba(255, 50, 50, 0.7);
        color: #ffe0e0;
        text-shadow: 0 1px 3px rgba(100, 0, 0, 0.7);
    }
    </style>
""", unsafe_allow_html=True)

# App title and subtitle
st.title("🫀 Heart Disease Prediction App")
st.markdown("### 🔎 Enter your health details below")

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

if submitted:
    # Prepare input array
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

    # Scale input and predict
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)

    # Show result with styled box
    if prediction[0] == 1:
        st.markdown(
            '<div class="result negative">⚠️ You may have a heart disease. Please consult a doctor.</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="result positive">✅ You are unlikely to have heart disease. Keep taking care of your health!</div>',
            unsafe_allow_html=True
        )
