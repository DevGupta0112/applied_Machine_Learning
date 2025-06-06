import streamlit as st
import joblib
import numpy as np
import base64

# Load model and scaler
model = joblib.load('models/diabetes_knn_model.pkl')
scaler = joblib.load('models/scaler.pkl')

# Function to set background from URL
def set_background(image_url):
    st.markdown(f"""
        <style>
        .stApp {{
            background-image: url('{image_url}');
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        .title-text {{
            background-color: rgba(255, 255, 255, 0.8);
            padding: 1rem;
            border-radius: 10px;
        }}
        .form-container {{
            background-color: rgba(255, 255, 255, 0.85);
            padding: 2rem;
            border-radius: 12px;
            box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.2);
        }}
        </style>
    """, unsafe_allow_html=True)

# Set background
bg_url = "https://www.apollo247.com/images/blog/diabetes-blog.jpg"
set_background(bg_url)

# Sidebar for navigation
st.sidebar.title("⚙️ Settings")
theme = st.sidebar.radio("Choose Theme:", ("Light", "Dark"))

# Apply theme CSS
if theme == "Dark":
    st.markdown("""
    <style>
    .stApp {{
        color: white;
        background-color: #222;
    }}
    </style>
    """, unsafe_allow_html=True)

# Page Title
st.markdown("<div class='title-text'><h1 style='text-align: center;'>🩺 Diabetes Prediction App</h1></div>", unsafe_allow_html=True)

# Input Form
st.markdown("<div class='form-container'>", unsafe_allow_html=True)
st.subheader("Enter the following health details:")

features = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]

user_input = []

for feature in features:
    value = st.number_input(f"{feature}", min_value=0.0, format="%0.2f")
    user_input.append(value)

if st.button("🔍 Predict"):
    input_array = np.array(user_input).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    prediction_proba = model.predict_proba(input_scaled)
    prediction = np.argmax(prediction_proba, axis=1)

    result = "🔴 Diabetic" if prediction[0] == 1 else "🟢 Not Diabetic"
    confidence = prediction_proba[0][prediction[0]] * 100

    st.markdown(f"## Prediction: **{result}**")
    st.markdown(f"### Confidence: **{confidence:.2f}%**")

st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("""
---
<small>Made with ❤️ by Dev | Source: Pima Indians Diabetes Dataset</small>
""", unsafe_allow_html=True)