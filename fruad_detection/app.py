import streamlit as st
import pandas as pd
import joblib

# ✅ Function to set background using CSS
def set_bg_image():
    st.markdown(
        """
        <style>
        [data-testid="stAppViewContainer"] {
            background-image: url("https://images.unsplash.com/photo-1605902711622-cfb43c4437d5?ixlib=rb-4.0.3&auto=format&fit=crop&w=1470&q=80");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }
        [data-testid="stHeader"] {
            background: rgba(0,0,0,0);
        }
        [data-testid="stToolbar"] {
            right: 2rem;
        }
        .main-container {
            background-color: rgba(255, 255, 255, 0.95);
            padding: 2rem;
            border-radius: 12px;
            max-width: 800px;
            margin: auto;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# ✅ Call the background function
set_bg_image()

# ✅ Load models
model_dir = "fruad_detection"

# Only load working models (remove non-working)
models = {
    "Logistic Regression": joblib.load(f"{model_dir}/logistic_model_b.pkl"),
    "Support Vector Machine": joblib.load(f"{model_dir}/svm.pkl")
}

# ✅ UI Section
st.markdown("<div class='main-container'>", unsafe_allow_html=True)
st.title("💳 Credit Card Fraud Detection")
st.markdown("Select a model and input transaction features to detect fraud:")

# Model selection
model_choice = st.selectbox("🔍 Choose a Model", list(models.keys()))

# Input form
st.subheader("🧾 Transaction Input (30 Features)")
input_features = {}

# Time
input_features["Time"] = st.number_input("Time", 0.0, 200000.0, 10000.0)

# V1–V28
for i in range(1, 29):
    input_features[f"V{i}"] = st.number_input(f"V{i}", -100.0, 100.0, 0.0)

# Amount
input_features["Amount"] = st.number_input("Amount ($)", 0.0, 100000.0, 100.0)

# Predict
if st.button("🧠 Predict"):
    input_df = pd.DataFrame([input_features])
    model = models[model_choice]
    prediction = model.predict(input_df)[0]
    result = "⚠️ Fraudulent Transaction!" if prediction == 1 else "✅ Legitimate Transaction."

    st.subheader("📊 Prediction Result")
    if prediction == 1:
        st.error(result)
    else:
        st.success(result)

st.markdown("</div>", unsafe_allow_html=True)
