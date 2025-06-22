import streamlit as st
import pandas as pd
import joblib

# -------- Background image setup --------
def set_bg(url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("{url}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        .main-container {{
            background-color: rgba(255, 255, 255, 0.93);
            padding: 2rem;
            border-radius: 12px;
            max-width: 900px;
            margin: auto;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Background image
background_url = "https://images.unsplash.com/photo-1605902711622-cfb43c4437d5?ixlib=rb-4.0.3&auto=format&fit=crop&w=1470&q=80"
set_bg(background_url)

# -------- Load ML models --------
model_dir = "fruad_detection"

try:
    models = {
        "Random Forest": joblib.load(f"{model_dir}/random_forest.pkl"),
        "Support Vector Machine": joblib.load(f"{model_dir}/svm.pkl"),
        "Logistic Regression": joblib.load(f"{model_dir}/logistic_model_b.pkl"),
        "Gradient Boosting": joblib.load(f"{model_dir}/gradient_boosting.pkl")
    }
except FileNotFoundError as e:
    st.error(f"❌ Missing model files: {e}")
    st.stop()

# -------- UI --------
st.markdown("<div class='main-container'>", unsafe_allow_html=True)
st.title("💳 Credit Card Fraud Detection")
st.markdown("Select a model and input transaction features to detect fraud:")

model_choice = st.selectbox("🔍 Choose a Model", list(models.keys()))

# -------- Input Section --------
st.subheader("🧾 Transaction Input (30 Features)")

# Input fields for Time
time = st.number_input("Time", 0.0, 200000.0, 10000.0)

# Input fields for V1 - V28
v_features = {}
cols = st.columns(3)  # Divide input fields into 3 columns

for i in range(1, 29):
    col = cols[(i - 1) % 3]
    with col:
        v_features[f"V{i}"] = st.number_input(f"V{i}", -100.0, 100.0, 0.0)

# Input for Amount
amount = st.number_input("Amount ($)", 0.0, 100000.0, 100.0)

# Predict
if st.button("🧠 Predict"):
    input_values = [time] + list(v_features.values()) + [amount]
    input_df = pd.DataFrame([input_values], columns=["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"])

    try:
        model = models[model_choice]
        prediction = model.predict(input_df)[0]
        result = "⚠️ Fraudulent Transaction!" if prediction == 1 else "✅ Legitimate Transaction."

        st.subheader("📊 Prediction Result")
        st.success(result) if prediction == 0 else st.error(result)
    except Exception as e:
        st.error(f"Prediction error: {e}")

st.markdown("</div>", unsafe_allow_html=True)
