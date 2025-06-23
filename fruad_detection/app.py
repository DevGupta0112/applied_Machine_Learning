import streamlit as st
import pandas as pd
import joblib

# -------- Set background image --------
def set_bg(url):
    st.markdown(
        f"""
        <style>
        /* Background for entire app */
        [data-testid="stAppViewContainer"] > .main {{
            background-image: url("{url}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}

        /* Remove header background */
        [data-testid="stHeader"] {{
            background: rgba(0,0,0,0);
        }}

        .main-container {{
            background-color: rgba(255, 255, 255, 0.90);
            padding: 2rem;
            border-radius: 12px;
            max-width: 800px;
            margin: auto;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Background image URL
background_url = "https://images.unsplash.com/photo-1605902711622-cfb43c4437d5?ixlib=rb-4.0.3&auto=format&fit=crop&w=1470&q=80"
set_bg(background_url)

# -------- Load working models only --------
model_dir = "fruad_detection"  # Adjust if needed

try:
    models = {
        "Support Vector Machine": joblib.load(f"{model_dir}/svm.pkl"),
        "Logistic Regression": joblib.load(f"{model_dir}/logistic_model_b.pkl")
    }
except FileNotFoundError as e:
    st.error(f"❌ Missing model file: {e}")
    st.stop()

# -------- UI Layout --------
st.markdown("<div class='main-container'>", unsafe_allow_html=True)
st.title("💳 Credit Card Fraud Detection")
st.markdown("Select a model and input transaction features to detect fraud:")

# Model choice dropdown
model_choice = st.selectbox("🔍 Choose a Model", list(models.keys()))

# -------- Input Section --------
st.subheader("🧾 Transaction Input (30 Features)")

input_features = {}
input_features["Time"] = st.number_input("Time", 0.0, 200000.0, 10000.0)

# V1 to V28
for i in range(1, 29):
    input_features[f"V{i}"] = st.number_input(f"V{i}", -100.0, 100.0, 0.0)

input_features["Amount"] = st.number_input("Amount ($)", 0.0, 100000.0, 100.0)

# -------- Prediction --------
if st.button("🧠 Predict"):
    input_df = pd.DataFrame([input_features])
    model = models[model_choice]
    prediction = model.predict(input_df)[0]

    st.subheader("📊 Prediction Result")
    result = "⚠️ Fraudulent Transaction!" if prediction == 1 else "✅ Legitimate Transaction."
    
    if prediction == 0:
        st.success(result)
    else:
        st.error(result)

st.markdown("</div>", unsafe_allow_html=True)
