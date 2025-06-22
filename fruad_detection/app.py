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
            max-width: 700px;
            margin: auto;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Background image
background_url = "https://cdn.prod.website-files.com/64035b5f27be9bd2bfb64a75/66a2616fd5b7d9bab6484f65_Enhance%20Your%20Risk%20Insu
