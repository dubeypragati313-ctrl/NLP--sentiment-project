import streamlit as st
import pandas as pd
import joblib

# Page Config
st.set_page_config(page_title="Sentiment Analysis App", page_icon="📊", layout="centered")

# Title
st.title("📊 Sentiment Analysis App")

# 1. Load Dataset (Dataset attach karne ke liye)
@st.cache_data
def load_dataset():
    return pd.read_csv("sentimentData.csv")

try:
    df = load_dataset()
except Exception as e:
    st.error("Error: 'sentimentData.csv' nahi mili!")
    st.stop()

# Load Model & Vectorizer
@st.cache_resource
def load_assets():
    model = joblib.load("model.pkl")
    vectorizer = joblib.load("vect.pkl")
    return model, vectorizer

model, vectorizer = load_assets()

# Sidebar me ya Expandable section me Dataset dikhane ke liye
with st.expander("📁 View Attached Dataset (sentimentData.csv)"):
    st.dataframe(df.head(10))  # Pehle 10 rows dikhayega

st.write("---")

# 2. Prediction UI
text_input = st.text_area("Enter Text Here:", placeholder="Type your text...")

if st.button("Predict Sentiment", type="primary"):
    if text_input.strip() == "":
        st.warning("Kripya text enter karein!")
    else:
        # Transform & Predict
        text_vector = vectorizer.transform([text_input])
        prediction = model.predict(text_vector)[0]
        
        st.subheader("Prediction Result:")
        if str(prediction).lower() in ["positive", "1", "pos"]:
            st.success("😊 **Positive Sentiment**")
        elif str(prediction).lower() in ["negative", "0", "neg"]:
            st.error("😠 **Negative Sentiment**")
        else:
            st.info(f"😐 **Result:** {prediction}")