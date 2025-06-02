import streamlit as st
import joblib
import numpy as np

# Load models
sentiment_model = joblib.load("sentiment_Model.pkl")
price_model = joblib.load("price_predictor.pkl")

# Title
st.title("📈 AI-Powered Stock Sentiment & Price Predictor")

# User input
st.header("🔍 Step 1: Enter News/Headline")
user_text = st.text_area("Enter a news headline or stock-related summary", height=100)

st.header("📊 Step 2: Enter Market Data")
open_price = st.number_input("Open Price", format="%.2f")
high_price = st.number_input("High Price", format="%.2f")
low_price = st.number_input("Low Price", format="%.2f")
close_price = st.number_input("Close Price", format="%.2f")
adj_close = st.number_input("Adj Close", format="%.2f")
volume = st.number_input("Volume", format="%.0f")
company = st.number_input("Company Code (as integer)", min_value=0)

# Predict button
if st.button("🚀 Predict Sentiment and Price"):
    if not user_text.strip():
        st.error("Please enter a news or summary text.")
    else:
        # Predict sentiment
        sentiment = sentiment_model.predict([user_text])[0]
        st.success(f"Predicted Sentiment: {sentiment}")

        # Prepare features for price prediction
        features = np.array([[close_price, adj_close, high_price, low_price,
                              open_price, volume, company, sentiment]])

        # Predict price
        predicted_price = price_model.predict(features)[0]
        st.success(f"📉 Predicted Next Close Price: ${predicted_price:.2f}")
