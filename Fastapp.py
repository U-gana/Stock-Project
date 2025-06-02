from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Load models
sentiment_model = joblib.load("sentiment_Model.pkl")
price_model = joblib.load("price_predictor.pkl")

# Define input schema
class PredictionInput(BaseModel):
    text: str
    open: float
    high: float
    low: float
    close: float
    adj_close: float
    volume: float
    company: str  # company name as string

# Create FastAPI app
app = FastAPI()

@app.post("/predict")
def predict(data: PredictionInput):
    # Step 1: Predict Sentiment
    sentiment = sentiment_model.predict([data.text])[0]

    # Step 2: Prepare data for price prediction
    input_df = pd.DataFrame([{
        "Close": data.close,
        "Adj Close": data.adj_close,
        "High": data.high,
        "Low": data.low,
        "Open": data.open,
        "Volume": data.volume,
        "Company": data.company,
        "Sentiment": sentiment
    }])

    # Step 3: Predict price
    predicted_price = price_model.predict(input_df)[0]

    return {
        "predicted_sentiment": int(sentiment),
        "predicted_next_close_price": predicted_price
    }
