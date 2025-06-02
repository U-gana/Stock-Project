import pandas as pd
import numpy as np
from fastapi import FastAPI, Request
from pydantic import BaseModel
import joblib

# Load your models
sentiment_model = joblib.load("sentiment_Model.pkl")  # e.g., sklearn pipeline
price_model = joblib.load("price_predictor.pkl")          # e.g., RandomForestRegressor

# Create FastAPI app
app = FastAPI()

# Define input structure
class PredictionInput(BaseModel):
    text: str
    open: float
    high: float
    Close:float
    low: float
    volume: float
    Adj_Close:float
    Company:int
    Sentiment:int
    # Add any other features needed for price prediction


# Define a prediction endpoint
@app.post("/predict")
def predict(input: PredictionInput):
    # Perform prediction
    prediction = sentiment_model.predict([input.text])
    return {"text": input.text, "prediction": prediction[0]}
@app.post("/predict")
def predict(data: PredictionInput):
   

    # Step 2: Construct features for price model (example)
    features = [[data.Close, data.Adj_Close,data.high, data.low,data.open, data.volume,data.Company, data.Sentiment  ]]

    # Step 3: Predict price
    predicted_price = price_model.predict(features)

    return {
    
        "predicted_price": predicted_price
    }
