import pandas as pd 
import numpy as np 
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.pipeline import Pipeline


# loading data 
df = pd.read_csv("merged_stock_sentiment_data.csv")

# Drop rows where Cleaned_Text is empty, NaN, or only whitespace
df = df[df['Cleaned_Text'].notna()]  # Remove NaNs
df = df[df['Cleaned_Text'].str.strip() != '']  # Remove empty strings or whitespace

# Creatin a Dataframe containing important features for price prediction
trade_df = df[["Date","Close","Adj Close", "High","Low", "Open","Volume", "Company","Sentiment"]]

trade_df.index = pd.to_datetime(trade_df["Date"])

# turning sentiment Column to int since it is in object format 
encoder =  LabelEncoder()
trade_df["Sentiment"] = encoder.fit_transform(trade_df["Sentiment"])

#intansiate
preprocessor = ColumnTransformer(transformers=[
    ("encode_sentiment", OrdinalEncoder(), ["Company"])
])
model  = Pipeline([("preprocessor", preprocessor),("clf", LinearRegression())])

# inotialising features and target values 
x = trade_df
x = x.iloc[:-1]

y = trade_df['Close'].shift(-1)
y = y.iloc[:-1]

# splitting to training and test set 
x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.2, random_state=42)

# fitting data 
model.fit(x_train, y_train)

# make predictin 
y_pred = model.predict(x_test)

joblib.dump(model, "price_predictor.pkl")
print(mean_absolute_error(y_test, y_pred)),
print(r2_score(y_test, y_pred))
print(mean_squared_error(y_test, y_pred))
