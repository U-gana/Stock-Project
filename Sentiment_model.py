import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
import joblib
# loading data 
df = pd.read_csv("merged_stock_sentiment_data.csv")

# Drop rows where Cleaned_Text is empty, NaN, or only whitespace
df = df[df['Cleaned_Text'].notna()]  # Remove NaNs
df = df[df['Cleaned_Text'].str.strip() != '']  # Remove empty strings or whitespace

# Assingning sentiment data 
sentiment_df = df[["Date", "Cleaned_Text","Sentiment"]]

# Making Date th index 
sentiment_df.index = pd.to_datetime(sentiment_df["Date"])
sentiment_df.index

# encoding
encoder = LabelEncoder()
sentiment_df["Sentiment"] = encoder.fit_transform(sentiment_df["Sentiment"])
sentiment_df.head()

#Modeling
model  = Pipeline([("tfidf", TfidfVectorizer(stop_words = "english")),("clf", LinearSVC(max_iter=1000, C=1, loss="squared_hinge",class_weight="balanced"))])

# inotialising features and target values 
x = sentiment_df["Cleaned_Text"]
y = sentiment_df["Sentiment"]

# splitting to training and test set 
x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.2)

# fitting data 
model.fit(x_train, y_train)

# make predictin 
y_pred = model.predict(x_test)

joblib.dump(model, "sentiment_Model.pkl")

print(classification_report(y_test, y_pred)),
print(accuracy_score(y_test, y_pred)) 