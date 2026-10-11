import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import os

# Load the IMDB dataset word index
word_index=imdb.get_word_index()
reverse_word_index={value:key for key,value in word_index.items()}

# Load the pre-trained model with ReLu activation
model_path = os.path.join(os.path.dirname(__file__), "my_model.keras")
model = load_model(model_path)

# Function to decode reviews
def decode_reviews(encodeed_review):
    return " ".join([reverse_word_index.get(i-3,"?") for i in encodeed_review])

## Function to process user input
def preprocess_text(text):
    words=text.lower().split()
    encoded_review=[word_index.get(word,2)+3 for word in words]
    padded_review=sequence.pad_sequences([encoded_review],maxlen=500)
    return padded_review


## Prediction Function

def prediction_sentiment(review):
    preprocessed_input=preprocess_text(review)

    prediction=model.predict(preprocessed_input)

    sentiment="Positive" if prediction[0][0] > 0.5 else "Negative"

    return sentiment, prediction[0][0]


## STREAMLIT APP
import streamlit as st

st.title("IMDB Movie Review Sentiment Analysis")
st.write("Enter a movie review to classify it as positive or negative")

# User input 
user_input=st.text_area("Movie Review")

if st.button("Classify"):
    ## get input
    preprocessed_input=preprocess_text(user_input)

    ## Make prediction
    prediction=model.predict(preprocessed_input)
    sentiment="Positive" if prediction[0][0] > 0.5 else "Negative"

    ## Display the result
    st.write(f"Sentiment: {sentiment}")
    st.write(f"Prediction Score: {prediction[0][0]}")
else:
    st.write("Please enter a movie review.")

