import os
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import streamlit as st


# Load the pre-trained model with ReLu activation
model_path = os.path.join(os.path.dirname(__file__), "lstm_model.keras")
model = load_model(model_path)

# Load the tokenizer
tokenizer_path = os.path.join(os.path.dirname(__file__), "tokenizer.pickle")

with open(tokenizer_path,'rb') as file:
    tokenizer=pickle.load(file)


## Function to predict the next word
def predict_next_word(model,tokenizer,text,max_sequence_len):
    token_list=tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_len:
        token_list=token_list[-(max_sequence_len-1):] ## Ensure the sequence length matches the max_sequence_length
    token_list=pad_sequences([token_list],maxlen=max_sequence_len-1,padding='pre')
    predicted=model.predict(token_list,verbose=0)
    predict_next_word=np.argmax(predicted,axis=1) ## this picks out the word with the highest probability in the predicited proba and returns the index. recall that is the index of the word in the corpus.
    for word, index in tokenizer.word_index.items():
        if index==predict_next_word:
            return word
    return None

## streamlit app
st.title("Next Word Prediction With LSTM And Early Stopping")
input_text=st.text_input("Enter the sequence of words","To be or not be")
if st.button("Predict Next Word"):
    max_sequence_len=model.input_shape[1]+1 #Retrive the max sequence length from the model input shape
    next_word=predict_next_word(model,tokenizer,input_text,max_sequence_len)
    st.write(f"Next word: {next_word}")

