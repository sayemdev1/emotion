# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 01:45:39 2024

@author: pahar
"""

import numpy as np
import pickle
import streamlit as st
from transformers import BertTokenizer, TFBertModel
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.naive_bayes import MultinomialNB  # Import Naive Bayes model class

# Load the saved models
svm_model = pickle.load(open('C:/Users/pahar/Work/emo_pred/svm_model.sav', 'rb'))
logistic_regression_model = pickle.load(open('C:/Users/pahar/Work/emo_pred/logistic_regression_model.sav', 'rb'))
random_forest_model = pickle.load(open('C:/Users/pahar/Work/emo_pred/random_forest_model.sav', 'rb'))
naive_bayes_model = pickle.load(open('C:/Users/pahar/Work/emo_pred/naive_bayes_model.sav', 'rb'))  # Load Naive Bayes model
bert_model = pickle.load(open('C:/Users/pahar/Work/emo_pred/logistic_regression_model.sav', 'rb'))

# Load tokenizer and BERT model
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
model = TFBertModel.from_pretrained('bert-base-multilingual-uncased')

# Create a mapping dictionary for the prediction labels
label_mapping = {
    0: 'angry',
    1: 'disgust',
    2: 'fear',
    3: 'happy',
    4: 'sad',
    5: 'surprise'
}

# Preprocessing function
def preprocess_function(texts):
    return tokenizer(texts, padding=True, truncation=True, return_tensors='tf', max_length=128)

# Get BERT embeddings function
def get_bert_embeddings(encodings):
    outputs = model(encodings)
    embeddings = outputs.last_hidden_state[:, 0, :]
    return embeddings.numpy()

# Prediction function
def emotion_prediction(input_text, model, model_name):
    input_encodings = preprocess_function([input_text])
    
    if model_name == 'BERT Fine-Tuned':
        input_ids = input_encodings['input_ids']
        attention_mask = input_encodings['attention_mask']
        prediction = model.predict([input_ids, attention_mask])
        predicted_class = np.argmax(prediction, axis=1)[0]
    else:
        input_embeddings = get_bert_embeddings(input_encodings)
        
        if model_name == 'SVM':
            prediction = model.predict(input_embeddings)
            predicted_class = int(prediction[0])
        elif model_name == 'Logistic Regression':
            prediction = model.predict(input_embeddings)
            predicted_class = int(prediction[0])
        elif model_name == 'Random Forest':
            prediction = model.predict(input_embeddings)
            predicted_class = int(prediction[0])
        elif model_name == 'Naive Bayes':
            # Naive Bayes model might need the embeddings directly as input
            prediction = model.predict(input_embeddings)
            predicted_class = int(prediction[0])
        else:
            raise ValueError("Unknown model type")
        
    predicted_label = label_mapping[predicted_class]
    return predicted_label

def main():
    # Title of the web app
    st.title('Bengali Emotion Prediction from Text')

    # Input text from the user
    bengali_sentence = st.text_input('Enter a sentence in Bengali')

    # Model selection
    model_option = st.selectbox('Select the model', (
        'SVM', 'Logistic Regression', 'Random Forest', 'Naive Bayes', 'BERT Fine-Tuned'))

    # Result initialization
    result = ''

    # Prediction button
    if st.button('Predict Emotion'):
        result = emotion_prediction(bengali_sentence, 
                                    svm_model if model_option == 'SVM' else
                                    logistic_regression_model if model_option == 'Logistic Regression' else
                                    random_forest_model if model_option == 'Random Forest' else
                                    naive_bayes_model if model_option == 'Naive Bayes' else
                                    bert_model, 
                                    model_option)

    # Display the result
    if result:
        st.success(f'Predicted Emotion: {result}')
        
    # Credits
    st.markdown("---")  # Separator
    st.markdown("**Developed by Soham Pahari**")
    st.markdown("**Mentored by Dr. Sahinur Rahaman**")

if __name__ == '__main__':
    main()
