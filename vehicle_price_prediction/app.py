import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle


model = tf.keras.models.load_model('used_bikes.h5')



# Load the encoders and scaler
with open('label_encoder_owner.pkl', 'rb') as file:
    label_encoder_owner = pickle.load(file)

with open('onehot_encoder_brand.pkl', 'rb') as file:
    onehot_encoder_brand = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# streamlit app
st.title('Used_bikes _price_prediction')

# user input
brand = st.selectbox('brand', onehot_encoder_brand.categories_[0])
owner = st.selectbox('owner', label_encoder_owner.classes_)
age = st.slider('age', 3, 25)
power = st.slider('power', 100,1800)
kms_driven = st.number_input('kms_driven')



# prepare the input data (match training data structure)
input_data = pd.DataFrame({
    'kms_driven': [kms_driven],
    'owner': [label_encoder_owner.transform([owner])[0]],
    'age': [age],
    'power':[power],
    'brand': [brand]  # Will be one-hot encoded next
})

# One-hot encode 'brand' (only once)
brand_encoded = onehot_encoder_brand.transform([[brand]]).toarray()
brand_encoded_df = pd.DataFrame(brand_encoded, 
                              columns=onehot_encoder_brand.get_feature_names_out(['brand']))

# Combine and drop original brand column
input_data = pd.concat([input_data.drop('brand', axis=1), brand_encoded_df], axis=1)

# Scale the input data (without power feature, like in training)
input_data_scaled = scaler.transform(input_data)




# Predict bike price
prediction = model.predict(input_data_scaled)
predicted_price = prediction[0][0]

st.write(f'Predicted Bike Price is: ₹{predicted_price:.2f}')