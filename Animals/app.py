import streamlit as st
import pickle
import numpy as np
import cv2
from tensorflow.keras.models import model_from_json
from PIL import Image

# ==== 1. Load model from pickle ====
with open("cat_dog_model.pkl", "rb") as f:
    model_dict = pickle.load(f)

model = model_from_json(model_dict["architecture"])
model.set_weights(model_dict["weights"])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ==== 2. App UI ====
st.title("🐱 Cat vs 🐶 Dog Classifier")
st.write("Upload an image and the model will predict whether it's a cat or a dog.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # ==== 3. Preprocess ====
    img_size = 64
    img = image.convert("RGB")
    img = img.resize((img_size, img_size))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # ==== 4. Predict ====
    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction)
    class_label = "Cat 🐱" if class_idx == 0 else "Dog 🐶"

    st.markdown(f"### Prediction: **{class_label}**")
    st.write(f"Confidence: {prediction[0][class_idx]*100:.2f}%")
