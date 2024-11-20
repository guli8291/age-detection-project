import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image
import os

# Load your trained models
basic_cnn_model = tf.keras.models.load_model("baseline_model.h5")
resnet50_model = tf.keras.models.load_model("enhanced_model.h5")

# Define image size
IMAGE_SIZE = (224, 224)

# Function to preprocess the uploaded image
def preprocess_image(image):
    image = image.resize(IMAGE_SIZE)
    image_array = img_to_array(image) / 255.0  # Normalize
    return np.expand_dims(image_array, axis=0)  # Add batch dimension

# Prediction function
def predict_image(image, model):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction)
    return predicted_class, confidence

# Class labels
class_labels = {0: "18-25", 1: "26-35", 2: "36-45", 3: "46-55", 4: "56-60"}

# Streamlit App Layout
st.title("Age Detection with Basic CNN and ResNet50")
st.write("Upload an image, and let the models predict the age range!")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Predictions
    st.write("**Predictions:**")
    if st.button("Predict with Basic CNN"):
        predicted_class, confidence = predict_image(image, basic_cnn_model)
        st.write(f"**Basic CNN Prediction:** {class_labels[predicted_class]} with confidence {confidence:.2f}")

    if st.button("Predict with ResNet50"):
        predicted_class, confidence = predict_image(image, resnet50_model)
        st.write(f"**ResNet50 Prediction:** {class_labels[predicted_class]} with confidence {confidence:.2f}")