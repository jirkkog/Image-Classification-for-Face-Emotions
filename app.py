import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
from PIL import Image

# Load the trained model
MODEL_PATH = 'path_to_your_saved_model/imageclassifier.h5'
model = load_model(MODEL_PATH)

# Helper function to preprocess the image
def preprocess_image(img):
    img = img.resize((256, 256))  # Resize to match model input size
    img_array = np.array(img)
    img_array = img_array / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Streamlit app
st.title("Image Classification App")
st.write("Upload an image to classify it using the trained model.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocess the image
    img = Image.open(uploaded_file)
    img_array = preprocess_image(img)

    # Make a prediction
    prediction = model.predict(img_array)
    
    # Assuming binary classification: 0 -> Class A, 1 -> Class B
    if prediction > 0.5:
        st.write(f"Prediction: Class B (Confidence: {prediction[0][0]:.2f})")
    else:
        st.write(f"Prediction: Class A (Confidence: {1 - prediction[0][0]:.2f})")
