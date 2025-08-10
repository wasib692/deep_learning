import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import warnings
warnings.filterwarnings('ignore',category=DeprecationWarning)

# st.set_option('deprecation.showfileUploaderEncoding', False)
# st.set_option('deprecation.showPyplotGlobalUse', False)
# st.set_option('deprecation.showwarn', False)
# Force page styling
st.set_page_config(
    page_title="Cat vs Dog Classifier",
    page_icon="🐾",
    layout="centered"
)

# CSS for styling
st.markdown("""
    <style>
        body, .stApp {
            background-color: #ffffff !important;
            color: #000000 !important;
        }
        h1, h2, h3, h4, h5, h6, p, div {
            color: #000000 !important;
        }
        .upload-box {
            border: 2px dashed #aaa;
            padding: 20px;
            border-radius: 15px;
            text-align: center;
            background-color: #fafafa;
        }
        .result-box {
            padding: 15px;
            border-radius: 10px;
            font-size: 20px;
            font-weight: bold;
            color: black;
            text-align: center;
        }
        .dog {
            background-color: #ffd8b1; /* Soft orange */
        }
        .cat {
            background-color: #d8c1ff; /* Soft purple */
        }
    </style>
""", unsafe_allow_html=True)

# Load model with error handling
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "cats_dogs_model_final.h5")
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

model = load_model()
if model is None:
    st.stop()

# Title
st.title("🐶🐱 Cat vs Dog Classifier")
st.write("Upload an image and the AI will predict if it’s a **Cat** or a **Dog**.")

# Upload box
with st.container():
    st.markdown('<div class="upload-box">📤 **Drop or select your image**</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg", "webp"], accept_multiple_files=False)

# Prediction
if uploaded_file is not None:
    try:
        # Load and display image
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Preprocess image
        img = image.resize((256, 256))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict with spinner
        with st.spinner("Classifying..."):
            prediction = model.predict(img_array)[0][0]

        # Show result
        if prediction > 0.5:
            st.markdown(
                f'<div class="result-box dog">🐶 <b>Dog</b> — {prediction*100:.2f}% confidence</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="result-box cat">🐱 <b>Cat</b> — {(1-prediction)*100:.2f}% confidence</div>',
                unsafe_allow_html=True
            )
    except Exception as e:
        st.error(f"Error processing image or making prediction: {str(e)}")

# Footer
st.markdown("---")
st.caption("Created with ❤️ using Streamlit and TensorFlow")