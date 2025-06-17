# pneumonia_web.py

import streamlit as st
import numpy as np
import cv2
import joblib
from PIL import Image

# --- Load the Trained Model ---
try:
    model = joblib.load("pneumonia_model.pkl")
except FileNotFoundError:
    st.error("❌ Model file not found. Please upload pneumonia_model.pkl in your GitHub repo.")
    st.stop()

# --- Streamlit UI ---
st.set_page_config(page_title="Pneumonia Detector", page_icon="🩻")

st.title("🩻 Pneumonia Detector")
st.write("Upload a chest X-ray image to check for Pneumonia.")

# Uploading the image
uploaded_file = st.file_uploader("📂 Choose an X-ray image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded X-ray", use_column_width=True)

    # --- Preprocess image ---
    img_array = np.array(image.convert("L"))  # Convert to grayscale
    img_resized = cv2.resize(img_array, (100, 100)).flatten().reshape(1, -1)

    # --- Predict ---
    prediction = model.predict(img_resized)[0]
    label = "🟢 NORMAL" if prediction == 0 else "🔴 PNEUMONIA"

    st.markdown(f"### 🧠 Prediction: {label}")
