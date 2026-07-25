import os
import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image, UnidentifiedImageError

# Page config
st.set_page_config(page_title="TOMATO LEAF HEALTH CLASSIFIER", page_icon="🍅", layout="centered")

MODEL_PATH = "model/tomato_model.h5"
IMG_SIZE = (224, 224)
# This must match the class_names printed at the end of the training notebook
CLASS_NAMES = ["Early Blight", "Healthy"]  # index 0 -> Tomato_Early_blight, index 1 -> Tomato_healthy

# Sidebar
with st.sidebar:
    st.header("ABOUT THIS PROJECT")
    st.write(
        "**GET 324 Mini-Project**\n\n"
        "Group CO17\n\n"
        "COMPUTER ENGINEERING\n\n"
        "TASK: Healthy Tomato Leaf vs Tomato Early Blight"
    )
    st.markdown("---")
    st.write("**HOW IT WORKS**")
    st.write(
        "1. Upload a tomato leaf photo (or try a sample below)\n"
        "2. The model resizes and normalizes the image\n"
        "3. A MobileNetV2-based CNN predicts Healthy vs Early Blight\n"
        "4. You get a label plus a confidence score"
    )
    st.markdown("---")
    st.caption("Model: MobileNetV2 transfer learning, trained on the PlantVillage dataset.")

st.title("🍅TOMATO LEAF HEALTH CLASSIFIER")
st.write(
    "Upload a photo of a tomato leaf and the model will predict whether it is "
    "**Healthy** or shows signs of **Early Blight**."
)
# Load model (cached so it only loads once), with graceful error handling
@st.cache_resource
def get_model():
    from tensorflow.keras.models import load_model
    return load_model(MODEL_PATH)

model = None
model_load_error = None

if not os.path.exists(MODEL_PATH):
    model_load_error = (
        f"Model file not found at `{MODEL_PATH}`. Train the model using "
        "`notebooks/train_model.ipynb` and make sure `tomato_model.h5` is saved "
        "into the `model/` folder before running this app."
    )
else:
    try:
        model = get_model()
    except Exception as e:
        model_load_error = f"Failed to load the model: {e}"

if model_load_error:
    st.error(model_load_error)
    st.stop()

# Sample images (optional quick-test gallery)
# Place a few sample images in a "samples/" folder next to app.py, e.g.:
#   samples/healthy_1.jpg, samples/early_blight_1.jpg
SAMPLES_DIR = "samples"
sample_choice = None

if os.path.isdir(SAMPLES_DIR):
    sample_files = [f for f in os.listdir(SAMPLES_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if sample_files:
        st.write("**Or try a sample image:**")
        cols = st.columns(len(sample_files))
        for col, fname in zip(cols, sample_files):
            with col:
                st.image(os.path.join(SAMPLES_DIR, fname), use_container_width=True)
                if st.button(fname.split(".")[0].replace("_", " ").title(), key=f"sample_{fname}"):
                    sample_choice = os.path.join(SAMPLES_DIR, fname)

st.markdown("---")

# Image upload + prediction
uploaded_file = st.file_uploader("Upload a tomato leaf image", type=["jpg", "jpeg", "png"])

image_source = uploaded_file if uploaded_file is not None else sample_choice

if image_source is not None:
    try:
        img = Image.open(image_source).convert("RGB")
    except UnidentifiedImageError:
        st.error("The uploaded file doesn't look like a valid image. Please try a different JPG or PNG.")
        st.stop()
    except Exception as e:
        st.error(f"Couldn't open that image: {e}")
        st.stop()

    st.image(img, caption="Selected image", use_container_width=True)

    # Preprocess to match training pipeline
    img_resized = img.resize(IMG_SIZE)
    img_array = image.img_to_array(img_resized)  # keep raw 0-255 values — the model rescales internally
    img_array = np.expand_dims(img_array, axis=0)

    if st.button("Classify Leaf"):
        try:
            with st.spinner("Analyzing image..."):
                probs = model.predict(img_array)[0]  # two numbers: [P(early blight), P(healthy)]
                pred_idx = int(np.argmax(probs))
                label = CLASS_NAMES[pred_idx]
                confidence = float(probs[pred_idx])
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()

        st.subheader("Result")
        if label == "Healthy":
            st.success(f"✅ {label} — confidence: {confidence * 100:.2f}%")
        else:
            st.error(f"⚠️ {label} — confidence: {confidence * 100:.2f}%")

        st.progress(float(confidence))

st.markdown("---")
st.caption(
    "Built for GET 324 Lab Exercise 10 (Mini-Project) — Group CO17."
)
