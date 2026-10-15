"""
app.py - Streamlit app for brain tumor detection from MRI (tumor vs no_tumor)

Uses load_artifacts() and predict_pil_image() from predict.py,
so the prediction logic lives in one place.

Run:
    streamlit run app.py
"""

import os
import json
import streamlit as st
from PIL import Image

from predict import load_artifacts, predict_pil_image


# ------------------------------------------------------------------
# 1. Page setup
# ------------------------------------------------------------------
st.set_page_config(page_title="Brain Tumor MRI Classifier", page_icon="🧠", layout="centered")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TUNED_METRICS_PATH = os.path.join(BASE_DIR, "reports", "test_metrics_tuned.json")


# ------------------------------------------------------------------
# 2. Load model once and keep it in memory
# ------------------------------------------------------------------
@st.cache_resource
def get_model_and_meta():
    model, meta = load_artifacts()
    return model, meta


model, meta = get_model_and_meta()


# ------------------------------------------------------------------
# 3. Sidebar: about the model
# ------------------------------------------------------------------
st.sidebar.header("About the model")
st.sidebar.write("Custom CNN (4 conv blocks) trained from scratch on the Kaggle Brain Tumor MRI dataset.")
st.sidebar.write("Glioma, meningioma and pituitary images were merged into one **tumor** class.")
st.sidebar.write("Decision threshold tuned on the validation set to catch more tumors:")
st.sidebar.write("**Threshold:** " + str(round(float(meta["threshold"]), 4)))

if os.path.exists(TUNED_METRICS_PATH):
    with open(TUNED_METRICS_PATH, "r") as f:
        metrics = json.load(f)

    st.sidebar.subheader("Test set results")
    st.sidebar.write("Accuracy: " + str(round(metrics["accuracy"] * 100, 2)) + "%")
    st.sidebar.write("Tumor recall: " + str(round(metrics["tumor_recall"] * 100, 2)) + "%")
    st.sidebar.write("Tumor precision: " + str(round(metrics["tumor_precision"] * 100, 2)) + "%")
    st.sidebar.write("No-tumor recall: " + str(round(metrics["no_tumor_recall"] * 100, 2)) + "%")


# ------------------------------------------------------------------
# 4. Main page
# ------------------------------------------------------------------
st.title("🧠 Brain Tumor MRI Classifier")
st.write("Upload a brain MRI image and the model will predict whether a tumor is present.")

st.warning("This is a learning project, not a medical device. It must not be used for diagnosis.")

uploaded_file = st.file_uploader("Upload an MRI image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    pil_image = Image.open(uploaded_file)

    st.image(pil_image, caption="Uploaded MRI", width=350)

    if st.button("Predict"):
        with st.spinner("Analysing the scan..."):
            result = predict_pil_image(model, meta, pil_image)

        tumor_probability = result["tumor_probability"]
        threshold = result["threshold"]

        if result["label"] == "tumor":
            st.error("Prediction: TUMOR detected")
        else:
            st.success("Prediction: NO TUMOR detected")

        st.write("**Tumor probability:** " + str(round(tumor_probability * 100, 2)) + "%")
        st.progress(min(max(tumor_probability, 0.0), 1.0))

        st.caption(
            "The model flags a tumor when the probability is at or above "
            + str(round(threshold * 100, 2))
            + "%. The threshold is kept low on purpose: missing a tumor is costlier than a false alarm."
        )
