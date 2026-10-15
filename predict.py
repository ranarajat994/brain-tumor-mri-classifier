"""
predict.py - predict tumor / no_tumor for a single brain MRI image

Uses:
    models/brain_tumor_cnn.keras   (saved by train.py)
    models/model_meta.json         (class names, image size, tuned threshold)

Run from the command line:
    python predict.py path/to/mri_image.jpg

Or import in app.py:
    from predict import load_artifacts, predict_pil_image
"""

import os
import sys
import json
import numpy as np
from PIL import Image
from tensorflow import keras


# ------------------------------------------------------------------
# 1. Paths
# ------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "brain_tumor_cnn.keras")
META_PATH = os.path.join(BASE_DIR, "models", "model_meta.json")


# ------------------------------------------------------------------
# 2. Load model + metadata
# ------------------------------------------------------------------
def load_artifacts():
    model = keras.models.load_model(MODEL_PATH)

    with open(META_PATH, "r") as f:
        meta = json.load(f)

    return model, meta


# ------------------------------------------------------------------
# 3. Prepare one image exactly like training did
#    - RGB
#    - resized to img_size
#    - NO division by 255 here: the model has its own Rescaling layer
# ------------------------------------------------------------------
def prepare_image(pil_image, meta):
    img_height = meta["img_size"][0]
    img_width = meta["img_size"][1]

    pil_image = pil_image.convert("RGB")
    pil_image = pil_image.resize((img_width, img_height))

    img_array = keras.utils.img_to_array(pil_image)   # shape: (224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)      # shape: (1, 224, 224, 3)

    return img_array


# ------------------------------------------------------------------
# 4. Predict from a PIL image (used by app.py for uploaded files)
# ------------------------------------------------------------------
def predict_pil_image(model, meta, pil_image):
    img_array = prepare_image(pil_image, meta)

    tumor_probability = float(model.predict(img_array, verbose=0)[0][0])
    threshold = float(meta["threshold"])

    if tumor_probability >= threshold:
        label = "tumor"
    else:
        label = "no_tumor"

    result = {
        "label": label,
        "tumor_probability": round(tumor_probability, 4),
        "threshold": round(threshold, 4),
    }

    return result


# ------------------------------------------------------------------
# 5. Predict from an image file path
# ------------------------------------------------------------------
def predict_image_path(model, meta, image_path):
    pil_image = Image.open(image_path)
    result = predict_pil_image(model, meta, pil_image)
    return result


# ------------------------------------------------------------------
# 6. Command-line use
# ------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py path/to/mri_image.jpg")
        sys.exit(1)

    image_path = sys.argv[1]

    if not os.path.exists(image_path):
        print("File not found:", image_path)
        sys.exit(1)

    model, meta = load_artifacts()
    result = predict_image_path(model, meta, image_path)

    print("Image            :", image_path)
    print("Prediction       :", result["label"])
    print("Tumor probability:", result["tumor_probability"])
    print("Threshold used   :", result["threshold"])
