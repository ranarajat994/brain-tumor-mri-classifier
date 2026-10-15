"""
tune_threshold.py - pick a better decision threshold for the brain tumor CNN

Why: at threshold 0.5 the model misses 176 of 1200 tumors, but ROC-AUC is 0.978,
so the model ranks images well; only the cut-off is too strict.

Rule: choose the threshold on the VALIDATION set (never on the test set),
then check it once on the test set.

Run in Colab (same folder as train.py):
    !cd /content/brain_tumor_project && python tune_threshold.py
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import precision_recall_curve, classification_report, confusion_matrix


# ------------------------------------------------------------------
# 1. Paths + settings (must match train.py)
# ------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(BASE_DIR, "data", "Training")
TEST_DIR = os.path.join(BASE_DIR, "data", "Testing")
MODEL_PATH = os.path.join(BASE_DIR, "models", "brain_tumor_cnn.keras")
META_PATH = os.path.join(BASE_DIR, "models", "model_meta.json")
METRICS_PATH = os.path.join(BASE_DIR, "reports", "test_metrics_tuned.json")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42
CLASS_NAMES = ["no_tumor", "tumor"]

TARGET_RECALL = 0.98  # we want to catch at least 98% of tumors on validation


# ------------------------------------------------------------------
# 2. Rebuild the SAME validation split as train.py (same seed + split)
# ------------------------------------------------------------------
val_ds = keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels="inferred",
    label_mode="binary",
    class_names=CLASS_NAMES,
    color_mode="rgb",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED,
    validation_split=0.2,
    subset="validation",
)

test_ds = keras.utils.image_dataset_from_directory(
    TEST_DIR,
    labels="inferred",
    label_mode="binary",
    class_names=CLASS_NAMES,
    color_mode="rgb",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

model = keras.models.load_model(MODEL_PATH)


# ------------------------------------------------------------------
# 3. Validation predictions
#    val_ds reshuffles every time it is read, so images and labels must be
#    collected in ONE pass. This is the one place a loop is unavoidable.
# ------------------------------------------------------------------
val_prob_batches = []
val_label_batches = []
for batch_images, batch_labels in val_ds:
    val_prob_batches.append(model.predict(batch_images, verbose=0).ravel())
    val_label_batches.append(batch_labels.numpy().ravel())

val_prob = np.concatenate(val_prob_batches)
val_true = np.concatenate(val_label_batches).astype(int)


# ------------------------------------------------------------------
# 4. Find the highest threshold that still gives recall >= TARGET_RECALL
# ------------------------------------------------------------------
precision, recall, thresholds = precision_recall_curve(val_true, val_prob)

# precision/recall have one more value than thresholds -> drop the last one
recall_t = recall[:-1]
precision_t = precision[:-1]

ok_mask = recall_t >= TARGET_RECALL
best_threshold = float(thresholds[ok_mask].max())

best_index = int(np.where(thresholds == best_threshold)[0][0])
print("Chosen threshold:", round(best_threshold, 4))
print("Validation recall at this threshold:   ", round(float(recall_t[best_index]), 4))
print("Validation precision at this threshold:", round(float(precision_t[best_index]), 4))


# ------------------------------------------------------------------
# 5. Check once on the test set: old 0.5 vs new threshold
# ------------------------------------------------------------------
test_prob = model.predict(test_ds).ravel()
test_true = np.concatenate(list(test_ds.map(lambda x, y: y).as_numpy_iterator())).ravel().astype(int)

pred_old = (test_prob >= 0.5).astype(int)
pred_new = (test_prob >= best_threshold).astype(int)

print("\n===== TEST @ 0.5 =====")
print(classification_report(test_true, pred_old, target_names=CLASS_NAMES, digits=4))
print(confusion_matrix(test_true, pred_old))

print("\n===== TEST @", round(best_threshold, 4), "=====")
print(classification_report(test_true, pred_new, target_names=CLASS_NAMES, digits=4))

cm = confusion_matrix(test_true, pred_new)
tn = int(cm[0, 0])
fp = int(cm[0, 1])
fn = int(cm[1, 0])
tp = int(cm[1, 1])
print("True negatives :", tn)
print("False positives:", fp)
print("False negatives (tumor MISSED):", fn)
print("True positives :", tp)


# ------------------------------------------------------------------
# 6. Save the new threshold so predict.py / app.py use it
# ------------------------------------------------------------------
report_dict = classification_report(test_true, pred_new, target_names=CLASS_NAMES, output_dict=True)

tuned_metrics = {
    "threshold": best_threshold,
    "chosen_on": "validation set, target tumor recall " + str(TARGET_RECALL),
    "accuracy": report_dict["accuracy"],
    "tumor_precision": report_dict["tumor"]["precision"],
    "tumor_recall": report_dict["tumor"]["recall"],
    "no_tumor_recall": report_dict["no_tumor"]["recall"],
    "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
}

with open(METRICS_PATH, "w") as f:
    json.dump(tuned_metrics, f, indent=4)

with open(META_PATH, "r") as f:
    meta = json.load(f)

meta["threshold"] = best_threshold

with open(META_PATH, "w") as f:
    json.dump(meta, f, indent=4)

print("\nSaved tuned metrics ->", METRICS_PATH)
print("Updated threshold in ->", META_PATH)
