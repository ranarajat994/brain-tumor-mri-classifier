"""
train.py - Brain Tumor MRI: binary classification (no_tumor vs tumor) with a CNN

Expected folder layout (created by the merge step in Colab):
brain_tumor_project/
    train.py
    data/
        Training/
            no_tumor/
            tumor/
        Testing/
            no_tumor/
            tumor/

Run in Colab (GPU runtime):
    !cd /content/brain_tumor_project && python train.py
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ------------------------------------------------------------------
# 1. Paths
# ------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(BASE_DIR, "data", "Training")
TEST_DIR = os.path.join(BASE_DIR, "data", "Testing")

MODEL_DIR = os.path.join(BASE_DIR, "models")
REPORT_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "brain_tumor_cnn.keras")
META_PATH = os.path.join(MODEL_DIR, "model_meta.json")
METRICS_PATH = os.path.join(REPORT_DIR, "test_metrics.json")
CURVES_PATH = os.path.join(REPORT_DIR, "training_curves.png")
CM_PATH = os.path.join(REPORT_DIR, "confusion_matrix.png")


# ------------------------------------------------------------------
# 2. Settings
# ------------------------------------------------------------------
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 30
SEED = 42
THRESHOLD = 0.5

# Fixed order -> no_tumor = 0, tumor = 1
CLASS_NAMES = ["no_tumor", "tumor"]

tf.keras.utils.set_random_seed(SEED)

print("GPU available:", tf.config.list_physical_devices("GPU"))


# ------------------------------------------------------------------
# 3. Load datasets
#    Training folder -> 80% train, 20% validation
#    Testing folder  -> untouched final test set
# ------------------------------------------------------------------
train_ds = keras.utils.image_dataset_from_directory(
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
    subset="training",
)

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
    shuffle=False,  # keep order so predictions line up with labels
)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)


# ------------------------------------------------------------------
# 4. Class weights (tumor images outnumber no_tumor images)
#    Balanced weights stop the model from simply predicting "tumor"
#    for everything.
# ------------------------------------------------------------------
n_no_tumor = len(os.listdir(os.path.join(TRAIN_DIR, "no_tumor")))
n_tumor = len(os.listdir(os.path.join(TRAIN_DIR, "tumor")))
n_total = n_no_tumor + n_tumor

weight_no_tumor = n_total / (2 * n_no_tumor)
weight_tumor = n_total / (2 * n_tumor)
class_weight = {0: weight_no_tumor, 1: weight_tumor}

print("Training images -> no_tumor:", n_no_tumor, "| tumor:", n_tumor)
print("Class weights:", class_weight)


# ------------------------------------------------------------------
# 5. Data augmentation (active only during training)
# ------------------------------------------------------------------
data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.10),
        layers.RandomContrast(0.10),
    ],
    name="augmentation",
)


# ------------------------------------------------------------------
# 6. CNN model
#    Conv -> BatchNorm -> ReLU -> MaxPool, four blocks
# ------------------------------------------------------------------
model = keras.Sequential(
    [
        keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
        data_augmentation,
        layers.Rescaling(1.0 / 255),

        # Block 1
        layers.Conv2D(32, (3, 3), padding="same", use_bias=False),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D((2, 2)),

        # Block 2
        layers.Conv2D(64, (3, 3), padding="same", use_bias=False),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D((2, 2)),

        # Block 3
        layers.Conv2D(128, (3, 3), padding="same", use_bias=False),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D((2, 2)),

        # Block 4
        layers.Conv2D(256, (3, 3), padding="same", use_bias=False),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D((2, 2)),

        # Head
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.4),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid"),  # probability of "tumor"
    ],
    name="brain_tumor_cnn",
)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        keras.metrics.Precision(name="precision"),
        keras.metrics.Recall(name="recall"),
        keras.metrics.AUC(name="auc"),
    ],
)

model.summary()


# ------------------------------------------------------------------
# 7. Callbacks
# ------------------------------------------------------------------
early_stop = keras.callbacks.EarlyStopping(
    monitor="val_auc",
    mode="max",
    patience=6,
    restore_best_weights=True,
    verbose=1,
)

checkpoint = keras.callbacks.ModelCheckpoint(
    MODEL_PATH,
    monitor="val_auc",
    mode="max",
    save_best_only=True,
    verbose=1,
)

reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=2,
    min_lr=1e-6,
    verbose=1,
)


# ------------------------------------------------------------------
# 8. Train
# ------------------------------------------------------------------
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weight,
    callbacks=[early_stop, checkpoint, reduce_lr],
)


# ------------------------------------------------------------------
# 9. Training curves
# ------------------------------------------------------------------
hist = history.history

plt.figure(figsize=(15, 4))

plt.subplot(1, 3, 1)
plt.plot(hist["accuracy"], label="train")
plt.plot(hist["val_accuracy"], label="validation")
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(hist["loss"], label="train")
plt.plot(hist["val_loss"], label="validation")
plt.title("Loss")
plt.xlabel("Epoch")
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(hist["recall"], label="train")
plt.plot(hist["val_recall"], label="validation")
plt.title("Recall (tumor)")
plt.xlabel("Epoch")
plt.legend()

plt.tight_layout()
plt.savefig(CURVES_PATH, dpi=120)
plt.close()
print("Saved training curves ->", CURVES_PATH)


# ------------------------------------------------------------------
# 10. Evaluate on the untouched test set
# ------------------------------------------------------------------
best_model = keras.models.load_model(MODEL_PATH)

y_prob = best_model.predict(test_ds).ravel()
y_true = np.concatenate(list(test_ds.map(lambda x, y: y).as_numpy_iterator())).ravel().astype(int)
y_pred = (y_prob >= THRESHOLD).astype(int)

report_text = classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4)
report_dict = classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True)
test_auc = roc_auc_score(y_true, y_prob)

print("\n===== TEST SET RESULTS =====")
print(report_text)
print("ROC-AUC:", round(test_auc, 4))

cm = confusion_matrix(y_true, y_pred)
tn = int(cm[0, 0])
fp = int(cm[0, 1])
fn = int(cm[1, 0])
tp = int(cm[1, 1])

print("True negatives  (no_tumor correctly found):", tn)
print("False positives (no_tumor called tumor):   ", fp)
print("False negatives (tumor MISSED):            ", fn)
print("True positives  (tumor correctly found):   ", tp)


# ------------------------------------------------------------------
# 11. Confusion matrix plot
# ------------------------------------------------------------------
plt.figure(figsize=(5, 4))
plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix (Test)")
plt.xticks([0, 1], CLASS_NAMES)
plt.yticks([0, 1], CLASS_NAMES)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.text(0, 0, str(tn), ha="center", va="center", fontsize=14)
plt.text(1, 0, str(fp), ha="center", va="center", fontsize=14)
plt.text(0, 1, str(fn), ha="center", va="center", fontsize=14)
plt.text(1, 1, str(tp), ha="center", va="center", fontsize=14)
plt.colorbar()
plt.tight_layout()
plt.savefig(CM_PATH, dpi=120)
plt.close()
print("Saved confusion matrix ->", CM_PATH)


# ------------------------------------------------------------------
# 12. Save metrics + metadata (predict.py and app.py will read this)
# ------------------------------------------------------------------
metrics = {
    "accuracy": report_dict["accuracy"],
    "tumor_precision": report_dict["tumor"]["precision"],
    "tumor_recall": report_dict["tumor"]["recall"],
    "tumor_f1": report_dict["tumor"]["f1-score"],
    "no_tumor_recall": report_dict["no_tumor"]["recall"],
    "roc_auc": float(test_auc),
    "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    "threshold": THRESHOLD,
}

with open(METRICS_PATH, "w") as f:
    json.dump(metrics, f, indent=4)

meta = {
    "class_names": CLASS_NAMES,
    "img_size": list(IMG_SIZE),
    "color_mode": "rgb",
    "threshold": THRESHOLD,
    "positive_class": "tumor",
}

with open(META_PATH, "w") as f:
    json.dump(meta, f, indent=4)

print("Saved metrics  ->", METRICS_PATH)
print("Saved metadata ->", META_PATH)
print("Saved model    ->", MODEL_PATH)
