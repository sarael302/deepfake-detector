import tensorflow as tf
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

# ============================
# CONFIG
# ============================
MODEL_PATH = "best_xception.h5"
IMG_SIZE = (299, 299)
LAST_CONV_LAYER = "block14_sepconv2_act"
OUTPUT_DIR = "figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================
# LOAD MODEL
# ============================
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# ============================
# IMAGE LOADING
# ============================
def load_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    img = img.astype("float32")
    img = tf.keras.applications.xception.preprocess_input(img)
    return np.expand_dims(img, axis=0)

# ============================
# GRAD-CAM
# ============================
def gradcam(img_path, out_name):
    img = load_image(img_path)

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer(LAST_CONV_LAYER).output,
            model.output
        ]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) + 1e-8

    # Load original image
    original = cv2.imread(img_path)
    original = cv2.resize(original, IMG_SIZE)

    heatmap = cv2.resize(heatmap, IMG_SIZE)
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)

    out_path = os.path.join(OUTPUT_DIR, out_name)
    cv2.imwrite(out_path, overlay)

    print(f"[OK] Grad-CAM saved → {out_path}")

# ============================
# RUN EXAMPLES
# ============================
if __name__ == "__main__":

    real_img = os.path.join("dataset_tiny", "test", "Real", "real_1152.jpg")
    fake_img = os.path.join("dataset_tiny", "test", "Fake", "fake_21386.jpg")

    gradcam(real_img, "gradcam_real.png")
    gradcam(fake_img, "gradcam_fake.png")
