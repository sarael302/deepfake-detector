import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.xception import Xception, preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# ============================================================
# GLOBAL SETTINGS
# ============================================================
FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

# ============================================================
# 1. BUILD MODEL
# ============================================================
def build_model(trainable=False, fine_tune_layers=40):
    base = Xception(include_top=False, weights="imagenet", input_shape=(299, 299, 3))

    if trainable:
        for layer in base.layers[:-fine_tune_layers]:
            layer.trainable = False
        for layer in base.layers[-fine_tune_layers:]:
            layer.trainable = True
    else:
        base.trainable = False

    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.4)(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.3)(x)
    out = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=base.input, outputs=out)

    lr = 1e-4 if not trainable else 1e-5
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
    )

    return model

# ============================================================
# 2. DATA LOADING
# ============================================================
def load_data(data_dir):
    train_gen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        horizontal_flip=True,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        brightness_range=[0.8, 1.2],
        zoom_range=0.2,
        shear_range=10
    )

    val_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train = train_gen.flow_from_directory(
        os.path.join(data_dir, "train"),
        target_size=(299, 299),
        batch_size=16,
        shuffle=True,
        class_mode="binary"
    )

    val = val_gen.flow_from_directory(
        os.path.join(data_dir, "validation"),
        target_size=(299, 299),
        batch_size=16,
        shuffle=False,
        class_mode="binary"
    )

    return train, val

# ============================================================
# 3. TRAINING PIPELINE
# ============================================================
def main(data_dir):
    train, val = load_data(data_dir)

    steps = train.samples // train.batch_size
    val_steps = val.samples // val.batch_size

    print(f"\n➡ Steps per epoch: {steps}")
    print(f"➡ Validation steps: {val_steps}\n")

    ckpt = ModelCheckpoint(
        "best_xception.h5",
        monitor="val_auc",
        save_best_only=True,
        mode="max"
    )

    early = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    # ======================
    # PHASE 1 — WARM-UP
    # ======================
    print("\n===== PHASE 1: WARM-UP TRAINING =====\n")
    model = build_model(trainable=False)

    # Save model summary
    with open(os.path.join(FIG_DIR, "summary.json"), "w") as f:
        model.summary(print_fn=lambda x: f.write(x + "\n"))

    history1 = model.fit(
        train,
        steps_per_epoch=steps,
        validation_data=val,
        validation_steps=val_steps,
        epochs=5,
        callbacks=[ckpt, early]
    )

    # ======================
    # PHASE 2 — FINE-TUNING
    # ======================
    print("\n===== PHASE 2: FINE-TUNING LAST 40 LAYERS =====\n")
    model = build_model(trainable=True, fine_tune_layers=40)

    history2 = model.fit(
        train,
        steps_per_epoch=steps,
        validation_data=val,
        validation_steps=val_steps,
        epochs=10,
        callbacks=[ckpt, early]
    )

    # ============================================================
    # 4. MERGE HISTORIES
    # ============================================================
    history = {}
    for k in history1.history:
        history[k] = history1.history[k] + history2.history[k]

    with open(os.path.join(FIG_DIR, "training_history.json"), "w") as f:
        json.dump(history, f, indent=4)

    # ============================================================
    # 5. PLOTS (IEEE / PhD QUALITY)
    # ============================================================

    # Accuracy
    plt.figure()
    plt.plot(history["accuracy"], label="Train Accuracy")
    plt.plot(history["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(FIG_DIR, "accuracy_curve.png"), dpi=300)
    plt.close()

    # Loss
    plt.figure()
    plt.plot(history["loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(FIG_DIR, "loss_curve.png"), dpi=300)
    plt.close()

    # ============================================================
    # 6. ROC CURVE
    # ============================================================
    y_true = val.classes
    y_pred = model.predict(val, verbose=1).ravel()

    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(FIG_DIR, "roc_curve.png"), dpi=300)
    plt.close()

    print("\n✔ Training completed")
    print("✔ Best model saved as best_xception.h5")
    print("✔ Figures & JSON saved in ./figures/")

# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    args = parser.parse_args()
    main(args.data_dir)
