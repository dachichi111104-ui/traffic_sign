import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import logging

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

from src.config import (
    CNN_MODEL_PATH, CLASS_MAPPING_PATH, TRAINING_HISTORY_PATH, MODEL_METADATA_PATH, RESULT_DIR,
    TRAINING_ACCURACY_PATH, TRAINING_LOSS_PATH,
    INPUT_SHAPE, BATCH_SIZE, EPOCHS, LEARNING_RATE, DROPOUT_RATE,
    GTSRB_CLASSES, RANDOM_STATE, TEST_RATIO, VAL_RATIO
)
from src.data_loader import load_raw_dataset
from sklearn.utils.class_weight import compute_class_weight
from src.preprocessing import split_data, resize_image, normalize_pixels, augment_single_image, apply_clahe

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def build_cnn_model(input_shape=INPUT_SHAPE, num_classes=43, dropout_rate=DROPOUT_RATE, learning_rate=LEARNING_RATE):
    """
    Builds Custom CNN architecture with BatchNormalization for Traffic Sign Recognition.
    Block 1: Conv2D -> BatchNorm -> ReLU -> Conv2D -> BatchNorm -> ReLU -> MaxPool -> Dropout
    Block 2: Conv2D -> BatchNorm -> ReLU -> Conv2D -> BatchNorm -> ReLU -> MaxPool -> Dropout
    Block 3: Conv2D -> BatchNorm -> ReLU -> MaxPool -> Dropout
    FC Head: Flatten -> Dense(256) -> BatchNorm -> ReLU -> Dropout -> Dense(Softmax)
    """
    model = models.Sequential([
        # Block 1
        layers.Conv2D(32, (3, 3), padding='same', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(32, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),

        # Block 2
        layers.Conv2D(64, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(64, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),

        # Block 3
        layers.Conv2D(128, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),

        # Fully Connected Classifier Head
        layers.Flatten(),
        layers.Dense(256),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(dropout_rate),
        layers.Dense(num_classes, activation='softmax')
    ], name="Traffic_Sign_Custom_CNN")

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def build_transfer_learning_model(input_shape=INPUT_SHAPE, num_classes=43, learning_rate=LEARNING_RATE):
    """
    Builds MobileNetV2 Transfer Learning Model for Advanced Deep Learning comparisons.
    """
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = True
    # Fine-tune starting from layer 100 onwards
    for layer in base_model.layers[:100]:
        layer.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ], name="Traffic_Sign_MobileNetV2")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate / 10),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def plot_and_save_training_curves(history_dict: dict):
    """
    Plots & saves training history curves:
    - results/training_accuracy.png
    - results/training_loss.png
    - results/accuracy_loss.png
    """
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history_dict['accuracy']) + 1)

    # 1. Plot Accuracy Curve
    plt.figure(figsize=(7, 5))
    plt.plot(epochs, history_dict['accuracy'], 'o-', label='Training Accuracy', color='#1E40AF', linewidth=2)
    plt.plot(epochs, history_dict['val_accuracy'], 's--', label='Validation Accuracy', color='#047857', linewidth=2)
    plt.title('CNN Training vs Validation Accuracy', fontsize=12, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(TRAINING_ACCURACY_PATH, dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Plot Loss Curve
    plt.figure(figsize=(7, 5))
    plt.plot(epochs, history_dict['loss'], 'o-', label='Training Loss', color='#DC2626', linewidth=2)
    plt.plot(epochs, history_dict['val_loss'], 's--', label='Validation Loss', color='#D97706', linewidth=2)
    plt.title('CNN Training vs Validation Loss', fontsize=12, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(TRAINING_LOSS_PATH, dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Plot Combined Side-by-side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(epochs, history_dict['accuracy'], 'o-', label='Training Accuracy', color='#1E40AF', linewidth=2)
    ax1.plot(epochs, history_dict['val_accuracy'], 's--', label='Validation Accuracy', color='#047857', linewidth=2)
    ax1.set_title('Training vs Validation Accuracy', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend(loc='lower right')
    ax1.grid(True, linestyle=':', alpha=0.6)

    ax2.plot(epochs, history_dict['loss'], 'o-', label='Training Loss', color='#DC2626', linewidth=2)
    ax2.plot(epochs, history_dict['val_loss'], 's--', label='Validation Loss', color='#D97706', linewidth=2)
    ax2.set_title('Training vs Validation Loss', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend(loc='upper right')
    ax2.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.savefig(RESULT_DIR / "accuracy_loss.png", dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved training history curves to {RESULT_DIR}")


def train_cnn(dataset_dir: Path = None, retrain: bool = False, model_type: str = "custom"):
    """
    Train CNN model with EarlyStopping, ModelCheckpoint, and ReduceLROnPlateau.
    Dynamically adapts num_classes to any dataset (Vietnamese or GTSRB).
    """
    if CNN_MODEL_PATH.exists() and not retrain:
        logger.info(f"CNN model already exists at {CNN_MODEL_PATH}. Loading existing model.")
        model = models.load_model(CNN_MODEL_PATH)
        with open(TRAINING_HISTORY_PATH, 'r', encoding='utf-8') as f:
            history_dict = json.load(f)
        return model, history_dict

    logger.info("--- Starting CNN Model Training ---")
    
    # 1. Load Dataset & Apply CLAHE
    images_raw, labels, stats = load_raw_dataset(dataset_dir=dataset_dir)
    images_clahe = [apply_clahe(resize_image(img, (32, 32))) for img in images_raw]
    
    num_classes = stats["total_classes"]
    logger.info(f"Dataset has {num_classes} classes.")

    # 2. Strict Train / Val / Test Split (70% / 15% / 15%) BEFORE augmentation
    X_train_raw, X_val_raw, X_test_raw, y_train, y_val, y_test = split_data(
        images_clahe, labels, test_ratio=TEST_RATIO, val_ratio=VAL_RATIO, random_state=RANDOM_STATE
    )

    # Calculate class weights to handle data imbalance
    class_weights_arr = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(enumerate(class_weights_arr))

    # Apply data augmentation STRICTLY to Train set only
    logger.info("Applying data augmentation STRICTLY to training set...")
    X_train_aug = []
    y_train_aug = []
    for img, lbl in zip(X_train_raw, y_train):
        X_train_aug.append(img)
        y_train_aug.append(lbl)
        X_train_aug.append(augment_single_image(img))
        y_train_aug.append(lbl)

    X_train_raw = np.array(X_train_aug)
    y_train = np.array(y_train_aug)

    # Normalize pixel values
    X_train = normalize_pixels(X_train_raw)
    X_val = normalize_pixels(X_val_raw)
    X_test = normalize_pixels(X_test_raw)

    # Save class mapping
    CLASS_MAPPING_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CLASS_MAPPING_PATH, 'w', encoding='utf-8') as f:
        json.dump(GTSRB_CLASSES, f, indent=4, ensure_ascii=False)

    # 3. Build Model (with dynamic num_classes)
    if model_type == "mobilenet":
        logger.info("Building MobileNetV2 Transfer Learning Model...")
        model = build_transfer_learning_model(input_shape=INPUT_SHAPE, num_classes=num_classes)
    else:
        logger.info("Building Custom CNN with BatchNormalization...")
        model = build_cnn_model(input_shape=INPUT_SHAPE, num_classes=num_classes)

    # 4. Callbacks
    CNN_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_cb = callbacks.ModelCheckpoint(
        str(CNN_MODEL_PATH), save_best_only=True, monitor='val_loss', mode='min'
    )
    early_stopping_cb = callbacks.EarlyStopping(
        monitor='val_loss', patience=8, restore_best_weights=True
    )
    reduce_lr_cb = callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6
    )

    # 5. Fit Model with Class Weights & 50 Epochs max
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight_dict,
        callbacks=[checkpoint_cb, early_stopping_cb, reduce_lr_cb],
        verbose=1
    )

    history_dict = {
        'accuracy': [float(x) for x in history.history['accuracy']],
        'val_accuracy': [float(x) for x in history.history['val_accuracy']],
        'loss': [float(x) for x in history.history['loss']],
        'val_loss': [float(x) for x in history.history['val_loss']]
    }

    with open(TRAINING_HISTORY_PATH, 'w', encoding='utf-8') as f:
        json.dump(history_dict, f, indent=4)

    # Save metadata
    metadata = {
        "dataset_path": stats.get("dataset_path"),
        "is_sample_dataset": stats.get("is_sample", False),
        "num_classes": num_classes,
        "trained_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "epochs_run": len(history_dict['accuracy']),
        "final_train_acc": round(history_dict['accuracy'][-1], 4),
        "final_val_acc": round(history_dict['val_accuracy'][-1], 4)
    }
    with open(MODEL_METADATA_PATH, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)

    # Plot & Save curves
    plot_and_save_training_curves(history_dict)

    logger.info("CNN training completed successfully!")
    return model, history_dict


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrain", action="store_true", help="Force retrain CNN model")
    args = parser.parse_args()
    
    train_cnn(retrain=args.retrain)
