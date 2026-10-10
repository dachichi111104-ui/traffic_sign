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


from src.preprocessing import (
    split_data, resize_image, augment_single_image, apply_clahe,
    preprocess_dataset_batch, apply_imbalanced_sampling
)


def train_cnn(
    dataset_dir: Path = None,
    retrain: bool = False,
    model_type: str = "custom",
    use_clahe: bool = True,
    use_gamma: bool = False,
    use_unsharp: bool = False,
    norm_type: str = "minmax",
    sampling_method: str = "none"
):
    """
    Train CNN model with EarlyStopping, ModelCheckpoint, and ReduceLROnPlateau.
    
    Sequential Pipeline:
    1. Load Raw Dataset -> RGB
    2. Split Train (70%) / Val (15%) / Test (15%) BEFORE augmentation & sampling
    3. Preprocess Train set via preprocess_dataset_batch (obtains train_mean & train_std if zscore)
    4. Preprocess Val & Test sets using the exact train_mean & train_std
    5. Apply imbalanced sampling ONLY to Train set
    6. Apply data augmentation ONLY to Train set
    7. Save complete preprocessing_config & train statistics to model_metadata.json
    """
    requested_config = {
        "use_clahe": use_clahe,
        "use_gamma": use_gamma,
        "use_unsharp": use_unsharp,
        "norm_type": norm_type,
        "target_size": [32, 32],
        "sampling_method": sampling_method
    }

    # Requirement 7: Check if existing model configuration matches. Force retrain if config changed.
    if CNN_MODEL_PATH.exists() and MODEL_METADATA_PATH.exists() and not retrain:
        try:
            with open(MODEL_METADATA_PATH, 'r', encoding='utf-8') as f:
                existing_meta = json.load(f)
            saved_config = existing_meta.get("preprocessing_config", {})
            config_changed = any(
                saved_config.get(k) != v for k, v in requested_config.items() if k not in ["train_mean", "train_std"]
            )
            if config_changed:
                logger.warning("Requested preprocessing configuration differs from saved model metadata! Forcing retrain...")
                retrain = True
            else:
                logger.info(f"CNN model already exists at {CNN_MODEL_PATH} with matching configuration. Loading model.")
                model = models.load_model(CNN_MODEL_PATH)
                with open(TRAINING_HISTORY_PATH, 'r', encoding='utf-8') as f:
                    history_dict = json.load(f)
                return model, history_dict
        except Exception as e:
            logger.warning(f"Could not verify existing model metadata: {e}. Retraining...")
            retrain = True

    logger.info("--- Starting CNN Model Training (Strict Sequential Pipeline) ---")
    
    # 1. Load Raw Dataset (RGB)
    images_raw, labels, stats = load_raw_dataset(dataset_dir=dataset_dir)
    num_classes = stats["total_classes"]
    logger.info(f"Loaded raw dataset with {num_classes} classes.")

    # 2. Split Data FIRST (70% Train, 15% Val, 15% Test) to prevent data leakage
    X_train_raw, X_val_raw, X_test_raw, y_train, y_val, y_test = split_data(
        images_raw, labels, test_ratio=TEST_RATIO, val_ratio=VAL_RATIO, random_state=RANDOM_STATE
    )

    # 3. Preprocess Train set first -> calculates train_mean & train_std if Z-score
    logger.info(f"Preprocessing Train batch (CLAHE={use_clahe}, Gamma={use_gamma}, Unsharp={use_unsharp}, Norm={norm_type})...")
    X_train_p, train_mean, train_std = preprocess_dataset_batch(
        X_train_raw,
        use_clahe=use_clahe,
        use_gamma=use_gamma,
        use_unsharp=use_unsharp,
        norm_type=norm_type,
        mean=None,
        std=None
    )

    # 4. Preprocess Val & Test sets using EXACT train_mean & train_std
    X_val, _, _ = preprocess_dataset_batch(
        X_val_raw,
        use_clahe=use_clahe,
        use_gamma=use_gamma,
        use_unsharp=use_unsharp,
        norm_type=norm_type,
        mean=train_mean,
        std=train_std
    )
    X_test, _, _ = preprocess_dataset_batch(
        X_test_raw,
        use_clahe=use_clahe,
        use_gamma=use_gamma,
        use_unsharp=use_unsharp,
        norm_type=norm_type,
        mean=train_mean,
        std=train_std
    )

    # 5. Apply Imbalanced Sampling STRICTLY to Train set ONLY
    if sampling_method != "none":
        logger.info(f"Applying imbalanced sampling ({sampling_method}) STRICTLY to Training set...")
        X_train_p, y_train = apply_imbalanced_sampling(X_train_p, y_train, method=sampling_method)

    # Calculate class weights for loss function
    class_weights_arr = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(enumerate(class_weights_arr))

    # 6. Apply Data Augmentation STRICTLY to Train set ONLY
    logger.info("Applying data augmentation STRICTLY to training set...")
    X_train_aug = []
    y_train_aug = []
    for img, lbl in zip(X_train_p, y_train):
        X_train_aug.append(img)
        y_train_aug.append(lbl)
        X_train_aug.append(augment_single_image(img))
        y_train_aug.append(lbl)

    X_train = np.array(X_train_aug, dtype=np.float32)
    y_train = np.array(y_train_aug, dtype=int)

    # Save class mapping
    CLASS_MAPPING_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CLASS_MAPPING_PATH, 'w', encoding='utf-8') as f:
        json.dump(GTSRB_CLASSES, f, indent=4, ensure_ascii=False)

    # 7. Build Model
    if model_type == "mobilenet":
        logger.info("Building MobileNetV2 Transfer Learning Model...")
        model = build_transfer_learning_model(input_shape=INPUT_SHAPE, num_classes=num_classes)
    else:
        logger.info("Building Custom CNN with BatchNormalization...")
        model = build_cnn_model(input_shape=INPUT_SHAPE, num_classes=num_classes)

    # 8. Callbacks
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

    # 9. Fit Model
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

    # 10. Save Complete Metadata with Preprocessing Config & Statistics
    preprocessing_config = {
        "use_clahe": use_clahe,
        "use_gamma": use_gamma,
        "use_unsharp": use_unsharp,
        "norm_type": norm_type,
        "target_size": [32, 32],
        "train_mean": train_mean,
        "train_std": train_std,
        "sampling_method": sampling_method
    }

    metadata = {
        "dataset_path": stats.get("dataset_path"),
        "is_sample_dataset": stats.get("is_sample", False),
        "num_classes": num_classes,
        "trained_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "epochs_run": len(history_dict['accuracy']),
        "final_train_acc": round(history_dict['accuracy'][-1], 4),
        "final_val_acc": round(history_dict['val_accuracy'][-1], 4),
        "preprocessing_config": preprocessing_config
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
    parser.add_argument("--no-clahe", action="store_true", help="Disable CLAHE")
    parser.add_argument("--use-gamma", action="store_true", help="Apply Gamma Correction")
    parser.add_argument("--use-unsharp", action="store_true", help="Apply Unsharp Masking")
    parser.add_argument("--use-zscore", action="store_true", help="Apply Z-score normalization")
    parser.add_argument("--sampling", type=str, default="none", choices=["none", "random_undersample", "nearmiss", "cluster_centroids"], help="Imbalanced sampling method")
    args = parser.parse_args()
    
    train_cnn(
        retrain=args.retrain,
        use_clahe=not args.no_clahe,
        use_gamma=args.use_gamma,
        use_unsharp=args.use_unsharp,
        norm_type="zscore" if args.use_zscore else "minmax",
        sampling_method=args.sampling
    )
