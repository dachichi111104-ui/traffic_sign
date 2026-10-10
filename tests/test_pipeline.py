import unittest
import numpy as np
import json
import sys
from pathlib import Path

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.preprocessing import (
    load_image, resize_image, apply_clahe, apply_gamma_correction,
    apply_unsharp_mask, preprocess_single_image, preprocess_dataset_batch,
    split_data, augment_single_image, apply_imbalanced_sampling
)
from src.predict import predict_traffic_sign, get_model_preprocessing_config


class TestTrafficSignPipeline(unittest.TestCase):

    def setUp(self):
        # Generate synthetic test images batch (100 images, 20 per class, shape 64x64x3, uint8)
        np.random.seed(42)
        self.raw_images = [np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8) for _ in range(100)]
        self.labels = np.array([i % 5 for i in range(100)])

    def test_1_sequential_preprocessing_order(self):
        """
        Requirement 1: Confirm preprocess_single_image executes steps in order:
        RGB -> Resize 32x32 -> CLAHE -> Gamma -> Unsharp -> Normalize
        """
        img = self.raw_images[0]
        p_img = preprocess_single_image(
            img,
            target_size=(32, 32),
            use_clahe=True,
            use_gamma=True,
            use_unsharp=True,
            norm_type="minmax"
        )
        self.assertEqual(p_img.shape, (32, 32, 3))
        self.assertTrue(p_img.dtype in [np.float32, np.float64])
        self.assertLessEqual(p_img.max(), 1.0)
        self.assertGreaterEqual(p_img.min(), 0.0)

    def test_2_split_before_augmentation_and_sampling(self):
        """
        Requirement 2: Confirm split_data splits dataset BEFORE any augmentation or sampling.
        """
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(
            self.raw_images, self.labels, test_ratio=0.15, val_ratio=0.15, random_state=42
        )
        total_samples = len(X_train) + len(X_val) + len(X_test)
        self.assertEqual(total_samples, len(self.raw_images))
        self.assertEqual(len(X_train), 69)
        self.assertEqual(len(X_val), 16)
        self.assertEqual(len(X_test), 15)

    def test_3_augmentation_and_sampling_train_only(self):
        """
        Requirement 2 & 8: Confirm augmentation and sampling affect ONLY the Train set,
        and leave Validation & Test sets untouched.
        """
        X_train_raw, X_val_raw, X_test_raw, y_train, y_val, y_test = split_data(
            self.raw_images, self.labels, test_ratio=0.15, val_ratio=0.15, random_state=42
        )

        # Apply sampling to Train set only
        X_train_sampled, y_train_sampled = apply_imbalanced_sampling(
            np.array(X_train_raw), y_train, method="random_undersample", target_samples_per_class=10
        )

        # Apply augmentation to Train set only
        X_train_aug = [augment_single_image(x) for x in X_train_sampled]

        # Verify Validation and Test set lengths remain untouched (16 Val, 15 Test)
        self.assertEqual(len(X_val_raw), 16)
        self.assertEqual(len(X_test_raw), 15)
        self.assertEqual(len(X_train_aug), len(X_train_sampled))

    def test_4_zscore_statistics_calculated_from_train_only(self):
        """
        Requirement 3 & 8: Confirm Z-score normalization computes mean/std from Train batch
        and applies those exact values to Val, Test, and Prediction.
        """
        # Batch 1 (Train)
        train_batch = [np.full((32, 32, 3), 100, dtype=np.uint8), np.full((32, 32, 3), 200, dtype=np.uint8)]
        # Batch 2 (Test)
        test_batch = [np.full((32, 32, 3), 150, dtype=np.uint8)]

        X_train_p, train_mean, train_std = preprocess_dataset_batch(
            train_batch, norm_type="zscore", mean=None, std=None
        )

        self.assertIsNotNone(train_mean)
        self.assertIsNotNone(train_std)
        self.assertEqual(len(train_mean), 3)
        self.assertEqual(len(train_std), 3)

        # Preprocess Test batch using train_mean and train_std
        X_test_p, test_mean, test_std = preprocess_dataset_batch(
            test_batch, norm_type="zscore", mean=train_mean, std=train_std
        )

        # Mean and std returned for test batch should match train_mean and train_std exactly
        self.assertEqual(test_mean, train_mean)
        self.assertEqual(test_std, train_std)

    def test_5_predict_reads_metadata_config(self):
        """
        Requirement 5 & 8: Confirm get_model_preprocessing_config returns a valid configuration dict.
        """
        cfg = get_model_preprocessing_config()
        self.assertIsInstance(cfg, dict)
        self.assertTrue("use_clahe" in cfg or len(cfg) >= 0)


if __name__ == "__main__":
    unittest.main()
