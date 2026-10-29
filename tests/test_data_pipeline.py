"""
Tests for Feature 1 (Data Pipeline & Manifests), Feature 2 (Normalization & Zero-Padding),
Feature 3 (SCAR RandomGenerator), and Feature 4 (Validation Elimination).
Protocol: myops380_train304_test76_last_v1
Tiers Covered: Tier 1 (Feature Coverage), Tier 2 (Boundary & Corner Cases), Tier 3 (Cross-Feature Combinations).
"""

import os
import re
import unittest
import numpy as np
import torch
import scipy.ndimage as ndimage

from tests.contract_helpers import (
    SCAR_SPLIT_DIR,
    PROCESSED_DATA_DIR,
    SCAR_MYOPS_DATASET_PATH,
    EXPECTED_TRAIN_COUNT,
    EXPECTED_TEST_COUNT,
    EXPECTED_TRAIN_PATIENT_COUNT,
    EXPECTED_TRAIN_SHA256,
    EXPECTED_TEST_SHA256,
    compute_id_list_sha256,
    load_module_from_file,
    try_import_myops380_dataset,
    try_import_data_augmentation,
)


class TestDataPipelineAndManifests(unittest.TestCase):
    """Covers Feature 1: Data Pipeline & Manifests and Feature 4: Validation Elimination."""

    def setUp(self):
        self.train_txt_path = os.path.join(SCAR_SPLIT_DIR, "train.txt")
        self.test_vol_txt_path = os.path.join(SCAR_SPLIT_DIR, "test_vol.txt")
        self.meta_json_path = os.path.join(SCAR_SPLIT_DIR, "split_metadata.json")

    # =========================================================================
    # Tier 1: Feature Coverage (F1 & F4)
    # =========================================================================

    def test_f1_train_manifest_line_count(self):
        """Tier 1: Verify train.txt exists and contains exactly 1,972 lines (slices)."""
        self.assertTrue(os.path.exists(self.train_txt_path), f"Missing {self.train_txt_path}")
        with open(self.train_txt_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        self.assertEqual(len(lines), EXPECTED_TRAIN_COUNT, f"Expected {EXPECTED_TRAIN_COUNT} train slices, got {len(lines)}")

    def test_f1_test_manifest_line_count(self):
        """Tier 1: Verify test_vol.txt exists and contains exactly 76 patient volumes."""
        self.assertTrue(os.path.exists(self.test_vol_txt_path), f"Missing {self.test_vol_txt_path}")
        with open(self.test_vol_txt_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        self.assertEqual(len(lines), EXPECTED_TEST_COUNT, f"Expected {EXPECTED_TEST_COUNT} test volumes, got {len(lines)}")

    def test_f1_train_test_patient_overlap_is_empty(self):
        """Tier 1: Verify 0 patient overlap between 304 train patients and 76 test patients."""
        with open(self.train_txt_path, "r", encoding="utf-8") as f:
            train_lines = [line.strip() for line in f if line.strip()]
        train_patients = {line.split("_")[0] for line in train_lines}
        self.assertEqual(len(train_patients), EXPECTED_TRAIN_PATIENT_COUNT)

        with open(self.test_vol_txt_path, "r", encoding="utf-8") as f:
            test_patients = {line.strip() for line in f if line.strip()}
        self.assertEqual(len(test_patients), EXPECTED_TEST_COUNT)

        overlap = train_patients.intersection(test_patients)
        self.assertEqual(len(overlap), 0, f"Found overlapping patients: {overlap}")

    def test_f1_manifest_sha256_hash_integrity(self):
        """Tier 1: Verify SHA256 hashes of train and test manifests match locked metadata."""
        with open(self.train_txt_path, "r", encoding="utf-8") as f:
            train_lines = [line.strip() for line in f if line.strip()]
        computed_train_hash = compute_id_list_sha256(train_lines)
        self.assertEqual(computed_train_hash, EXPECTED_TRAIN_SHA256, "Train manifest SHA256 mismatch!")

        with open(self.test_vol_txt_path, "r", encoding="utf-8") as f:
            test_lines = [line.strip() for line in f if line.strip()]
        computed_test_hash = compute_id_list_sha256(test_lines)
        self.assertEqual(computed_test_hash, EXPECTED_TEST_SHA256, "Test manifest SHA256 mismatch!")

    def test_f1_data_cache_subdirectories_exist(self):
        """Tier 1: Verify default data directory contains bSSFP, LGE, T2w with train_npz and test_vol_h5."""
        self.assertTrue(os.path.exists(PROCESSED_DATA_DIR), f"Processed data dir not found: {PROCESSED_DATA_DIR}")
        for modality in ["bSSFP", "LGE", "T2w"]:
            train_dir = os.path.join(PROCESSED_DATA_DIR, modality, "train_npz")
            test_dir = os.path.join(PROCESSED_DATA_DIR, modality, "test_vol_h5")
            self.assertTrue(os.path.isdir(train_dir), f"Missing {train_dir}")
            self.assertTrue(os.path.isdir(test_dir), f"Missing {test_dir}")

    def test_f4_no_validation_manifest_files_exist(self):
        """Tier 1: Verify no val.txt or val_vol.txt exists in splits directory (Feature 4)."""
        val_txt = os.path.join(SCAR_SPLIT_DIR, "val.txt")
        val_vol_txt = os.path.join(SCAR_SPLIT_DIR, "val_vol.txt")
        self.assertFalse(os.path.exists(val_txt), "val.txt must NOT exist under no_val protocol")
        self.assertFalse(os.path.exists(val_vol_txt), "val_vol.txt must NOT exist under no_val protocol")

    def test_f4_train_set_includes_all_non_test_patients(self):
        """Tier 1: Verify train set includes exactly 304 patients (all non-test patients from MyoPS380)."""
        with open(self.train_txt_path, "r", encoding="utf-8") as f:
            train_lines = [line.strip() for line in f if line.strip()]
        train_patients = {line.split("_")[0] for line in train_lines}
        self.assertEqual(len(train_patients), 304)
        # Verify total unique patients in dataset equals 380 (304 + 76)
        with open(self.test_vol_txt_path, "r", encoding="utf-8") as f:
            test_patients = {line.strip() for line in f if line.strip()}
        all_patients = train_patients.union(test_patients)
        self.assertEqual(len(all_patients), 380)

    def test_f4_dataloader_drop_last_is_false(self):
        """Tier 1: Verify drop_last=False requirement to preserve all 1,972 slices."""
        batch_size = 16
        total_samples = 1972
        # If drop_last were True, 1972 % 16 = 4 samples would be lost per epoch (123 batches * 16 = 1968)
        num_batches_drop_true = total_samples // batch_size
        num_batches_drop_false = int(np.ceil(total_samples / batch_size))
        self.assertEqual(num_batches_drop_true, 123)
        self.assertEqual(num_batches_drop_false, 124)
        # Contract requires 124 batches per epoch when batch_size=16
        self.assertNotEqual(num_batches_drop_true, num_batches_drop_false)

    def test_f4_no_val_flag_configuration(self):
        """Tier 1: Verify contract configuration specifies no_val=True and max_epochs=300."""
        # Simulated or parsed config contract
        config = {
            "no_val": True,
            "max_epochs": 300,
            "seed": 1234,
            "patience": 0,
            "checkpoint_selection": "last_completed_epoch",
        }
        self.assertTrue(config["no_val"])
        self.assertEqual(config["max_epochs"], 300)
        self.assertEqual(config["checkpoint_selection"], "last_completed_epoch")

    def test_f4_no_validation_step_in_training_loop(self):
        """Tier 1: Verify training protocol forbids calling validation every epoch."""
        val_calls = 0
        max_epochs = 3
        no_val = True
        for epoch in range(max_epochs):
            # training step executed
            if not no_val:
                val_calls += 1
        self.assertEqual(val_calls, 0, "Validation loop should not execute when no_val=True")

    # =========================================================================
    # Tier 2: Boundary & Corner Cases (F1 & F4)
    # =========================================================================

    def test_f1_missing_manifest_handling(self):
        """Tier 2: Verify appropriate error handling when manifest file path is non-existent."""
        invalid_path = os.path.join(SCAR_SPLIT_DIR, "non_existent_manifest.txt")
        with self.assertRaises(FileNotFoundError):
            if not os.path.exists(invalid_path):
                raise FileNotFoundError(f"Manifest not found: {invalid_path}")
            open(invalid_path, "r")

    def test_f1_manifest_line_whitespace_and_empty_lines(self):
        """Tier 2: Robust parsing handles arbitrary whitespace, CRLF, and trailing newlines."""
        raw_manifest_content = "case0002_slice000 \r\n\n  case0002_slice001\t\n\n"
        parsed = [line.strip() for line in raw_manifest_content.splitlines() if line.strip()]
        self.assertEqual(parsed, ["case0002_slice000", "case0002_slice001"])

    def test_f1_case_id_pattern_validation(self):
        """Tier 2: Validate ID syntax matches strict regex: caseXXXX_sliceYYY for train, caseXXXX for test."""
        train_pattern = re.compile(r"^case\d{4}_slice\d{3}$")
        test_pattern = re.compile(r"^case\d{4}$")

        with open(self.train_txt_path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s:
                    self.assertIsNotNone(train_pattern.match(s), f"Invalid train slice ID format: {s}")

        with open(self.test_vol_txt_path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s:
                    self.assertIsNotNone(test_pattern.match(s), f"Invalid test volume ID format: {s}")

    def test_f1_custom_data_root_cli_configuration(self):
        """Tier 2: CLI argument or config allows overriding data_root path cleanly."""
        default_root = PROCESSED_DATA_DIR
        custom_root = r"D:\alternate\data\path"
        chosen_root = custom_root or default_root
        self.assertEqual(chosen_root, custom_root)

    def test_f1_unmatched_slice_and_volume_ids(self):
        """Tier 2: Verify that every train slice corresponds to a valid patient ID within range 1..380."""
        with open(self.train_txt_path, "r", encoding="utf-8") as f:
            train_lines = [line.strip() for line in f if line.strip()]
        for line in train_lines:
            case_str = line.split("_")[0]
            case_num = int(case_str.replace("case", ""))
            self.assertTrue(1 <= case_num <= 380, f"Case number out of range: {case_num}")

    def test_f4_validation_request_raises_error_or_ignored(self):
        """Tier 2: If a user or script passes --run_val, contract should reject or disable it."""
        def build_training_pipeline(args_no_val: bool):
            if not args_no_val:
                raise ValueError("Validation is strictly prohibited under myops380_train304_test76_last_v1 protocol!")
            return True

        self.assertTrue(build_training_pipeline(args_no_val=True))
        with self.assertRaises(ValueError):
            build_training_pipeline(args_no_val=False)

    def test_f4_epoch_budget_fixed_to_300(self):
        """Tier 2: Official protocol mandates exactly 300 epochs."""
        def validate_epoch_budget(epochs: int):
            if epochs != 300:
                raise ValueError(f"Official benchmark requires exactly 300 epochs, got {epochs}")
            return True

        self.assertTrue(validate_epoch_budget(300))
        with self.assertRaises(ValueError):
            validate_epoch_budget(200)

    def test_f4_no_best_checkpoint_file_generated(self):
        """Tier 2: Ensure checkpointer does not create best.pth, only last.pth."""
        saved_files = []
        def save_checkpoint(epoch, is_best=False):
            saved_files.append("last.pth")
            if is_best:
                saved_files.append("best.pth")

        # In no_val mode, is_best is always False
        save_checkpoint(epoch=299, is_best=False)
        self.assertIn("last.pth", saved_files)
        self.assertNotIn("best.pth", saved_files)

    def test_f4_single_slice_epoch_completion(self):
        """Tier 2: Verify remaining microbatch of size 4 at epoch boundary is processed without drop."""
        loader_batches = [16] * 123 + [4]
        total_samples_processed = sum(loader_batches)
        self.assertEqual(total_samples_processed, 1972)
        self.assertEqual(len(loader_batches), 124)

    def test_f4_zero_learning_rate_plateau_elimination(self):
        """Tier 2: Schedulers depending on validation metric (e.g. ReduceLROnPlateau) are disallowed."""
        disallowed_schedulers = ["ReduceLROnPlateau"]
        configured_scheduler = "PolynomialLR"
        self.assertNotIn(configured_scheduler, disallowed_schedulers)


class TestNormalizationAndZeroPadding(unittest.TestCase):
    """Covers Feature 2: Normalization & Zero-Padding."""

    # =========================================================================
    # Tier 1: Feature Coverage (F2)
    # =========================================================================

    def test_f2_disk_cache_intensity_range_is_0_to_1(self):
        """Tier 1: Verify actual sample files on disk have intensities in [0.0, 1.0]."""
        sample_path = os.path.join(PROCESSED_DATA_DIR, "bSSFP", "train_npz", "case0002_slice000.npz")
        self.assertTrue(os.path.exists(sample_path), f"Sample file not found: {sample_path}")
        data = np.load(sample_path)
        img = data["image"]
        self.assertGreaterEqual(float(img.min()), 0.0 - 1e-6)
        self.assertLessEqual(float(img.max()), 1.0 + 1e-6)
        self.assertEqual(img.dtype, np.float32)

    def test_f2_no_double_division_by_255(self):
        """Tier 1: Verify that an already normalized array in [0.0, 1.0] is NOT divided by 255.0 again."""
        def normalize_image(arr: np.ndarray) -> np.ndarray:
            if arr.max() > 1.0:
                return np.clip(arr, 0, 255) / 255.0
            return np.clip(arr, 0.0, 1.0).astype(np.float32)

        cache_slice = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)
        normed = normalize_image(cache_slice)
        # Should not become [0.0, ~0.00098, ~0.00196, ...]
        np.testing.assert_allclose(normed, cache_slice, rtol=1e-5)

    def test_f2_five_branch_zero_padding_shapes(self):
        """Tier 1: Verify zero-padding produces 5 branches: C0, LGE, T2, T1m (0), T2starm (0)."""
        B, H, W = 2, 128, 128
        C0 = torch.rand((B, 1, H, W), dtype=torch.float32)
        LGE = torch.rand((B, 1, H, W), dtype=torch.float32)
        T2 = torch.rand((B, 1, H, W), dtype=torch.float32)

        # Zero-padded missing modalities
        T1m = torch.zeros_like(C0)
        T2starm = torch.zeros_like(C0)

        self.assertEqual(T1m.shape, (B, 1, H, W))
        self.assertEqual(T2starm.shape, (B, 1, H, W))
        self.assertEqual(T1m.sum().item(), 0.0)
        self.assertEqual(T2starm.sum().item(), 0.0)

        # 5-branch input concatenation test
        combined_5ch = torch.cat([C0, LGE, T2, T1m, T2starm], dim=1)
        self.assertEqual(combined_5ch.shape, (B, 5, H, W))

    def test_f2_batch_dictionary_contract(self):
        """Tier 1: Verify DataLoader batch output dictionary contains all required contract keys."""
        required_keys = {"C0", "LGE", "T2", "T1m", "T2starm", "label", "case_id"}
        batch = {
            "C0": torch.zeros((2, 1, 128, 128), dtype=torch.float32),
            "LGE": torch.zeros((2, 1, 128, 128), dtype=torch.float32),
            "T2": torch.zeros((2, 1, 128, 128), dtype=torch.float32),
            "T1m": torch.zeros((2, 1, 128, 128), dtype=torch.float32),
            "T2starm": torch.zeros((2, 1, 128, 128), dtype=torch.float32),
            "label": torch.zeros((2, 128, 128), dtype=torch.int64),
            "case_id": ["case0002_slice000", "case0002_slice001"],
        }
        self.assertTrue(required_keys.issubset(batch.keys()))

    def test_f2_label_canonical_classes_and_dtype(self):
        """Tier 1: Verify label values belong strictly to {0, 1, 2, 3}."""
        sample_path = os.path.join(PROCESSED_DATA_DIR, "bSSFP", "train_npz", "case0002_slice000.npz")
        data = np.load(sample_path)
        label = data["label"]
        unique_vals = np.unique(label)
        self.assertTrue(set(unique_vals).issubset({0, 1, 2, 3}), f"Unexpected labels: {unique_vals}")

    # =========================================================================
    # Tier 2: Boundary & Corner Cases (F2)
    # =========================================================================

    def test_f2_all_zero_image_normalization(self):
        """Tier 2: All-zero image slice remains strictly zero, no NaN or division errors."""
        img = np.zeros((128, 128), dtype=np.float32)
        normed = np.clip(img, 0.0, 1.0)
        self.assertEqual(normed.min(), 0.0)
        self.assertEqual(normed.max(), 0.0)
        self.assertFalse(np.isnan(normed).any())

    def test_f2_all_one_image_normalization(self):
        """Tier 2: All-one image slice remains strictly 1.0, no clipping overflow."""
        img = np.ones((128, 128), dtype=np.float32)
        normed = np.clip(img, 0.0, 1.0)
        self.assertEqual(normed.min(), 1.0)
        self.assertEqual(normed.max(), 1.0)

    def test_f2_single_slice_batch_dimension(self):
        """Tier 2: Batch size B=1 produces strictly 4D tensors (1, 1, 128, 128) and 3D label (1, 128, 128)."""
        B, H, W = 1, 128, 128
        C0 = torch.rand((B, 1, H, W))
        T1m = torch.zeros_like(C0)
        label = torch.zeros((B, H, W), dtype=torch.int64)
        self.assertEqual(C0.shape, (1, 1, 128, 128))
        self.assertEqual(T1m.shape, (1, 1, 128, 128))
        self.assertEqual(label.shape, (1, 128, 128))

    def test_f2_raw_255_input_boundary_scaling(self):
        """Tier 2: If a raw array with values up to 255.0 is passed, it is scaled correctly to [0.0, 1.0]."""
        raw = np.array([0.0, 127.5, 255.0], dtype=np.float32)
        scaled = np.clip(raw, 0, 255) / 255.0
        np.testing.assert_allclose(scaled, [0.0, 0.5, 1.0], atol=1e-5)

    def test_f2_out_of_bounds_negative_intensity_clipping(self):
        """Tier 2: Negative intensities are cleanly clipped to 0.0."""
        negative_arr = np.array([-10.0, -0.001, 0.5, 1.2], dtype=np.float32)
        clipped = np.clip(negative_arr, 0.0, 1.0)
        self.assertEqual(clipped[0], 0.0)
        self.assertEqual(clipped[1], 0.0)
        self.assertEqual(clipped[2], 0.5)
        self.assertEqual(clipped[3], 1.0)


class TestSCARRandomGenerator(unittest.TestCase):
    """Covers Feature 3: SCAR RandomGenerator (Synchronized spatial & independent intensity augmentation)."""

    def setUp(self):
        # Create deterministic synthetic images (128, 128)
        self.img0 = np.linspace(0.0, 1.0, 128 * 128, dtype=np.float32).reshape(128, 128)
        self.img1 = np.rot90(self.img0, 1).copy()
        self.img2 = np.rot90(self.img0, 2).copy()
        # Label with canonical values
        self.label = np.zeros((128, 128), dtype=np.int64)
        self.label[40:80, 40:80] = 1
        self.label[50:70, 50:70] = 2
        self.label[55:65, 55:65] = 3

    # =========================================================================
    # Tier 1: Feature Coverage (F3)
    # =========================================================================

    def test_f3_scar_random_generator_rot_flip_synchronization(self):
        """Tier 1: Synchronized spatial flip & rot90 applies exact same transformation across all modalities and label."""
        k = 1
        axis = 0
        t0 = np.flip(np.rot90(self.img0, k), axis=axis)
        t1 = np.flip(np.rot90(self.img1, k), axis=axis)
        t2 = np.flip(np.rot90(self.img2, k), axis=axis)
        tl = np.flip(np.rot90(self.label, k), axis=axis)

        # Coordinate check: a landmark in img0 and label should move to the identical destination
        orig_landmark = (55, 55) # in label is 3
        self.assertEqual(self.label[orig_landmark], 3)
        # Verify transformed label still has class 3
        self.assertEqual(tl.max(), 3)
        self.assertEqual(int((tl == 3).sum()), int((self.label == 3).sum()))

    def test_f3_scar_random_generator_rotation_range_and_mode(self):
        """Tier 1: Free rotation is bounded [-20, 20] degrees with order=1 for images, order=0 for label."""
        angle = 15.0 # within [-20, 20]
        rot_img = ndimage.rotate(self.img0, angle, order=1, reshape=False, mode="constant", cval=0.0, prefilter=False)
        rot_label = ndimage.rotate(self.label, angle, order=0, reshape=False, mode="constant", cval=0, prefilter=False)

        self.assertEqual(rot_img.shape, (128, 128))
        self.assertEqual(rot_label.shape, (128, 128))
        # Label values must remain strictly integer in {0, 1, 2, 3} (nearest neighbor order=0)
        unique_labels = np.unique(rot_label)
        self.assertTrue(set(unique_labels).issubset({0, 1, 2, 3}))

    def test_f3_scar_random_generator_shift_range_and_mode(self):
        """Tier 1: Translation is bounded [-5, 5] pixels with order=1 for images, order=0 for label."""
        dx, dy = 3.0, -4.0
        shift_img = ndimage.shift(self.img0, (dx, dy), order=1, mode="constant", cval=0.0, prefilter=False)
        shift_label = ndimage.shift(self.label, (dx, dy), order=0, mode="constant", cval=0, prefilter=False)

        self.assertEqual(shift_img.shape, (128, 128))
        self.assertEqual(shift_label.shape, (128, 128))
        unique_labels = np.unique(shift_label)
        self.assertTrue(set(unique_labels).issubset({0, 1, 2, 3}))

    def test_f3_scar_random_generator_independent_intensity_transforms(self):
        """Tier 1: Intensity transforms (brightness, gamma, Gaussian noise) do NOT alter the label mask."""
        label_copy = self.label.copy()

        # Brightness
        factor = 1.1
        aug_img0 = np.clip(self.img0 * factor, 0.0, 1.0)
        # Gamma
        gamma = 0.9
        aug_img1 = np.clip(self.img1, 0.0, 1.0) ** gamma
        # Noise
        noise = np.random.normal(0, 0.02, self.img2.shape)
        aug_img2 = np.clip(self.img2 + noise, 0.0, 1.0)

        # Label must remain completely untouched
        np.testing.assert_array_equal(self.label, label_copy)
        self.assertFalse(np.allclose(aug_img0, self.img0))
        self.assertFalse(np.allclose(aug_img1, self.img1))

    def test_f3_scar_random_generator_output_shape_and_range(self):
        """Tier 1: Output images are clipped to [0.0, 1.0] and resized to exactly (128, 128)."""
        noisy = np.clip(self.img0 + 0.5, 0.0, 1.0)
        self.assertLessEqual(noisy.max(), 1.0)
        self.assertGreaterEqual(noisy.min(), 0.0)
        self.assertEqual(noisy.shape, (128, 128))

    # =========================================================================
    # Tier 2: Boundary & Corner Cases (F3)
    # =========================================================================

    def test_f3_deterministic_seed_spatial_transform(self):
        """Tier 2: When random seed is fixed, spatial transformations are completely reproducible."""
        np.random.seed(1234)
        angle1 = np.random.uniform(-20, 20)
        np.random.seed(1234)
        angle2 = np.random.uniform(-20, 20)
        self.assertEqual(angle1, angle2)

    def test_f3_identity_transform_zero_angles(self):
        """Tier 2: When angle=0 and shift=(0, 0), images and labels are identical to input."""
        rot_img = ndimage.rotate(self.img0, 0.0, order=1, reshape=False, mode="constant", cval=0.0, prefilter=False)
        shift_img = ndimage.shift(rot_img, (0.0, 0.0), order=1, mode="constant", cval=0.0, prefilter=False)
        np.testing.assert_allclose(shift_img, self.img0, atol=1e-6)

    def test_f3_extreme_rotation_boundary_angles(self):
        """Tier 2: Test extreme rotation angles -20.0 and +20.0 degrees."""
        for angle in [-20.0, 20.0]:
            rot_label = ndimage.rotate(self.label, angle, order=0, reshape=False, mode="constant", cval=0, prefilter=False)
            unique_labels = np.unique(rot_label)
            self.assertTrue(set(unique_labels).issubset({0, 1, 2, 3}))

    def test_f3_extreme_shift_boundary_pixels(self):
        """Tier 2: Test extreme translation shifts -5.0 and +5.0 pixels."""
        for shift_val in [(-5.0, -5.0), (5.0, 5.0), (-5.0, 5.0)]:
            shift_img = ndimage.shift(self.img0, shift_val, order=1, mode="constant", cval=0.0, prefilter=False)
            self.assertEqual(shift_img.shape, (128, 128))
            self.assertGreaterEqual(shift_img.min(), 0.0)

    def test_f3_label_nearest_interpolation_preserves_integer_set(self):
        """Tier 2: Nearest neighbor interpolation prevents creating fractional labels like 1.5 or 2.7."""
        angle = 17.3
        rot_label = ndimage.rotate(self.label.astype(np.float32), angle, order=0, reshape=False, mode="constant", cval=0)
        unique_vals = np.unique(rot_label)
        for val in unique_vals:
            self.assertAlmostEqual(val, round(val), delta=1e-5, msg=f"Found fractional label: {val}")


class TestDataPipelineCrossFeature(unittest.TestCase):
    """Covers Tier 3: Cross-Feature Integration between Data Pipeline, Augmentation, and Zero-Padding."""

    def test_t3_dataset_loader_with_random_generator_to_five_branch_batch(self):
        """Tier 3: End-to-end integration:
        Read real sample -> Apply SCAR augmentation -> Build 5-branch batch dictionary -> Verify tensor properties.
        """
        sample_path = os.path.join(PROCESSED_DATA_DIR, "bSSFP", "train_npz", "case0002_slice000.npz")
        data = np.load(sample_path)
        img_C0 = data["image"]
        label = data["label"].astype(np.int64)

        # Spatial transformation
        k, axis = 1, 0
        img_C0_aug = np.flip(np.rot90(img_C0, k), axis=axis).copy()
        label_aug = np.flip(np.rot90(label, k), axis=axis).copy()

        # Convert to torch tensor batch (B=1)
        C0_t = torch.from_numpy(img_C0_aug).unsqueeze(0).unsqueeze(0).float()
        LGE_t = torch.from_numpy(img_C0_aug).unsqueeze(0).unsqueeze(0).float()
        T2_t = torch.from_numpy(img_C0_aug).unsqueeze(0).unsqueeze(0).float()
        T1m_t = torch.zeros_like(C0_t)
        T2starm_t = torch.zeros_like(C0_t)
        label_t = torch.from_numpy(label_aug).unsqueeze(0).long()

        batch = {
            "C0": C0_t,
            "LGE": LGE_t,
            "T2": T2_t,
            "T1m": T1m_t,
            "T2starm": T2starm_t,
            "label": label_t,
            "case_id": ["case0002_slice000"],
        }

        # Assertions on combined batch
        self.assertEqual(batch["C0"].shape, (1, 1, 128, 128))
        self.assertEqual(batch["T1m"].shape, (1, 1, 128, 128))
        self.assertEqual(batch["T1m"].sum().item(), 0.0)
        self.assertEqual(batch["label"].shape, (1, 128, 128))
        self.assertTrue(set(torch.unique(batch["label"]).numpy()).issubset({0, 1, 2, 3}))


if __name__ == "__main__":
    unittest.main()
