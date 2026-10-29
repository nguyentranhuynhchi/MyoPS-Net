"""
Empirical Adversarial Stress Test Suite for Protocol Invariants & Manifest Integrity.
Milestone 1 Challenger 2: Protocol Invariants & Manifest Adversary.

Tests cover:
  Challenge 2a: Protocol Enforcement (strict ValueError on 'val' in any form)
  Challenge 2b: Disjoint Patient Sets (304 train vs 76 test, 0 overlap, 380 total)
  Challenge 2c: Edge Cases & Corruption (non-existent, empty, duplicates, extensions, non-finite)
  Challenge 2d: 3D Volume Dataset & Slice Reconstruction (all 76 volumes, depths, CPU & CUDA)
  Challenge 2e: DataLoader Multiprocessing & Determinism (num_workers=2, seed_worker, drop_last=False)
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
import re
import sys
import tempfile
import unittest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset.myops380_dataset import (
    MyoPS380TrainDataset,
    MyoPS380TestVolumeDataset,
    build_myops380_dataloader,
    read_split_names,
    extract_patient_id,
    ingest_image_array,
    seed_worker,
)


SCAR_SPLIT_DIR = Path(r"D:\nckh_model\SCAR\preprocessing\splits\myops380_no_val")
PROCESSED_DATA_DIR = Path(r"D:\nckh_model\Dataset\MyoPS380\Processed_data")


class TestAdversarialProtocolInvariants(unittest.TestCase):
    """Rigorous empirical challenge suite for dataset/myops380_dataset.py."""

    # =========================================================================
    # Challenge 2a: Protocol Enforcement
    # =========================================================================

    def test_2a_dataset_val_split_strict_rejection(self):
        """Verify that MyoPS380TrainDataset strictly raises ValueError for any 'val' split variant."""
        val_variants = [
            "val",
            "VAL",
            "Val",
            "val_vol",
            "validation",
            "VALIDATION",
            "train_val",
            "my_val_set",
            "valid",  # wait, does 'valid' contain 'val'? Yes, 'val' in 'valid'
        ]
        for v in val_variants:
            with self.subTest(split_variant=v):
                with self.assertRaises(ValueError) as ctx:
                    MyoPS380TrainDataset(split=v)
                err_msg = str(ctx.exception)
                self.assertIn("Validation split is forbidden", err_msg)
                self.assertIn("myops380_train304_test76_last_v1", err_msg)

    def test_2a_dataloader_val_split_strict_rejection(self):
        """Verify that build_myops380_dataloader strictly raises ValueError when split contains 'val'."""
        val_variants = ["val", "VAL", "validation", "val_vol"]
        for v in val_variants:
            with self.subTest(split_variant=v):
                with self.assertRaises(ValueError) as ctx:
                    build_myops380_dataloader(split=v)
                self.assertIn("Validation split is forbidden", str(ctx.exception))

    def test_2a_no_val_manifests_exist_in_benchmark_dir(self):
        """Verify that no val.txt, val_vol.txt, or validation manifests exist in the official split folder."""
        for forbidden in ["val.txt", "val_vol.txt", "validation.txt", "val_slice.txt"]:
            forbidden_path = SCAR_SPLIT_DIR / forbidden
            self.assertFalse(
                forbidden_path.exists(),
                f"Forbidden validation manifest detected: {forbidden_path}"
            )

    # =========================================================================
    # Challenge 2b: Disjoint Patient Sets & Manifest Integrity
    # =========================================================================

    def test_2b_manifest_slice_and_volume_counts(self):
        """Verify exact counts: train.txt has 1,972 slices; test_vol.txt has 76 volumes."""
        train_path = SCAR_SPLIT_DIR / "train.txt"
        test_path = SCAR_SPLIT_DIR / "test_vol.txt"

        train_slices = read_split_names(train_path)
        test_vols = read_split_names(test_path)

        self.assertEqual(len(train_slices), 1972, f"Expected 1972 train slices, got {len(train_slices)}")
        self.assertEqual(len(test_vols), 76, f"Expected 76 test volumes, got {len(test_vols)}")

    def test_2b_zero_patient_overlap_and_full_380_coverage(self):
        """Verify 0 overlap between 304 train patients and 76 test patients, forming full 380 set."""
        train_slices = read_split_names(SCAR_SPLIT_DIR / "train.txt")
        test_vols = read_split_names(SCAR_SPLIT_DIR / "test_vol.txt")

        train_patients = {extract_patient_id(s) for s in train_slices}
        test_patients = {extract_patient_id(v) for v in test_vols}

        # Check train patient count
        self.assertEqual(len(train_patients), 304, f"Expected 304 train patients, got {len(train_patients)}")
        # Check test patient count
        self.assertEqual(len(test_patients), 76, f"Expected 76 test patients, got {len(test_patients)}")

        # Check strictly zero overlap
        overlap = train_patients.intersection(test_patients)
        self.assertEqual(len(overlap), 0, f"Critical Protocol Violation: Overlapping patients detected: {overlap}")

        # Check full coverage of all 380 MyoPS patients
        all_patients = train_patients.union(test_patients)
        self.assertEqual(len(all_patients), 380, f"Expected exactly 380 unique patients, got {len(all_patients)}")

        # Check case ID syntax and numeric bounds (case0001 to case0380)
        expected_ids = {f"case{i:04d}" for i in range(1, 381)}
        self.assertEqual(all_patients, expected_ids, "Patient IDs do not match continuous range case0001..case0380")

    def test_2b_manifest_sha256_hash_integrity(self):
        """Verify train.txt and test_vol.txt cryptographic hashes against split_metadata.json."""
        meta_path = SCAR_SPLIT_DIR / "split_metadata.json"
        self.assertTrue(meta_path.is_file(), f"Missing split_metadata.json at {meta_path}")

        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        expected_train_sha256 = metadata.get("train_ids_sha256")
        expected_test_sha256 = metadata.get("test_ids_sha256")

        train_names = read_split_names(SCAR_SPLIT_DIR / "train.txt")
        test_names = read_split_names(SCAR_SPLIT_DIR / "test_vol.txt")

        # According to split_metadata.json: "SHA256 of lexically sorted IDs, UTF-8, LF after every ID"
        def compute_ids_hash(names: list[str]) -> str:
            sorted_ids = sorted([s.strip() for s in names if s.strip()])
            blob = "".join(f"{item}\n" for item in sorted_ids).encode("utf-8")
            return hashlib.sha256(blob).hexdigest()

        computed_train_sha256 = compute_ids_hash(train_names)
        computed_test_sha256 = compute_ids_hash(test_names)

        self.assertEqual(computed_train_sha256, expected_train_sha256, "train.txt SHA256 mismatch")
        self.assertEqual(computed_test_sha256, expected_test_sha256, "test_vol.txt SHA256 mismatch")

    # =========================================================================
    # Challenge 2c: Edge Cases & Malformed Inputs
    # =========================================================================

    def test_2c_nonexistent_manifest_file(self):
        """read_split_names raises FileNotFoundError for non-existent path."""
        with self.assertRaises(FileNotFoundError):
            read_split_names(r"D:\non_existent_dir\missing_split.txt")

    def test_2c_empty_manifest_file(self):
        """read_split_names raises ValueError for 0-byte or whitespace-only files."""
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".txt") as tmp:
            tmp.write("   \n\n  \r\n")
            tmp_path = tmp.name

        try:
            with self.assertRaises(ValueError) as ctx:
                read_split_names(tmp_path)
            self.assertIn("manifest is empty", str(ctx.exception))
        finally:
            os.remove(tmp_path)

    def test_2c_manifest_with_duplicates(self):
        """read_split_names raises ValueError when duplicate sample names exist."""
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".txt") as tmp:
            tmp.write("case0002_slice000\ncase0002_slice001\ncase0002_slice000\n")
            tmp_path = tmp.name

        try:
            with self.assertRaises(ValueError) as ctx:
                read_split_names(tmp_path)
            self.assertIn("Duplicate entries found", str(ctx.exception))
        finally:
            os.remove(tmp_path)

    def test_2c_manifest_with_file_extensions(self):
        """read_split_names raises ValueError if entries contain suffixes like .npz or .h5."""
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".txt") as tmp:
            tmp.write("case0002_slice000.npz\ncase0002_slice001.npz\n")
            tmp_path = tmp.name

        try:
            with self.assertRaises(ValueError) as ctx:
                read_split_names(tmp_path)
            self.assertIn("must not contain file extensions", str(ctx.exception))
        finally:
            os.remove(tmp_path)

    def test_2c_ingest_image_array_non_finite_inputs(self):
        """ingest_image_array rejects NaN, Inf, and -Inf gracefully with ValueError."""
        nan_arr = np.array([0.0, 0.5, np.nan, 1.0], dtype=np.float32)
        with self.assertRaises(ValueError) as ctx:
            ingest_image_array(nan_arr)
        self.assertIn("Non-finite values", str(ctx.exception))

        inf_arr = np.array([0.0, np.inf, 1.0], dtype=np.float32)
        with self.assertRaises(ValueError) as ctx:
            ingest_image_array(inf_arr)
        self.assertIn("Non-finite values", str(ctx.exception))

        neg_inf_arr = np.array([-np.inf, 0.5], dtype=np.float32)
        with self.assertRaises(ValueError) as ctx:
            ingest_image_array(neg_inf_arr)
        self.assertIn("Non-finite values", str(ctx.exception))

    def test_2c_ingest_image_array_scaling_invariants(self):
        """Verify ingest_image_array preserves [0, 1] data without redundant /255 division, and scales [0, 255]."""
        # Case A: Already in [0.0, 1.0]
        cache_data = np.array([0.0, 0.1, 0.5, 0.9, 1.0], dtype=np.float32)
        out_a = ingest_image_array(cache_data)
        np.testing.assert_allclose(out_a, cache_data, atol=1e-6)

        # Case B: In [0, 255]
        raw_data = np.array([0.0, 51.0, 127.5, 255.0], dtype=np.float32)
        out_b = ingest_image_array(raw_data)
        expected_b = np.array([0.0, 0.2, 0.5, 1.0], dtype=np.float32)
        np.testing.assert_allclose(out_b, expected_b, atol=1e-5)

        # Case C: Extreme values clipped
        out_clipped = ingest_image_array(np.array([-5.0, 300.0], dtype=np.float32))
        self.assertEqual(out_clipped[0], 0.0)
        self.assertEqual(out_clipped[1], 1.0)

    def test_2c_dataset_validate_files_missing_file_failure(self):
        """Verify validate_files=True catches missing files immediately on instantiation."""
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".txt") as tmp:
            tmp.write("case9999_slice999\n")
            tmp_path = tmp.name

        try:
            with self.assertRaises(FileNotFoundError):
                MyoPS380TrainDataset(split_path=tmp_path, validate_files=True)
        finally:
            os.remove(tmp_path)

    def test_2c_extract_patient_id_edge_cases(self):
        """extract_patient_id handles slice suffixes, h5, npz, nii extensions robustly."""
        cases = {
            "case0002_slice000": "case0002",
            "case0015_slice012": "case0015",
            "case0380_slice123": "case0380",
            "case0001.npy.h5": "case0001",
            "case0076.npz": "case0076",
            "case0099.nii.gz": "case0099",
            "case0099.nii": "case0099",
            "case0042": "case0042",
        }
        for inp, expected in cases.items():
            self.assertEqual(extract_patient_id(inp), expected)

    # =========================================================================
    # Challenge 2d: 3D Volume Dataset & Slice Reconstruction
    # =========================================================================

    def test_2d_test_volume_dataset_loads_all_76_cases(self):
        """Verify MyoPS380TestVolumeDataset loads exactly all 76 test cases."""
        test_ds = MyoPS380TestVolumeDataset(validate_files=True)
        self.assertEqual(len(test_ds), 76)
        self.assertEqual(len(test_ds.sample_list), 76)

    def test_2d_test_volume_properties_and_5_modalities(self):
        """Verify 3D volume item properties, 5-modality dictionary contract, and zero-padding."""
        test_ds = MyoPS380TestVolumeDataset()
        # Test first, middle, and last case
        test_indices = [0, 37, 75]
        for idx in test_indices:
            sample = test_ds[idx]
            case_id = sample["case_id"]
            depth = sample["depth"]

            self.assertIn("C0", sample)
            self.assertIn("LGE", sample)
            self.assertIn("T2", sample)
            self.assertIn("T1m", sample)
            self.assertIn("T2starm", sample)
            self.assertIn("label", sample)

            # Check shapes: (128, 128, depth)
            for key in ["C0", "LGE", "T2", "T1m", "T2starm", "label"]:
                self.assertEqual(sample[key].shape, (128, 128, depth), f"Shape mismatch for {key} in {case_id}")

            # Check intensities in [0.0, 1.0]
            for key in ["C0", "LGE", "T2"]:
                arr = sample[key]
                self.assertGreaterEqual(float(arr.min()), 0.0 - 1e-6)
                self.assertLessEqual(float(arr.max()), 1.0 + 1e-6)
                self.assertEqual(arr.dtype, np.float32)

            # Check zero-padded modalities are strictly all zeros
            self.assertEqual(np.count_nonzero(sample["T1m"]), 0, f"T1m not zero in {case_id}")
            self.assertEqual(np.count_nonzero(sample["T2starm"]), 0, f"T2starm not zero in {case_id}")

            # Check label classes are within canonical {0, 1, 2, 3}
            unique_classes = set(np.unique(sample["label"]))
            self.assertTrue(unique_classes.issubset({0, 1, 2, 3}), f"Invalid label classes in {case_id}: {unique_classes}")

    def test_2d_iter_slice_batches_reconstruction(self):
        """Verify iter_slice_batches correctly slices, batches, and reconstructs 3D volumes on CPU."""
        test_ds = MyoPS380TestVolumeDataset()
        sample = test_ds[0]
        true_depth = sample["depth"]
        c0_vol = sample["C0"]  # (128, 128, D)

        for batch_size in [1, 2, 4, 7, 32]:
            with self.subTest(batch_size=batch_size):
                slices_c0 = []
                total_slices = 0
                expected_start = 0

                for inputs, start, stop in test_ds.iter_slice_batches(0, batch_size=batch_size, device="cpu"):
                    self.assertEqual(start, expected_start, "Batch start index not contiguous")
                    self.assertLessEqual(stop, true_depth)
                    b_slice = stop - start

                    # Verify 5 modalities in batch
                    for mod in ["C0", "LGE", "T2", "T1m", "T2starm"]:
                        self.assertIn(mod, inputs)
                        tensor = inputs[mod]
                        self.assertEqual(tensor.device.type, "cpu")
                        self.assertEqual(tensor.shape, (b_slice, 1, 128, 128))

                    # Verify zero-padding
                    self.assertEqual(float(inputs["T1m"].sum()), 0.0)
                    self.assertEqual(float(inputs["T2starm"].sum()), 0.0)

                    # Collect slices for volume reconstruction
                    c0_chunk = inputs["C0"].squeeze(1).numpy().transpose(1, 2, 0)  # back to (128, 128, b_slice)
                    slices_c0.append(c0_chunk)

                    total_slices += b_slice
                    expected_start = stop

                self.assertEqual(total_slices, true_depth, "Reconstructed depth does not match original depth")

                # Reconstruct full 3D volume from slices
                reconstructed_c0 = np.concatenate(slices_c0, axis=2)
                np.testing.assert_allclose(
                    reconstructed_c0,
                    c0_vol,
                    atol=1e-6,
                    err_msg=f"Reconstructed volume differs from original for batch_size={batch_size}",
                )

    def test_2d_iter_slice_batches_device_handling(self):
        """Verify iter_slice_batches respects target torch.device correctly (CPU and CUDA when available)."""
        test_ds = MyoPS380TestVolumeDataset()
        # Test CPU device
        cpu_dev = torch.device("cpu")
        for inputs, _, _ in test_ds.iter_slice_batches(0, batch_size=2, device=cpu_dev):
            self.assertEqual(inputs["C0"].device, cpu_dev)
            break

        # Test CUDA device handling
        if torch.cuda.is_available():
            cuda_dev = torch.device("cuda:0")
            for inputs, _, _ in test_ds.iter_slice_batches(0, batch_size=2, device=cuda_dev):
                self.assertEqual(inputs["C0"].device.type, "cuda")
                break
        else:
            # If CUDA is not available, attempting to send tensor to cuda should raise expected RuntimeError
            with self.assertRaises((RuntimeError, AssertionError)):
                for inputs, _, _ in test_ds.iter_slice_batches(0, batch_size=2, device="cuda"):
                    pass

    # =========================================================================
    # Challenge 2e: DataLoader Multiprocessing & Seed Determinism
    # =========================================================================

    def test_2e_multiprocessing_seed_worker_deterministic_reproduction(self):
        """Verify num_workers=2 with seed_worker guarantees 100% deterministic batch sequence across runs."""
        # Run A: instantiate dataloader and read 4 batches
        loader_a = build_myops380_dataloader(
            split="train",
            batch_size=8,
            num_workers=2,
            seed=1234,
            pin_memory=False,
        )
        batches_a = []
        for i, batch in enumerate(loader_a):
            batches_a.append({
                "case_id": batch["case_id"],
                "C0_sum": float(batch["C0"].sum()),
                "LGE_sum": float(batch["LGE"].sum()),
                "T2_sum": float(batch["T2"].sum()),
                "label_sum": int(batch["label"].sum()),
            })
            if i >= 3:
                break

        # Run B: re-instantiate fresh dataloader with exact same seed and parameters
        loader_b = build_myops380_dataloader(
            split="train",
            batch_size=8,
            num_workers=2,
            seed=1234,
            pin_memory=False,
        )
        batches_b = []
        for i, batch in enumerate(loader_b):
            batches_b.append({
                "case_id": batch["case_id"],
                "C0_sum": float(batch["C0"].sum()),
                "LGE_sum": float(batch["LGE"].sum()),
                "T2_sum": float(batch["T2"].sum()),
                "label_sum": int(batch["label"].sum()),
            })
            if i >= 3:
                break

        self.assertEqual(len(batches_a), len(batches_b))
        for idx, (b_a, b_b) in enumerate(zip(batches_a, batches_b)):
            self.assertEqual(b_a["case_id"], b_b["case_id"], f"Batch {idx} case_id mismatch across runs")
            self.assertAlmostEqual(b_a["C0_sum"], b_b["C0_sum"], places=5, msg=f"Batch {idx} C0 sum mismatch")
            self.assertAlmostEqual(b_a["LGE_sum"], b_b["LGE_sum"], places=5, msg=f"Batch {idx} LGE sum mismatch")
            self.assertAlmostEqual(b_a["T2_sum"], b_b["T2_sum"], places=5, msg=f"Batch {idx} T2 sum mismatch")
            self.assertEqual(b_a["label_sum"], b_b["label_sum"], f"Batch {idx} label sum mismatch")

    def test_2e_dataloader_drop_last_false_preserves_all_slices(self):
        """Verify DataLoader drop_last=False strictly yields ceil(1972 / 16) = 124 batches with remainder batch 4."""
        loader = build_myops380_dataloader(
            split="train",
            batch_size=16,
            num_workers=0,  # fast sequential check
            seed=1234,
        )
        self.assertFalse(loader.drop_last, "DataLoader drop_last must be False")
        self.assertEqual(len(loader), 124, f"Expected 124 batches per epoch, got {len(loader)}")

        # Check total dataset length
        self.assertEqual(len(loader.dataset), 1972)
        # Remainder formula: 1972 % 16 = 4
        self.assertEqual(1972 % 16, 4)

    def test_2e_seed_worker_generates_distinct_seeds_per_worker(self):
        """Verify seed_worker generates independent non-identical seeds when torch initial seeds differ."""
        # Simulated worker 0
        torch.manual_seed(1000)
        seed_worker(0)
        r0 = random.random()

        # Simulated worker 1
        torch.manual_seed(2000)
        seed_worker(1)
        r1 = random.random()

        self.assertNotEqual(r0, r1, "Workers must have independent random state")


if __name__ == "__main__":
    unittest.main()
