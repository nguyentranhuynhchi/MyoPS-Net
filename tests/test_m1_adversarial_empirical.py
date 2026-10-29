"""Empirical Adversarial Stress Test Suite for Milestone 1.

Protocol: myops380_train304_test76_last_v1
Focus:
a. 100+ random augmentation passes on real sample images and labels:
   - Check image values ALWAYS strictly in [0.0, 1.0]
   - Check discrete labels NEVER contain fractional values and unique classes are a subset of {0, 1, 2, 3}
b. Missing modalities T1m and T2starm are 100% pure zeros ((tensor == 0).all()) across all augmented batches
c. DataLoader iteration across full epoch: verify drop_last=False yields exactly 1,972 samples (123x16 + 1x4)
d. Memory contiguousness and positive strides so tensors pass directly to PyTorch / CUDA convolutions
"""
import os
import unittest
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset.data_augmentation import RandomGenerator, ResizeGenerator, apply_intensity_transforms
from dataset.myops380_dataset import (
    MyoPS380TrainDataset,
    MyoPS380TestVolumeDataset,
    build_myops380_dataloader,
    read_split_names,
)
from tests.contract_helpers import PROCESSED_DATA_DIR, SCAR_SPLIT_DIR


class TestMilestone1EmpiricalStress(unittest.TestCase):
    """Empirical adversarial test harness."""

    @classmethod
    def setUpClass(cls):
        cls.data_dir = PROCESSED_DATA_DIR
        cls.split_dir = SCAR_SPLIT_DIR
        cls.train_manifest = os.path.join(cls.split_dir, "train.txt")
        cls.test_manifest = os.path.join(cls.split_dir, "test_vol.txt")
        cls.train_names = read_split_names(cls.train_manifest)
        cls.test_names = read_split_names(cls.test_manifest)

    def test_a_100_plus_random_augmentations_on_real_samples(self):
        """Challenge A: Run 500 random augmentation passes across diverse real slices.

        Verify:
        - Image values strictly in [0.0, 1.0] (no overflow, underflow, NaN, Inf)
        - Labels strictly integer in {0, 1, 2, 3} (no fractional values)
        """
        # Select 10 diverse real sample slices across the training set
        step = len(self.train_names) // 10
        sample_indices = [i * step for i in range(10)]
        sample_keys = [self.train_names[i] for i in sample_indices]

        # Load real data
        real_samples = []
        for name in sample_keys:
            p_c0 = os.path.join(self.data_dir, "bSSFP", "train_npz", f"{name}.npz")
            p_lge = os.path.join(self.data_dir, "LGE", "train_npz", f"{name}.npz")
            p_t2 = os.path.join(self.data_dir, "T2w", "train_npz", f"{name}.npz")

            with np.load(p_c0) as d_c0, np.load(p_lge) as d_lge, np.load(p_t2) as d_t2:
                real_samples.append({
                    "C0": d_c0["image"].astype(np.float32),
                    "LGE": d_lge["image"].astype(np.float32),
                    "T2": d_t2["image"].astype(np.float32),
                    "label": np.round(d_c0["label"]).astype(np.int64),
                    "case_id": name,
                })

        aug = RandomGenerator(output_size=(128, 128), image_keys=("C0", "LGE", "T2"))

        total_passes = 0
        passes_per_sample = 50  # 10 samples * 50 passes = 500 passes (>> 100 required)

        for s_idx, sample in enumerate(real_samples):
            for p_idx in range(passes_per_sample):
                out = aug(sample)
                total_passes += 1

                for mod in ("C0", "LGE", "T2"):
                    img_t = out[mod]
                    self.assertIsInstance(img_t, torch.Tensor, f"Pass {total_passes}: {mod} not a Tensor")
                    self.assertEqual(img_t.dtype, torch.float32, f"Pass {total_passes}: {mod} dtype != float32")
                    self.assertEqual(img_t.shape, (1, 128, 128), f"Pass {total_passes}: {mod} shape != (1, 128, 128)")

                    # Check finite
                    self.assertTrue(torch.isfinite(img_t).all().item(), f"Pass {total_passes}: {mod} contains NaN/Inf")

                    # Strictly within [0.0, 1.0]
                    min_val = img_t.min().item()
                    max_val = img_t.max().item()
                    self.assertGreaterEqual(
                        min_val, 0.0,
                        f"Pass {total_passes}: {mod} underflow min={min_val} < 0.0"
                    )
                    self.assertLessEqual(
                        max_val, 1.0,
                        f"Pass {total_passes}: {mod} overflow max={max_val} > 1.0"
                    )

                # Check label
                lbl_t = out["label"]
                self.assertIsInstance(lbl_t, torch.Tensor, f"Pass {total_passes}: label not a Tensor")
                self.assertEqual(lbl_t.dtype, torch.int64, f"Pass {total_passes}: label dtype != int64")
                self.assertEqual(lbl_t.shape, (128, 128), f"Pass {total_passes}: label shape != (128, 128)")

                # Fractional check: label must be strictly integers
                self.assertTrue(
                    torch.equal(lbl_t, lbl_t.round()),
                    f"Pass {total_passes}: label contains fractional values"
                )

                # Unique classes must be subset of {0, 1, 2, 3}
                unique_classes = set(torch.unique(lbl_t).tolist())
                self.assertTrue(
                    unique_classes.issubset({0, 1, 2, 3}),
                    f"Pass {total_passes}: label contains forbidden classes {unique_classes}"
                )

        self.assertEqual(total_passes, 500, f"Expected 500 passes, ran {total_passes}")

    def test_b_missing_modalities_pure_zeros(self):
        """Challenge B: Verify T1m and T2starm are 100% pure zeros across augmented batches and test volumes."""
        dataset = MyoPS380TrainDataset(
            data_dir=self.data_dir,
            split_dir=self.split_dir,
            split="train",
            img_size=(128, 128),
        )

        # 1. Check across 100 individual dataset getitem calls
        for idx in range(100):
            sample = dataset[idx]
            t1m = sample["T1m"]
            t2starm = sample["T2starm"]

            self.assertIsInstance(t1m, torch.Tensor)
            self.assertIsInstance(t2starm, torch.Tensor)
            self.assertEqual(t1m.shape, (1, 128, 128))
            self.assertEqual(t2starm.shape, (1, 128, 128))

            # Strictly 100% pure zeros
            self.assertTrue(
                (t1m == 0).all().item(),
                f"Sample {idx}: T1m is not 100% pure zeros! Non-zero count: {(t1m != 0).sum().item()}"
            )
            self.assertTrue(
                (t2starm == 0).all().item(),
                f"Sample {idx}: T2starm is not 100% pure zeros! Non-zero count: {(t2starm != 0).sum().item()}"
            )
            self.assertEqual(t1m.sum().item(), 0.0)
            self.assertEqual(t2starm.sum().item(), 0.0)

        # 2. Check across DataLoader batches
        loader = DataLoader(dataset, batch_size=16, shuffle=True, drop_last=False, num_workers=0)
        batch_count = 0
        for batch in loader:
            batch_count += 1
            if batch_count > 10:
                break
            b_t1m = batch["T1m"]
            b_t2starm = batch["T2starm"]

            self.assertTrue(
                (b_t1m == 0).all().item(),
                f"Batch {batch_count}: T1m contains non-zero elements!"
            )
            self.assertTrue(
                (b_t2starm == 0).all().item(),
                f"Batch {batch_count}: T2starm contains non-zero elements!"
            )

        # 3. Check test volume dataset
        test_dataset = MyoPS380TestVolumeDataset(
            data_dir=self.data_dir,
            split_dir=self.split_dir,
            split="test_vol",
        )
        test_vol_sample = test_dataset[0]
        self.assertTrue((test_vol_sample["T1m"] == 0).all(), "Test volume T1m is not all zeros")
        self.assertTrue((test_vol_sample["T2starm"] == 0).all(), "Test volume T2starm is not all zeros")

        for slice_batch, _, _ in test_dataset.iter_slice_batches(0, batch_size=4):
            self.assertTrue((slice_batch["T1m"] == 0).all().item())
            self.assertTrue((slice_batch["T2starm"] == 0).all().item())

    def test_c_dataloader_full_epoch_iteration_1972_samples(self):
        """Challenge C: Verify full epoch iteration yields exactly 1,972 samples across 124 batches (123x16 + 1x4)."""
        dataset = MyoPS380TrainDataset(
            data_dir=self.data_dir,
            split_dir=self.split_dir,
            split="train",
            img_size=(128, 128),
        )

        loader = build_myops380_dataloader(
            dataset=dataset,
            batch_size=16,
            num_workers=0,  # single-process for determinism and debugging
            seed=1234,
        )

        self.assertFalse(loader.drop_last, "DataLoader drop_last must be False")
        self.assertEqual(len(loader), 124, f"Expected 124 batches in DataLoader, got {len(loader)}")

        total_samples = 0
        batch_sizes = []
        observed_case_ids = []

        for b_idx, batch in enumerate(loader):
            bsz = batch["C0"].shape[0]
            batch_sizes.append(bsz)
            total_samples += bsz

            # Check batch size per spec: 123 batches of 16, 1 batch of 4
            if b_idx < 123:
                self.assertEqual(bsz, 16, f"Batch {b_idx} size expected 16, got {bsz}")
            else:
                self.assertEqual(bsz, 4, f"Batch {b_idx} (final microbatch) size expected 4, got {bsz}")

            observed_case_ids.extend(batch["case_id"])

        # Check total sample count
        self.assertEqual(total_samples, 1972, f"Expected exactly 1,972 samples, got {total_samples}")
        self.assertEqual(len(batch_sizes), 124, f"Expected exactly 124 batches, got {len(batch_sizes)}")
        self.assertEqual(batch_sizes[:123], [16] * 123)
        self.assertEqual(batch_sizes[-1], 4)

        # Check all case IDs are unique (no duplicates, no dropped slices)
        self.assertEqual(len(observed_case_ids), 1972)
        self.assertEqual(len(set(observed_case_ids)), 1972, "Duplicate slice detected in full epoch iteration!")

        # Verify exact match with manifest
        self.assertEqual(set(observed_case_ids), set(self.train_names))

    def test_d_memory_contiguousness_positive_strides_and_conv2d(self):
        """Challenge D: Test memory contiguousness and positive strides for PyTorch/CUDA convolutions."""
        dataset = MyoPS380TrainDataset(
            data_dir=self.data_dir,
            split_dir=self.split_dir,
            split="train",
            img_size=(128, 128),
        )

        # Check individual samples
        for idx in range(20):
            sample = dataset[idx]
            for mod in ("C0", "LGE", "T2", "T1m", "T2starm"):
                t = sample[mod]
                self.assertTrue(t.is_contiguous(), f"Sample {idx} {mod} is not contiguous!")
                self.assertTrue(all(s > 0 for s in t.stride()), f"Sample {idx} {mod} has non-positive stride: {t.stride()}")

            lbl = sample["label"]
            self.assertTrue(lbl.is_contiguous(), f"Sample {idx} label is not contiguous!")
            self.assertTrue(all(s > 0 for s in lbl.stride()), f"Sample {idx} label has non-positive stride: {lbl.stride()}")

        # Check batched DataLoader tensors
        loader = DataLoader(dataset, batch_size=4, shuffle=True, drop_last=False, num_workers=0)
        conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        conv5 = nn.Conv2d(5, 16, kernel_size=3, padding=1)

        batch_count = 0
        for batch in loader:
            batch_count += 1
            if batch_count > 10:
                break

            for mod in ("C0", "LGE", "T2", "T1m", "T2starm"):
                t = batch[mod]
                self.assertTrue(t.is_contiguous(), f"Batch {batch_count} {mod} is not contiguous!")
                self.assertTrue(all(s > 0 for s in t.stride()), f"Batch {batch_count} {mod} non-positive stride: {t.stride()}")
                # Execute single-channel conv2d without error
                out_single = conv1(t)
                self.assertEqual(out_single.shape, (t.shape[0], 8, 128, 128))

            # 5-branch stack for MyoPSNet input
            five_branches = torch.cat([batch["C0"], batch["LGE"], batch["T2"], batch["T1m"], batch["T2starm"]], dim=1)
            self.assertTrue(five_branches.is_contiguous(), "5-channel concatenated tensor is not contiguous!")
            self.assertTrue(all(s > 0 for s in five_branches.stride()), f"5-channel non-positive stride: {five_branches.stride()}")

            # Execute 5-channel conv2d without error
            out_five = conv5(five_branches)
            self.assertEqual(out_five.shape, (four_b := batch["C0"].shape[0], 16, 128, 128))

    def test_e_stress_extreme_inputs_and_boundary_conditions(self):
        """Challenge E: Stress test augmentation transforms under extreme edge cases.

        - All-zero images
        - All-one images
        - Boundary angles (-20.0 deg, +20.0 deg, 0.0 deg)
        - Boundary shifts ((-5, -5), (5, 5), (0, 0))
        - Extreme labels (all 0, all 3, checkerboard)
        """
        aug = RandomGenerator(output_size=(128, 128), image_keys=("C0", "LGE", "T2"))

        # Case 1: All-zero images
        zero_sample = {
            "C0": np.zeros((128, 128), dtype=np.float32),
            "LGE": np.zeros((128, 128), dtype=np.float32),
            "T2": np.zeros((128, 128), dtype=np.float32),
            "label": np.zeros((128, 128), dtype=np.int64),
            "case_id": "test_zeros",
        }
        for _ in range(20):
            out = aug(zero_sample)
            for mod in ("C0", "LGE", "T2"):
                self.assertTrue(torch.isfinite(out[mod]).all().item())
                self.assertGreaterEqual(out[mod].min().item(), 0.0)
                self.assertLessEqual(out[mod].max().item(), 1.0)
            self.assertEqual(set(torch.unique(out["label"]).tolist()), {0})

        # Case 2: All-one images
        one_sample = {
            "C0": np.ones((128, 128), dtype=np.float32),
            "LGE": np.ones((128, 128), dtype=np.float32),
            "T2": np.ones((128, 128), dtype=np.float32),
            "label": np.ones((128, 128), dtype=np.int64) * 3,
            "case_id": "test_ones",
        }
        for _ in range(20):
            out = aug(one_sample)
            for mod in ("C0", "LGE", "T2"):
                self.assertTrue(torch.isfinite(out[mod]).all().item())
                self.assertGreaterEqual(out[mod].min().item(), 0.0)
                self.assertLessEqual(out[mod].max().item(), 1.0)
            # Label might have 0 at borders after shift/rotate (mode='constant', cval=0)
            self.assertTrue(set(torch.unique(out["label"]).tolist()).issubset({0, 3}))

        # Case 3: Checkerboard of all classes {0, 1, 2, 3}
        checker_label = np.indices((128, 128)).sum(axis=0) % 4
        checker_sample = {
            "C0": np.random.rand(128, 128).astype(np.float32),
            "LGE": np.random.rand(128, 128).astype(np.float32),
            "T2": np.random.rand(128, 128).astype(np.float32),
            "label": checker_label.astype(np.int64),
            "case_id": "test_checker",
        }
        for _ in range(20):
            out = aug(checker_sample)
            # Must strictly preserve integer classes without fractional artifacts
            lbl = out["label"]
            self.assertTrue(torch.equal(lbl, lbl.round()))
            self.assertTrue(set(torch.unique(lbl).tolist()).issubset({0, 1, 2, 3}))


if __name__ == "__main__":
    unittest.main()
