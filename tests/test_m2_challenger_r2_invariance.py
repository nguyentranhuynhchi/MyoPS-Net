"""Adversarial Invariance and Rank Restoration Stress Tests for Milestone 2 Iteration 2.

Empirical verification of Target Adapter shape handling, volumetric rank restoration,
clinical priority across ranks, and boundary conditions.
"""
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from dataset.target_adapter import (
    canonical_to_myops_targets,
    myops_outputs_to_canonical,
    TargetAdapter,
    TargetDict,
)


class TestTargetAdapterVolumetricInvarianceR2(unittest.TestCase):
    """Stress tests on volumetric, single slice, and rank restoration behaviors."""

    def setUp(self):
        self.adapter = TargetAdapter()

    def test_unbatched_3d_volume_various_depths(self):
        """Unbatched 3D volume (128, 128, D) across multiple depths (1, 4, 8, 16, 24)."""
        for D in [1, 4, 8, 16, 24]:
            with self.subTest(D=D):
                label = torch.zeros((128, 128, D), dtype=torch.int64)
                # Assign regions for each class
                label[10:30, 10:30, :] = 1  # Myo
                label[20:25, 20:25, :] = 2  # Edema
                label[22:24, 22:24, :] = 3  # Scar

                targets = canonical_to_myops_targets(label)
                self.assertIsInstance(targets, TargetDict)
                self.assertTrue(targets.is_unbatched_3d)
                self.assertTrue(targets.unbatched)

                # Check shapes: internally (1, 3, 128, 128, D)
                for k in ["scar", "edema", "cardiac"]:
                    self.assertIn(k, targets)
                    self.assertEqual(targets[k].shape, (1, 3, 128, 128, D))

                # Round trip to canonical
                recovered = myops_outputs_to_canonical(targets)
                self.assertEqual(recovered.shape, (128, 128, D))
                self.assertEqual(recovered.dtype, torch.int64)
                torch.testing.assert_close(recovered, label)

    def test_batched_3d_volume_various_batch_sizes(self):
        """Batched 3D volume (B, 128, 128, D) with B in [1, 2, 4]."""
        for B in [1, 2, 4]:
            with self.subTest(B=B):
                D = 8
                label = torch.zeros((B, 128, 128, D), dtype=torch.int64)
                label[:, 15:40, 15:40, :] = 1
                label[:, 25:35, 25:35, :] = 2
                label[:, 30:33, 30:33, :] = 3

                targets = canonical_to_myops_targets(label)
                self.assertFalse(targets.is_unbatched_3d)

                for k in ["scar", "edema", "cardiac"]:
                    self.assertEqual(targets[k].shape, (B, 3, 128, 128, D))

                recovered = myops_outputs_to_canonical(targets)
                self.assertEqual(recovered.shape, (B, 128, 128, D))
                torch.testing.assert_close(recovered, label)

    def test_unbatched_2d_slice_rank_restoration(self):
        """Unbatched 2D slice (128, 128) must recover exactly (128, 128)."""
        label = torch.zeros((128, 128), dtype=torch.int64)
        label[10:50, 10:50] = 1
        label[20:30, 20:30] = 2
        label[25:28, 25:28] = 3

        targets = canonical_to_myops_targets(label)
        self.assertTrue(targets.is_unbatched_2d)
        self.assertTrue(targets.unbatched)

        for k in ["scar", "edema", "cardiac"]:
            self.assertEqual(targets[k].shape, (1, 3, 128, 128))

        recovered = myops_outputs_to_canonical(targets)
        self.assertEqual(recovered.shape, (128, 128))
        self.assertEqual(recovered.dtype, torch.int64)
        torch.testing.assert_close(recovered, label)

    def test_batched_2d_slice_batch_one_preservation(self):
        """Batched 2D slice with B=1 (1, 128, 128) MUST retain batch dimension (1, 128, 128)."""
        label = torch.zeros((1, 128, 128), dtype=torch.int64)
        label[0, 10:50, 10:50] = 1
        label[0, 20:30, 20:30] = 2

        targets = canonical_to_myops_targets(label)
        self.assertFalse(targets.is_unbatched_2d)
        self.assertFalse(targets.is_unbatched_3d)

        for k in ["scar", "edema", "cardiac"]:
            self.assertEqual(targets[k].shape, (1, 3, 128, 128))

        recovered = myops_outputs_to_canonical(targets)
        self.assertEqual(recovered.shape, (1, 128, 128))
        torch.testing.assert_close(recovered, label)

    def test_singleton_channel_squeezing(self):
        """(B, 1, H, W) and (B, 1, H, W, D) singleton channel inputs."""
        # 4D singleton channel: (2, 1, 128, 128)
        lbl_4d = torch.randint(0, 4, (2, 1, 128, 128), dtype=torch.int64)
        targets_4d = canonical_to_myops_targets(lbl_4d)
        self.assertEqual(targets_4d["scar"].shape, (2, 3, 128, 128))

        # 5D singleton channel: (2, 1, 128, 128, 8)
        lbl_5d = torch.randint(0, 4, (2, 1, 128, 128, 8), dtype=torch.int64)
        targets_5d = canonical_to_myops_targets(lbl_5d)
        self.assertEqual(targets_5d["scar"].shape, (2, 3, 128, 128, 8))

    def test_direct_tensor_and_unbatched_logits(self):
        """Direct logits passed to myops_outputs_to_canonical for unbatched 3D volume."""
        # Unbatched 3D volume logits: (3, 128, 128, 8)
        logits_scar = torch.zeros((3, 128, 128, 8), dtype=torch.float32)
        logits_edema = torch.zeros((3, 128, 128, 8), dtype=torch.float32)
        logits_cardiac = torch.zeros((3, 128, 128, 8), dtype=torch.float32)

        # Set default to BG (ch 0)
        logits_scar[0] = 5.0
        logits_edema[0] = 5.0
        logits_cardiac[0] = 5.0

        # Region with scar
        logits_scar[2, 20:30, 20:30, :] = 10.0
        logits_scar[0, 20:30, 20:30, :] = -5.0

        # Region with edema
        logits_edema[2, 35:45, 35:45, :] = 10.0
        logits_edema[0, 35:45, 35:45, :] = -5.0

        # Region with normal myo
        logits_cardiac[1, 50:60, 50:60, :] = 10.0
        logits_cardiac[0, 50:60, 50:60, :] = -5.0

        pred_dict = {
            "scar": logits_scar,
            "edema": logits_edema,
            "cardiac": logits_cardiac,
        }
        canonical = myops_outputs_to_canonical(pred_dict)
        self.assertEqual(canonical.shape, (128, 128, 8))
        self.assertEqual((canonical == 3).sum().item(), 10 * 10 * 8)
        self.assertEqual((canonical == 2).sum().item(), 10 * 10 * 8)
        self.assertEqual((canonical == 1).sum().item(), 10 * 10 * 8)

    def test_clinical_priority_in_3d_volume(self):
        """Clinical priority: Scar > Edema > Normal Myo across 3D voxels."""
        B, D = 2, 6
        # Both scar and edema fire on same voxel
        pred_scar = torch.full((B, 128, 128, D), 2, dtype=torch.int64)  # Scar channel
        pred_edema = torch.full((B, 128, 128, D), 2, dtype=torch.int64)  # Edema channel
        pred_cardiac = torch.full((B, 128, 128, D), 1, dtype=torch.int64)  # Myo channel

        outputs = {"scar": pred_scar, "edema": pred_edema, "cardiac": pred_cardiac}
        canonical = myops_outputs_to_canonical(outputs)
        self.assertEqual(canonical.shape, (B, 128, 128, D))
        # Scar wins everywhere
        self.assertTrue((canonical == 3).all().item())

        # Now remove scar, edema fires with myo
        pred_scar.zero_()
        outputs = {"scar": pred_scar, "edema": pred_edema, "cardiac": pred_cardiac}
        canonical = myops_outputs_to_canonical(outputs)
        # Edema wins everywhere
        self.assertTrue((canonical == 2).all().item())

        # Now remove edema, normal myo fires
        pred_edema.zero_()
        outputs = {"scar": pred_scar, "edema": pred_edema, "cardiac": pred_cardiac}
        canonical = myops_outputs_to_canonical(outputs)
        # Normal myo wins everywhere
        self.assertTrue((canonical == 1).all().item())

    def test_non_contiguous_and_adapter_module(self):
        """Ensure adapter handles non-contiguous tensors and nn.Module forward."""
        label_base = torch.randint(0, 4, (128, 128, 8), dtype=torch.int64)
        # Create non-contiguous slice
        label_nc = label_base.permute(1, 0, 2)
        self.assertFalse(label_nc.is_contiguous())

        targets = self.adapter(label_nc)
        self.assertEqual(targets["scar"].shape, (1, 3, 128, 128, 8))

        recovered = self.adapter.to_canonical(targets)
        self.assertEqual(recovered.shape, (128, 128, 8))
        torch.testing.assert_close(recovered, label_nc)


if __name__ == "__main__":
    unittest.main()
