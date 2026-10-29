"""Empirical Adversarial Test Suite for Milestone 2 Challenger 1.

Verifies:
1. Volumetric single-voxel lesion test on 3D volume (128, 128, 8):
   - Exactly 1 voxel of scar, 1 voxel of edema
   - Target adapter behavior and one-hot properties
   - Forward and backward pass through MyoPSLoss
2. Volumetric conflicting logits:
   - Extreme logits (+1000.0) across all branches
   - Tested on 3D volumes (3, 128, 128, 8) and (2, 3, 128, 128, 8)
   - Clinical priority resolution: Scar (3) > Edema (2) > Normal Myo (1) > BG (0)
3. Round-trip shape invariance:
   - 2D slices (128, 128)
   - Batched 2D slices (B, 128, 128)
   - 3D volumes (128, 128, 8)
   - Batched 3D volumes (B, 128, 128, 8)
"""

import sys
import unittest
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F
import numpy as np

from dataset.target_adapter import (
    canonical_to_myops_targets,
    myops_outputs_to_canonical,
    TargetAdapter,
    TargetDict,
)
from network.criterion.loss import MyoPSLoss, WCELoss, DiceLoss


class TestVolumetricSingleVoxelLesion(unittest.TestCase):
    """Mission Item 1: Volumetric single-voxel lesion test."""

    def setUp(self):
        self.criterion = MyoPSLoss()

    def test_single_voxel_lesion_target_adaptation(self):
        """Create a 3D volume (128, 128, 8) with exactly 1 voxel of scar, 1 voxel of edema."""
        vol = torch.zeros((128, 128, 8), dtype=torch.int64)
        # 1 voxel scar
        vol[64, 64, 4] = 3
        # 1 voxel edema
        vol[60, 60, 4] = 2
        # A few voxels of normal myocardium
        vol[50:55, 50:55, 4] = 1 # 25 voxels of normal myo

        targets = canonical_to_myops_targets(vol)
        self.assertIsInstance(targets, TargetDict)

        # Shape verification: unbatched 3D becomes (1, 3, 128, 128, 8)
        for k in ["scar", "edema", "cardiac"]:
            self.assertIn(k, targets)
            self.assertEqual(
                targets[k].shape,
                (1, 3, 128, 128, 8),
                f"Expected target '{k}' to have shape (1, 3, 128, 128, 8), got {targets[k].shape}",
            )

        # Value verification:
        # 1. scar target channel 2 is exactly the 1 voxel of scar
        self.assertEqual(targets["scar"][:, 2].sum().item(), 1.0)
        # scar target channel 1 is myo | edema (25 + 1 = 26 voxels)
        self.assertEqual(targets["scar"][:, 1].sum().item(), 26.0)

        # 2. edema target channel 2 is edema | scar (1 + 1 = 2 voxels)
        self.assertEqual(targets["edema"][:, 2].sum().item(), 2.0)
        # edema target channel 1 is normal myo (25 voxels)
        self.assertEqual(targets["edema"][:, 1].sum().item(), 25.0)

        # 3. cardiac target channel 1 is myo | edema | scar (25 + 1 + 1 = 27 voxels)
        self.assertEqual(targets["cardiac"][:, 1].sum().item(), 27.0)
        # cardiac target channel 2 is strictly 0
        self.assertEqual(targets["cardiac"][:, 2].sum().item(), 0.0)

        # Background channel sum verification
        total_voxels = 128 * 128 * 8
        self.assertEqual(targets["scar"][:, 0].sum().item(), total_voxels - 27)
        self.assertEqual(targets["edema"][:, 0].sum().item(), total_voxels - 27)
        self.assertEqual(targets["cardiac"][:, 0].sum().item(), total_voxels - 27)

    def test_single_voxel_lesion_slice_stack_forward_backward(self):
        """Verify forward and backward pass through MyoPSLoss when volume is processed as slice stack (8, 3, 128, 128)."""
        vol = torch.zeros((128, 128, 8), dtype=torch.int64)
        vol[64, 64, 4] = 3 # 1 voxel scar on slice 4
        vol[60, 60, 4] = 2 # 1 voxel edema on slice 4
        vol[50:55, 50:55, 4] = 1 # 25 voxels normal myo on slice 4
        # Slices 0, 1, 2, 3, 5, 6, 7 are 100% background (testing absent class robustness)

        # Permute depth to batch dimension: (8, 128, 128)
        slices = vol.permute(2, 0, 1) # (8, 128, 128)
        targets = canonical_to_myops_targets(slices)

        self.assertEqual(targets["scar"].shape, (8, 3, 128, 128))
        self.assertEqual(targets["edema"].shape, (8, 3, 128, 128))
        self.assertEqual(targets["cardiac"].shape, (8, 3, 128, 128))

        # Mock predictions with requires_grad
        torch.manual_seed(42)
        logits_c0 = torch.randn((8, 3, 128, 128), requires_grad=True)
        logits_lge = torch.randn((8, 3, 128, 128), requires_grad=True)
        logits_t2 = torch.randn((8, 3, 128, 128), requires_grad=True)
        logits_map = torch.randn((8, 3, 128, 128), requires_grad=True)

        seg = {
            "C0": torch.softmax(logits_c0, dim=1),
            "LGE": torch.softmax(logits_lge, dim=1),
            "T2": torch.softmax(logits_t2, dim=1),
            "mapping": torch.softmax(logits_map, dim=1),
        }

        loss_seg, loss_inv, loss_inc, total_loss = self.criterion(seg, targets)

        # Verify finite scalar losses
        self.assertTrue(torch.isfinite(loss_seg).item(), "loss_seg is not finite")
        self.assertTrue(torch.isfinite(loss_inv).item(), "loss_inv is not finite")
        self.assertTrue(torch.isfinite(loss_inc).item(), "loss_inc is not finite")
        self.assertTrue(torch.isfinite(total_loss).item(), "total_loss is not finite")
        self.assertGreater(total_loss.item(), 0.0)

        # Backward pass
        total_loss.backward()

        for name, logit in [("C0", logits_c0), ("LGE", logits_lge), ("T2", logits_t2), ("mapping", logits_map)]:
            self.assertIsNotNone(logit.grad, f"Gradient for {name} is None")
            self.assertTrue(torch.isfinite(logit.grad).all().item(), f"Gradient for {name} contains non-finite values")
            self.assertGreater(logit.grad.abs().sum().item(), 0.0, f"Gradient for {name} is all zeros")

    def test_single_voxel_lesion_slice_by_slice_evaluation(self):
        """Test slice-by-slice evaluation on single-voxel slice and completely empty slices."""
        vol = torch.zeros((128, 128, 8), dtype=torch.int64)
        vol[64, 64, 4] = 3 # 1 voxel scar
        vol[60, 60, 4] = 2 # 1 voxel edema

        for z in range(8):
            slice_2d = vol[:, :, z] # (128, 128)
            targets_slice = canonical_to_myops_targets(slice_2d)

            # Targets are (1, 3, 128, 128)
            self.assertEqual(targets_slice["scar"].shape, (1, 3, 128, 128))

            logits_c0 = torch.randn((1, 3, 128, 128), requires_grad=True)
            logits_lge = torch.randn((1, 3, 128, 128), requires_grad=True)
            logits_t2 = torch.randn((1, 3, 128, 128), requires_grad=True)
            logits_map = torch.randn((1, 3, 128, 128), requires_grad=True)

            seg = {
                "C0": torch.softmax(logits_c0, dim=1),
                "LGE": torch.softmax(logits_lge, dim=1),
                "T2": torch.softmax(logits_t2, dim=1),
                "mapping": torch.softmax(logits_map, dim=1),
            }

            _, _, _, total_loss = self.criterion(seg, targets_slice)
            self.assertTrue(torch.isfinite(total_loss).item(), f"Slice {z}: non-finite loss")

            total_loss.backward()
            self.assertTrue(torch.isfinite(logits_lge.grad).all().item(), f"Slice {z}: non-finite grad")


class TestVolumetricConflictingLogits(unittest.TestCase):
    """Mission Item 2: Volumetric conflicting logits priority resolution."""

    def test_conflicting_logits_unbatched_3d_volume(self):
        """Test priority resolution on 3D unbatched volume (3, 128, 128, 8).

        Priority rule: Scar (3) > Edema (2) > Normal Myo (1) > BG (0).
        """
        shape = (3, 128, 128, 8)

        # Baseline: All branches predict Background (channel 0) with logit 0.0
        scar_logits = torch.zeros(shape, dtype=torch.float32)
        edema_logits = torch.zeros(shape, dtype=torch.float32)
        cardiac_logits = torch.zeros(shape, dtype=torch.float32)
        mapping_logits = torch.zeros(shape, dtype=torch.float32)

        # Default: Background channel 0 has high logit
        scar_logits[0, ...] = 50.0
        edema_logits[0, ...] = 50.0
        cardiac_logits[0, ...] = 50.0
        mapping_logits[0, ...] = 50.0

        # Scenario 1 (Conflict: ALL branches predict their positive classes with +1000.0)
        # Scar channel 2, Edema channel 2, Cardiac channel 1 all +1000.0
        # Expected: Scar (3) wins over Edema and Myo
        pt1 = (slice(None), 20, 20, 2)
        scar_logits[2, 20, 20, 2] = 1000.0
        edema_logits[2, 20, 20, 2] = 1000.0
        cardiac_logits[1, 20, 20, 2] = 1000.0
        mapping_logits[2, 20, 20, 2] = 1000.0

        # Scenario 2 (Conflict: Edema channel 2 and Cardiac channel 1 both +1000.0; Scar predicts BG)
        # Expected: Edema (2) wins over Myo (1)
        edema_logits[2, 30, 30, 3] = 1000.0
        cardiac_logits[1, 30, 30, 3] = 1000.0

        # Scenario 3 (Conflict: Cardiac channel 1 is +1000.0; Scar and Edema predict BG)
        # Expected: Normal Myo (1) wins over BG (0)
        cardiac_logits[1, 40, 40, 4] = 1000.0

        # Scenario 4 (No positive class: all predict BG)
        # Expected: Background (0) at (50, 50, 5)

        outputs = {
            "scar": scar_logits,
            "edema": edema_logits,
            "cardiac": cardiac_logits,
            "mapping": mapping_logits,
        }

        canonical = myops_outputs_to_canonical(outputs)

        # Shape verification: unbatched 3D volume produces (128, 128, 8)
        self.assertEqual(canonical.shape, (128, 128, 8))
        self.assertEqual(canonical.dtype, torch.int64)

        # Value verifications
        self.assertEqual(canonical[20, 20, 2].item(), 3, "Scar did not win over Edema & Myo at extreme +1000.0!")
        self.assertEqual(canonical[30, 30, 3].item(), 2, "Edema did not win over Myo at extreme +1000.0!")
        self.assertEqual(canonical[40, 40, 4].item(), 1, "Normal Myo did not win over BG at extreme +1000.0!")
        self.assertEqual(canonical[50, 50, 5].item(), 0, "Background failed to resolve!")

    def test_conflicting_logits_batched_3d_volume(self):
        """Test priority resolution on 4D batched 3D volume (2, 3, 128, 128, 8)."""
        shape = (2, 3, 128, 128, 8)

        scar_logits = torch.zeros(shape, dtype=torch.float32)
        edema_logits = torch.zeros(shape, dtype=torch.float32)
        cardiac_logits = torch.zeros(shape, dtype=torch.float32)
        mapping_logits = torch.zeros(shape, dtype=torch.float32)

        # Default BG = 50.0
        scar_logits[:, 0, ...] = 50.0
        edema_logits[:, 0, ...] = 50.0
        cardiac_logits[:, 0, ...] = 50.0
        mapping_logits[:, 0, ...] = 50.0

        # Batch item 0: Scar priority
        scar_logits[0, 2, 60, 60, 2] = 1000.0
        edema_logits[0, 2, 60, 60, 2] = 1000.0
        cardiac_logits[0, 1, 60, 60, 2] = 1000.0

        # Batch item 1: Edema priority
        edema_logits[1, 2, 70, 70, 3] = 1000.0
        cardiac_logits[1, 1, 70, 70, 3] = 1000.0

        outputs = {
            "scar": scar_logits,
            "edema": edema_logits,
            "cardiac": cardiac_logits,
            "mapping": mapping_logits,
        }

        canonical = myops_outputs_to_canonical(outputs)

        self.assertEqual(canonical.shape, (2, 128, 128, 8))
        self.assertEqual(canonical[0, 60, 60, 2].item(), 3)
        self.assertEqual(canonical[1, 70, 70, 3].item(), 2)


class TestRoundTripShapeInvariance(unittest.TestCase):
    """Mission Item 3: Round-trip shape invariance across dimensions."""

    def test_round_trip_2d_slice(self):
        """2D single slice (128, 128)."""
        label = torch.zeros((128, 128), dtype=torch.int64)
        label[30:50, 30:50] = 1
        label[40:48, 40:48] = 2
        label[44:46, 44:46] = 3

        targets = canonical_to_myops_targets(label)
        recovered = myops_outputs_to_canonical(targets)

        self.assertEqual(recovered.shape, (128, 128))
        self.assertEqual(recovered.dtype, torch.int64)
        torch.testing.assert_close(recovered, label)

    def test_round_trip_batched_2d_slices(self):
        """Batched 2D slices (B, 128, 128) with various batch sizes."""
        for B in [1, 4, 7]:
            label = torch.zeros((B, 128, 128), dtype=torch.int64)
            label[:, 20:40, 20:40] = 1
            label[:, 30:35, 30:35] = 2
            label[:, 32:34, 32:34] = 3

            targets = canonical_to_myops_targets(label)
            recovered = myops_outputs_to_canonical(targets)

            self.assertEqual(recovered.shape, (B, 128, 128))
            self.assertEqual(recovered.dtype, torch.int64)
            torch.testing.assert_close(recovered, label)

    def test_round_trip_3d_unbatched_volume(self):
        """3D unbatched volume (128, 128, 8)."""
        label = torch.zeros((128, 128, 8), dtype=torch.int64)
        label[20:40, 20:40, 2:6] = 1
        label[30:35, 30:35, 3:5] = 2
        label[32:34, 32:34, 4:5] = 3

        targets = canonical_to_myops_targets(label)
        recovered = myops_outputs_to_canonical(targets)

        self.assertEqual(recovered.shape, (128, 128, 8))
        self.assertEqual(recovered.dtype, torch.int64)
        torch.testing.assert_close(recovered, label)

    def test_round_trip_3d_batched_volume(self):
        """Batched 3D volume (B, 128, 128, 8) with B=2, 3."""
        for B in [2, 3]:
            label = torch.zeros((B, 128, 128, 8), dtype=torch.int64)
            label[:, 20:40, 20:40, 2:6] = 1
            label[:, 30:35, 30:35, 3:5] = 2
            label[:, 32:34, 32:34, 4:5] = 3

            targets = canonical_to_myops_targets(label)
            recovered = myops_outputs_to_canonical(targets)

            self.assertEqual(recovered.shape, (B, 128, 128, 8))
            self.assertEqual(recovered.dtype, torch.int64)
            torch.testing.assert_close(recovered, label)


if __name__ == "__main__":
    unittest.main()
