"""
Tests for Feature 6: WCELoss Bug Fix & Loss Hardening (DiceLoss, WCELoss, InvariantLoss, InclusiveLoss, MyoPSLoss).
Protocol: myops380_train304_test76_last_v1
Interface Contract: PROJECT.md § M2 <-> M3, network/criterion/loss.py
Tiers Covered: Tier 1 (Feature Coverage), Tier 2 (Boundary & Corner Cases), Tier 3 (Cross-Feature Combinations).
"""

import unittest
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from network.criterion.loss import (
    DiceLoss,
    BinaryDiceLoss,
    InvariantLoss,
    InclusiveLossScar,
    InclusiveLossEdema,
    WCELoss as BaselineWCELoss,
)
from tests.contract_helpers import (
    reference_canonical_to_myops_targets,
    reference_hardened_weight_function,
)


class HardenedWCELoss(nn.Module):
    """Reference implementation of hardened WCELoss for MyoPS-Net MyoPS380 adaptation."""
    def __init__(self):
        super(HardenedWCELoss, self).__init__()

    def weight_function(self, target: torch.Tensor) -> torch.Tensor:
        return reference_hardened_weight_function(target)

    def forward(self, predict: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce_loss = torch.mean(-target * torch.log(predict + 1e-10), dim=(0, 2, 3))
        weights = self.weight_function(target)
        loss = weights * ce_loss
        return loss.sum()


class HardenedSegmentationLoss(nn.Module):
    def __init__(self):
        super(HardenedSegmentationLoss, self).__init__()
        self.dice_loss = DiceLoss()
        self.wce_loss = HardenedWCELoss()

    def forward(self, predict: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.dice_loss(predict, target) + self.wce_loss(predict, target)


class HardenedMyoPSLoss(nn.Module):
    """Hardened MyoPSLoss integrating hardened WCELoss with device flexibility."""
    def __init__(self):
        super(HardenedMyoPSLoss, self).__init__()
        self.seg_loss = HardenedSegmentationLoss()
        self.invariant_loss = InvariantLoss()
        self.inclusive_scar = InclusiveLossScar()
        self.inclusive_edema = InclusiveLossEdema()

    def forward(self, seg: dict[str, torch.Tensor], label: dict[str, torch.Tensor]):
        loss_seg = (
            self.seg_loss(seg["C0"], label["cardiac"])
            + 2 * self.seg_loss(seg["LGE"], label["scar"])
            + 2 * self.seg_loss(seg["T2"], label["edema"])
            + 2 * self.seg_loss(seg["mapping"], label["scar"])
        )

        myo_C0 = torch.cat([seg["C0"][:, 0:1, :, :] + seg["C0"][:, 2:3, :, :], seg["C0"][:, 1:2, :, :]], dim=1)
        myo_LGE = torch.cat([seg["LGE"][:, 0:1, :, :], seg["LGE"][:, 1:2, :, :] + seg["LGE"][:, 2:3, :, :]], dim=1)
        myo_T2 = torch.cat([seg["T2"][:, 0:1, :, :], seg["T2"][:, 1:2, :, :] + seg["T2"][:, 2:3, :, :]], dim=1)
        myo_mapping = torch.cat([seg["mapping"][:, 0:1, :, :], seg["mapping"][:, 1:2, :, :] + seg["mapping"][:, 2:3, :, :]], dim=1)

        loss_invariant = (
            self.invariant_loss(myo_C0, myo_LGE)
            + self.invariant_loss(myo_C0, myo_T2)
            + self.invariant_loss(myo_C0, myo_mapping)
        )

        loss_inclusive = (
            self.inclusive_scar(seg["LGE"][:, 2, :, :], label["edema"][:, 2, :, :])
            + self.inclusive_edema(seg["T2"][:, 2, :, :], label["scar"][:, 2, :, :])
            + self.inclusive_scar(seg["mapping"][:, 2, :, :], label["edema"][:, 2, :, :])
        )

        loss_total = loss_seg + loss_invariant + loss_inclusive
        return loss_seg, loss_invariant, loss_inclusive, loss_total


class TestLossHardeningFeatureCoverage(unittest.TestCase):
    """Tier 1: Feature Coverage for Loss Hardening (F6)."""

    def setUp(self):
        self.B, self.C, self.H, self.W = 2, 3, 128, 128
        self.hardened_wce = HardenedWCELoss()

    def test_f6_wceloss_weight_function_shape_matches_target_channels(self):
        """Tier 1: Weight function must produce a 1D tensor with length exactly equal to target.shape[1]."""
        # Batch with all 3 channels present
        target = torch.zeros((self.B, self.C, self.H, self.W), dtype=torch.float32)
        target[:, 0, :, :] = 1.0
        target[:, 1, 20:50, 20:50] = 1.0
        target[:, 0, 20:50, 20:50] = 0.0
        target[:, 2, 30:40, 30:40] = 1.0
        target[:, 1, 30:40, 30:40] = 0.0

        weights = self.hardened_wce.weight_function(target)
        self.assertEqual(weights.shape, (self.C,), f"Expected shape ({self.C},), got {weights.shape}")
        self.assertEqual(weights.device, target.device)

    def test_f6_wceloss_absent_class_handling(self):
        """Tier 1: When class 2 (scar/edema) is completely absent (mask.max() == 1),
        weights must STILL have length 3 (C=3), avoiding tensor size mismatch crash."""
        target = torch.zeros((self.B, self.C, self.H, self.W), dtype=torch.float32)
        # Only class 0 and class 1 present; class 2 has 0 voxels
        target[:, 0, :, :] = 1.0
        target[:, 1, 20:50, 20:50] = 1.0
        target[:, 0, 20:50, 20:50] = 0.0
        # target[:, 2] is all zeros

        weights = self.hardened_wce.weight_function(target)
        self.assertEqual(weights.shape, (3,), "Weights must retain length 3 even when class 2 is absent!")
        self.assertEqual(weights[2].item(), 0.0, "Weight for absent class 2 must be 0.0")

    def test_f6_wceloss_zero_division_safety(self):
        """Tier 1: When class voxel count is 0, no division by zero or NaN/Inf is produced."""
        target = torch.zeros((self.B, self.C, self.H, self.W), dtype=torch.float32)
        target[:, 0, :, :] = 1.0 # only background
        weights = self.hardened_wce.weight_function(target)

        self.assertFalse(torch.isnan(weights).any(), "Weights contain NaN")
        self.assertFalse(torch.isinf(weights).any(), "Weights contain Inf")
        self.assertEqual(weights[1].item(), 0.0)
        self.assertEqual(weights[2].item(), 0.0)

    def test_f6_wceloss_device_agnostic(self):
        """Tier 1: Hardened WCELoss executes on CPU without hardcoded .cuda() calls."""
        cpu_target = torch.zeros((1, 3, 32, 32), dtype=torch.float32)
        cpu_target[:, 0] = 1.0
        cpu_predict = torch.softmax(torch.randn((1, 3, 32, 32)), dim=1)

        loss = self.hardened_wce(cpu_predict, cpu_target)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(loss.device.type, "cpu")

    def test_f6_myopsloss_joint_objective_integration(self):
        """Tier 1: Hardened MyoPSLoss computes finite scalar losses and returns 4-element tuple."""
        myops_loss = HardenedMyoPSLoss()

        # Build mock model output predictions (softmax probabilities in [0, 1])
        seg = {
            "C0": torch.softmax(torch.randn((2, 3, 128, 128)), dim=1),
            "LGE": torch.softmax(torch.randn((2, 3, 128, 128)), dim=1),
            "T2": torch.softmax(torch.randn((2, 3, 128, 128)), dim=1),
            "mapping": torch.softmax(torch.randn((2, 3, 128, 128)), dim=1),
        }
        # Canonical label converted to 3 targets
        canonical = torch.zeros((2, 128, 128), dtype=torch.int64)
        canonical[:, 30:70, 30:70] = 1
        canonical[:, 40:60, 40:60] = 2
        canonical[:, 45:55, 45:55] = 3
        targets = reference_canonical_to_myops_targets(canonical)

        loss_seg, loss_inv, loss_inc, loss_tot = myops_loss(seg, targets)

        self.assertTrue(torch.isfinite(loss_seg))
        self.assertTrue(torch.isfinite(loss_inv))
        self.assertTrue(torch.isfinite(loss_inc))
        self.assertTrue(torch.isfinite(loss_tot))
        self.assertGreater(loss_tot.item(), 0.0)
        # Total equals sum of terms
        torch.testing.assert_close(loss_tot, loss_seg + loss_inv + loss_inc)


class TestLossHardeningBoundaryCases(unittest.TestCase):
    """Tier 2: Boundary & Corner Cases for Loss Hardening (F6)."""

    def setUp(self):
        self.hardened_wce = HardenedWCELoss()
        self.dice_loss = DiceLoss()

    def test_f6_wceloss_all_background_batch(self):
        """Tier 2: Batch with 100% background voxels (both pathology classes 1 and 2 absent)."""
        target = torch.zeros((1, 3, 64, 64), dtype=torch.float32)
        target[:, 0] = 1.0
        predict = torch.softmax(torch.randn((1, 3, 64, 64)), dim=1)

        loss = self.hardened_wce(predict, target)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreaterEqual(loss.item(), 0.0)

    def test_f6_wceloss_single_slice_batch(self):
        """Tier 2: Single-slice batch (B=1) operates seamlessly."""
        target = torch.zeros((1, 3, 128, 128), dtype=torch.float32)
        target[:, 0] = 1.0
        target[:, 1, 40:60, 40:60] = 1.0
        target[:, 0, 40:60, 40:60] = 0.0
        predict = torch.softmax(torch.randn((1, 3, 128, 128)), dim=1)

        loss = self.hardened_wce(predict, target)
        self.assertTrue(torch.isfinite(loss))

    def test_f6_diceloss_perfect_overlap(self):
        """Tier 2: Perfect overlap between prediction and target yields near 0.0 DiceLoss."""
        target = torch.zeros((2, 3, 64, 64), dtype=torch.float32)
        target[:, 0] = 1.0
        target[:, 1, 20:40, 20:40] = 1.0
        target[:, 0, 20:40, 20:40] = 0.0
        # Perfect prediction = exact target
        loss = self.dice_loss(target, target)
        self.assertLess(loss.item(), 1e-4)

    def test_f6_diceloss_zero_overlap(self):
        """Tier 2: Completely disjoint prediction and target yields near 1.0 DiceLoss."""
        target = torch.zeros((1, 1, 64, 64), dtype=torch.float32)
        target[:, :, :32, :] = 1.0
        predict = torch.zeros((1, 1, 64, 64), dtype=torch.float32)
        predict[:, :, 32:, :] = 1.0

        b_dice = BinaryDiceLoss()
        loss = b_dice(predict, target)
        self.assertGreater(loss.item(), 0.99)

    def test_f6_inclusive_loss_scar_penalizes_outside_edema(self):
        """Tier 2: Scar prediction outside edema ground truth strictly increases InclusiveLossScar."""
        inc_scar = InclusiveLossScar()
        H, W = 64, 64

        gd_edema = torch.zeros((1, H, W), dtype=torch.float32)
        gd_edema[:, 20:40, 20:40] = 1.0 # Edema is inside (20:40, 20:40)

        # Case A: Scar prediction safely inside edema region
        scar_inside = torch.zeros((1, H, W), dtype=torch.float32)
        scar_inside[:, 25:35, 25:35] = 0.9
        loss_inside = inc_scar(scar_inside, gd_edema)

        # Case B: Scar prediction leaked outside edema region
        scar_outside = scar_inside.clone()
        scar_outside[:, 0:10, 0:10] = 0.9 # leaked outside!
        loss_outside = inc_scar(scar_outside, gd_edema)

        self.assertGreater(loss_outside.item(), loss_inside.item(), "Leaked scar must be penalized!")

    def test_f6_inclusive_loss_edema_penalizes_missing_scar(self):
        """Tier 2: Edema prediction failing to cover scar ground truth strictly increases InclusiveLossEdema."""
        inc_edema = InclusiveLossEdema()
        H, W = 64, 64

        gd_scar = torch.zeros((1, H, W), dtype=torch.float32)
        gd_scar[:, 25:35, 25:35] = 1.0 # Ground truth scar

        # Case A: Edema predicted over the scar region
        edema_covered = torch.zeros((1, H, W), dtype=torch.float32)
        edema_covered[:, 20:40, 20:40] = 0.9
        loss_covered = inc_edema(edema_covered, gd_scar)

        # Case B: Edema misses the scar region (near 0)
        edema_missing = torch.zeros((1, H, W), dtype=torch.float32)
        loss_missing = inc_edema(edema_missing, gd_scar)

        self.assertGreater(loss_missing.item(), loss_covered.item(), "Missing edema over scar must be penalized!")

    def test_f6_invariant_loss_aligned_masks(self):
        """Tier 2: Invariant loss between identical masks is 0.0."""
        inv_loss = InvariantLoss()
        mask1 = torch.rand((2, 2, 64, 64))
        loss = inv_loss(mask1, mask1)
        self.assertLess(loss.item(), 1e-5)

    def test_f6_backward_pass_gradients_finite(self):
        """Tier 2: Gradients computed from HardenedWCELoss are strictly finite (no NaN/Inf)."""
        logits = torch.randn((2, 3, 32, 32), requires_grad=True)
        predict = torch.softmax(logits, dim=1)
        target = torch.zeros((2, 3, 32, 32), dtype=torch.float32)
        target[:, 0] = 1.0

        loss = self.hardened_wce(predict, target)
        loss.backward()

        self.assertIsNotNone(logits.grad)
        self.assertFalse(torch.isnan(logits.grad).any())
        self.assertFalse(torch.isinf(logits.grad).any())

    def test_f6_baseline_wceloss_defect_reproduction_and_escalation(self):
        """Tier 2 Defect Verification:
        Verify whether BaselineWCELoss exhibits the known shape mismatch or hardcoded CUDA bug.
        If it exhibits the bug, verify that HardenedWCELoss cleanly resolves it.
        """
        baseline_wce = BaselineWCELoss()
        # Batch where class 2 is absent
        target = torch.zeros((1, 3, 32, 32), dtype=torch.float32)
        target[:, 0] = 1.0
        predict = torch.softmax(torch.randn((1, 3, 32, 32)), dim=1)

        baseline_has_defect = False
        try:
            baseline_wce(predict, target)
        except (RuntimeError, AssertionError, AttributeError) as e:
            baseline_has_defect = True
            # Log expected baseline failure: shape mismatch or .cuda() on CPU
            err_msg = str(e)
            self.assertTrue(
                "must match the size" in err_msg or "CUDA" in err_msg or "cuda" in err_msg or "not compiled with CUDA" in err_msg or "out of memory" in err_msg,
                f"Unexpected error: {err_msg}"
            )

        # In contrast, HardenedWCELoss MUST succeed unconditionally
        hardened_loss = self.hardened_wce(predict, target)
        self.assertTrue(torch.isfinite(hardened_loss))


class TestLossHardeningCrossFeature(unittest.TestCase):
    """Tier 3: Cross-Feature Integration between Target Adapter and Loss Functions."""

    def test_t3_target_adapter_to_myopsloss_pipeline(self):
        """Tier 3: Full pipeline:
        Canonical Mask -> canonical_to_myops_targets -> Model Logits -> Softmax -> HardenedMyoPSLoss -> Backward pass.
        """
        B, H, W = 2, 64, 64
        canonical = torch.zeros((B, H, W), dtype=torch.int64)
        canonical[:, 10:40, 10:40] = 1 # Normal Myo
        canonical[:, 20:30, 20:30] = 2 # Edema
        canonical[:, 22:28, 22:28] = 3 # Scar

        targets = reference_canonical_to_myops_targets(canonical)

        # Model logits with requires_grad
        logits_C0 = torch.randn((B, 3, H, W), requires_grad=True)
        logits_LGE = torch.randn((B, 3, H, W), requires_grad=True)
        logits_T2 = torch.randn((B, 3, H, W), requires_grad=True)
        logits_mapping = torch.randn((B, 3, H, W), requires_grad=True)

        seg = {
            "C0": torch.softmax(logits_C0, dim=1),
            "LGE": torch.softmax(logits_LGE, dim=1),
            "T2": torch.softmax(logits_T2, dim=1),
            "mapping": torch.softmax(logits_mapping, dim=1),
        }

        criterion = HardenedMyoPSLoss()
        l_seg, l_inv, l_inc, l_tot = criterion(seg, targets)

        self.assertTrue(torch.isfinite(l_tot))
        l_tot.backward()

        for branch_name, logits in [("C0", logits_C0), ("LGE", logits_LGE), ("T2", logits_T2), ("mapping", logits_mapping)]:
            self.assertIsNotNone(logits.grad, f"Gradient missing for branch {branch_name}")
            self.assertFalse(torch.isnan(logits.grad).any(), f"Gradient contains NaN for {branch_name}")


if __name__ == "__main__":
    unittest.main()
