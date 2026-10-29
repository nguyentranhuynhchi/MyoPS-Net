"""
Tests for Feature 5: Target Adapter (Canonical 4-Class to 3-Target Dictionaries and Reverse Mapping).
Protocol: myops380_train304_test76_last_v1
Interface Contract: PROJECT.md § M2 <-> M3, M2/M3 <-> M4
Tiers Covered: Tier 1 (Feature Coverage), Tier 2 (Boundary & Corner Cases), Tier 3 (Cross-Feature Combinations).
"""

import unittest
import torch
import numpy as np

from tests.contract_helpers import (
    reference_canonical_to_myops_targets,
    reference_myops_outputs_to_canonical,
    try_import_target_adapter,
)

# Load implementation if available, otherwise use reference contract
_impl_forward, _impl_reverse = try_import_target_adapter()


def canonical_to_myops_targets(label: torch.Tensor) -> dict[str, torch.Tensor]:
    if _impl_forward is not None:
        return _impl_forward(label)
    return reference_canonical_to_myops_targets(label)


def myops_outputs_to_canonical(seg_outputs: dict[str, torch.Tensor]) -> torch.Tensor:
    if _impl_reverse is not None:
        return _impl_reverse(seg_outputs)
    return reference_myops_outputs_to_canonical(seg_outputs)


class TestTargetAdapterFeatureCoverage(unittest.TestCase):
    """Tier 1: Feature Coverage for Target Adapter (F5)."""

    def setUp(self):
        self.B, self.H, self.W = 2, 128, 128
        # Create a canonical label tensor with all 4 classes present
        # 0: BG, 1: Normal Myo, 2: Edema, 3: Scar
        self.label = torch.zeros((self.B, self.H, self.W), dtype=torch.int64)
        self.label[:, 20:60, 20:60] = 1
        self.label[:, 30:50, 30:50] = 2
        self.label[:, 35:45, 35:45] = 3

    def test_f5_canonical_0_background_mapping(self):
        """Tier 1: Canonical 0 (Background) must be mapped to Channel 0 in scar, edema, and cardiac."""
        targets = canonical_to_myops_targets(self.label)
        bg_mask = (self.label == 0)

        for target_name in ["scar", "edema", "cardiac"]:
            t = targets[target_name]
            # Channel 0 must be 1.0 where label is 0
            self.assertTrue(torch.all(t[:, 0][bg_mask] == 1.0), f"Failed for {target_name} channel 0 on BG")
            # Channels 1 and 2 must be 0.0 where label is 0
            self.assertTrue(torch.all(t[:, 1][bg_mask] == 0.0), f"Failed for {target_name} channel 1 on BG")
            self.assertTrue(torch.all(t[:, 2][bg_mask] == 0.0), f"Failed for {target_name} channel 2 on BG")

    def test_f5_canonical_1_normal_myo_mapping(self):
        """Tier 1: Canonical 1 (Normal Myocardium) must map to Channel 1 in scar, edema, and cardiac."""
        targets = canonical_to_myops_targets(self.label)
        myo_mask = (self.label == 1)

        for target_name in ["scar", "edema", "cardiac"]:
            t = targets[target_name]
            self.assertTrue(torch.all(t[:, 0][myo_mask] == 0.0))
            self.assertTrue(torch.all(t[:, 1][myo_mask] == 1.0), f"{target_name} channel 1 must be 1 for Normal Myo")
            self.assertTrue(torch.all(t[:, 2][myo_mask] == 0.0))

    def test_f5_canonical_2_edema_mapping(self):
        """Tier 1: Canonical 2 (Edema) must map to:
        - 'scar': Channel 1 (Non-scar Myo, since edema is part of myocardium but not scar)
        - 'edema': Channel 2 (Edema inclusive / Area at Risk)
        - 'cardiac': Channel 1 (Total Myo)
        """
        targets = canonical_to_myops_targets(self.label)
        edema_mask = (self.label == 2)

        # Scar target: Ch1=1, Ch0=0, Ch2=0
        self.assertTrue(torch.all(targets["scar"][:, 1][edema_mask] == 1.0))
        self.assertTrue(torch.all(targets["scar"][:, 2][edema_mask] == 0.0))

        # Edema target: Ch2=1 (inclusive lesion), Ch0=0, Ch1=0
        self.assertTrue(torch.all(targets["edema"][:, 2][edema_mask] == 1.0))
        self.assertTrue(torch.all(targets["edema"][:, 1][edema_mask] == 0.0))

        # Cardiac target: Ch1=1 (total myocardium)
        self.assertTrue(torch.all(targets["cardiac"][:, 1][edema_mask] == 1.0))
        self.assertTrue(torch.all(targets["cardiac"][:, 2][edema_mask] == 0.0))

    def test_f5_canonical_3_scar_mapping(self):
        """Tier 1: Canonical 3 (Scar) must map to:
        - 'scar': Channel 2 (Scar lesion)
        - 'edema': Channel 2 (Edema inclusive / Area at Risk includes scar!)
        - 'cardiac': Channel 1 (Total Myo)
        """
        targets = canonical_to_myops_targets(self.label)
        scar_mask = (self.label == 3)

        # Scar target: Ch2=1 (scar lesion)
        self.assertTrue(torch.all(targets["scar"][:, 2][scar_mask] == 1.0))
        self.assertTrue(torch.all(targets["scar"][:, 1][scar_mask] == 0.0))

        # Edema target: Ch2=1 (inclusive area at risk subsumes scar)
        self.assertTrue(torch.all(targets["edema"][:, 2][scar_mask] == 1.0))
        self.assertTrue(torch.all(targets["edema"][:, 1][scar_mask] == 0.0))

        # Cardiac target: Ch1=1 (total myocardium)
        self.assertTrue(torch.all(targets["cardiac"][:, 1][scar_mask] == 1.0))
        self.assertTrue(torch.all(targets["cardiac"][:, 2][scar_mask] == 0.0))

    def test_f5_reverse_mapping_reconstructs_canonical_labels(self):
        """Tier 1: Verify myops_outputs_to_canonical correctly maps branch argmax predictions back to 0..3."""
        # Synthesize ideal branch predictions
        targets = canonical_to_myops_targets(self.label)
        reconstructed = myops_outputs_to_canonical(targets)
        self.assertEqual(reconstructed.shape, self.label.shape)
        self.assertEqual(reconstructed.dtype, torch.int64)
        torch.testing.assert_close(reconstructed, self.label)


class TestTargetAdapterBoundaryCases(unittest.TestCase):
    """Tier 2: Boundary & Corner Cases for Target Adapter (F5)."""

    def setUp(self):
        self.H, self.W = 128, 128

    def test_f5_all_background_slice(self):
        """Tier 2: Slice with 100% background (no myocardium or lesion)."""
        label = torch.zeros((1, self.H, self.W), dtype=torch.int64)
        targets = canonical_to_myops_targets(label)

        for name in ["scar", "edema", "cardiac"]:
            t = targets[name]
            self.assertEqual(t[:, 0].sum().item(), float(self.H * self.W))
            self.assertEqual(t[:, 1].sum().item(), 0.0)
            self.assertEqual(t[:, 2].sum().item(), 0.0)

        recon = myops_outputs_to_canonical(targets)
        self.assertEqual(recon.sum().item(), 0)

    def test_f5_all_scar_slice(self):
        """Tier 2: Extreme theoretical slice with 100% scar."""
        label = torch.full((1, self.H, self.W), 3, dtype=torch.int64)
        targets = canonical_to_myops_targets(label)

        self.assertEqual(targets["scar"][:, 2].sum().item(), float(self.H * self.W))
        self.assertEqual(targets["edema"][:, 2].sum().item(), float(self.H * self.W))
        self.assertEqual(targets["cardiac"][:, 1].sum().item(), float(self.H * self.W))

        recon = myops_outputs_to_canonical(targets)
        self.assertTrue(torch.all(recon == 3))

    def test_f5_absent_scar_slice(self):
        """Tier 2: Slice with normal myo (1) and edema (2), but scar (3) is completely absent."""
        label = torch.zeros((1, self.H, self.W), dtype=torch.int64)
        label[:, 30:50, 30:50] = 1
        label[:, 40:45, 40:45] = 2 # only edema, no scar

        targets = canonical_to_myops_targets(label)
        self.assertEqual(targets["scar"][:, 2].sum().item(), 0.0, "Scar channel 2 must be empty")
        self.assertGreater(targets["edema"][:, 2].sum().item(), 0.0, "Edema channel 2 must be non-empty")

        recon = myops_outputs_to_canonical(targets)
        self.assertEqual((recon == 3).sum().item(), 0)
        self.assertEqual((recon == 2).sum().item(), (label == 2).sum().item())

    def test_f5_absent_edema_slice(self):
        """Tier 2: Slice with normal myo (1) and scar (3), but edema (2) is completely absent."""
        label = torch.zeros((1, self.H, self.W), dtype=torch.int64)
        label[:, 30:50, 30:50] = 1
        label[:, 40:45, 40:45] = 3 # only scar, no edema

        targets = canonical_to_myops_targets(label)
        self.assertGreater(targets["scar"][:, 2].sum().item(), 0.0)
        # Even though edema (2) is absent, edema inclusive channel 2 includes scar (3)!
        self.assertGreater(targets["edema"][:, 2].sum().item(), 0.0)

        recon = myops_outputs_to_canonical(targets)
        self.assertEqual((recon == 2).sum().item(), 0, "No isolated edema voxels should be reconstructed")
        self.assertEqual((recon == 3).sum().item(), (label == 3).sum().item())

    def test_f5_out_of_range_label_handling(self):
        """Tier 2: Labels outside 0..3 (e.g. 4 or 255) must not silently corrupt one-hot sum."""
        label = torch.zeros((1, self.H, self.W), dtype=torch.int64)
        label[0, 0, 0] = 255 # invalid label
        # In strict mode, Target Adapter does not assign invalid label to any valid class 1..3
        targets = canonical_to_myops_targets(label)
        # 255 should not be counted as scar or edema
        self.assertEqual(targets["scar"][0, 2, 0, 0].item(), 0.0)
        self.assertEqual(targets["edema"][0, 2, 0, 0].item(), 0.0)

    def test_f5_batch_dimension_preservation(self):
        """Tier 2: Verify adapter accepts 2D (H, W), 3D (B, H, W) and returns appropriate shapes."""
        label_2d = torch.zeros((self.H, self.W), dtype=torch.int64)
        targets_2d = canonical_to_myops_targets(label_2d)
        self.assertEqual(targets_2d["scar"].dim(), 4)
        self.assertEqual(targets_2d["scar"].shape[0], 1)

        label_3d = torch.zeros((3, self.H, self.W), dtype=torch.int64)
        targets_3d = canonical_to_myops_targets(label_3d)
        self.assertEqual(targets_3d["scar"].shape, (3, 3, self.H, self.W))

    def test_f5_one_hot_mutual_exclusivity(self):
        """Tier 2: Verify channels in 'scar' and 'edema' are mutually exclusive (sum to 1.0 everywhere)."""
        label = torch.randint(0, 4, (2, self.H, self.W), dtype=torch.int64)
        targets = canonical_to_myops_targets(label)

        # Scar channels: 0 (BG), 1 (Non-scar Myo), 2 (Scar) must partition the space exactly
        scar_sum = targets["scar"].sum(dim=1)
        torch.testing.assert_close(scar_sum, torch.ones_like(scar_sum))

        # Edema channels: 0 (BG), 1 (Normal Myo), 2 (Edema+Scar) must partition the space exactly
        edema_sum = targets["edema"].sum(dim=1)
        torch.testing.assert_close(edema_sum, torch.ones_like(edema_sum))

    def test_f5_device_and_dtype_preservation(self):
        """Tier 2: Output tensors remain float32 and preserve the input device."""
        label = torch.zeros((1, self.H, self.W), dtype=torch.int64)
        targets = canonical_to_myops_targets(label)
        for name in ["scar", "edema", "cardiac"]:
            self.assertEqual(targets[name].dtype, torch.float32)
            self.assertEqual(targets[name].device, label.device)

    def test_f5_reverse_mapping_with_empty_predictions(self):
        """Tier 2: If model outputs all zeros everywhere, reverse mapping defaults to background (0)."""
        seg_outputs = {
            "scar": torch.zeros((1, 3, self.H, self.W)),
            "edema": torch.zeros((1, 3, self.H, self.W)),
            "cardiac": torch.zeros((1, 3, self.H, self.W)),
        }
        recon = myops_outputs_to_canonical(seg_outputs)
        self.assertEqual(recon.sum().item(), 0)

    def test_f5_reverse_mapping_edema_subsumed_by_scar(self):
        """Tier 2: In conflict where both scar branch and edema branch predict lesion at the same voxel,
        scar has clinical and anatomical priority -> Canonical 3."""
        seg_outputs = {
            "scar": torch.zeros((1, 3, self.H, self.W)),
            "edema": torch.zeros((1, 3, self.H, self.W)),
            "cardiac": torch.zeros((1, 3, self.H, self.W)),
        }
        # Conflict at voxel (50, 50)
        seg_outputs["scar"][0, 2, 50, 50] = 10.0 # Scar
        seg_outputs["edema"][0, 2, 50, 50] = 10.0 # Edema
        recon = myops_outputs_to_canonical(seg_outputs)
        self.assertEqual(recon[0, 50, 50].item(), 3, "Scar must take priority over edema")


class TestTargetAdapterCrossFeature(unittest.TestCase):
    """Tier 3: Cross-Feature Combinations for Target Adapter."""

    def test_t3_target_adapter_round_trip_consistency(self):
        """Tier 3: Full round-trip: Canonical Mask -> Target Adapter -> Model Logits Simulation -> Reverse -> Match."""
        B, H, W = 4, 128, 128
        # Create diverse random canonical mask with realistic anatomical rings
        canonical_gt = torch.zeros((B, H, W), dtype=torch.int64)
        for b in range(B):
            canonical_gt[b, 30:90, 30:90] = 1 # Myo
            canonical_gt[b, 45:65, 45:65] = 2 # Edema
            canonical_gt[b, 50:60, 50:60] = 3 # Scar

        # Step 1: Forward adapt
        targets = canonical_to_myops_targets(canonical_gt)

        # Step 2: Convert one-hot targets to sharp logits (simulating a well-trained model)
        logits_dict = {}
        for k, v in targets.items():
            # Large positive logits where 1, negative where 0
            logits_dict[k] = (v * 20.0) - 10.0

        # Step 3: Reverse map back to canonical
        reconstructed = myops_outputs_to_canonical(logits_dict)

        # Assert identical recovery
        torch.testing.assert_close(reconstructed, canonical_gt)

    def test_t3_target_adapter_integration_with_model_output_shapes(self):
        """Tier 3: Ensure adapter accepts raw model outputs with 4 branches including 'mapping'."""
        B, H, W = 2, 128, 128
        model_outputs = {
            "cardiac": torch.randn((B, 3, H, W)),
            "scar": torch.randn((B, 3, H, W)),
            "edema": torch.randn((B, 3, H, W)),
            "mapping": torch.randn((B, 3, H, W)),
        }
        canonical_pred = myops_outputs_to_canonical(model_outputs)
        self.assertEqual(canonical_pred.shape, (B, H, W))
        self.assertEqual(canonical_pred.dtype, torch.int64)
        unique_classes = set(torch.unique(canonical_pred).tolist())
        self.assertTrue(unique_classes.issubset({0, 1, 2, 3}))


if __name__ == "__main__":
    unittest.main()
