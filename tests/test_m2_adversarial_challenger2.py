"""
Comprehensive Empirical Adversarial Test Suite for Milestone 2 Challenger 2:
Pipeline Integration & Shape Invariance Adversary.
"""
import sys
import unittest
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np

from dataset.myops380_dataset import build_myops380_dataloader
from dataset.target_adapter import canonical_to_myops_targets, myops_outputs_to_canonical, TargetAdapter
from network.model import MyoPSNet
from network.criterion.loss import MyoPSLoss


class TestPipelineIntegrationRealData(unittest.TestCase):
    """Challenge 2.a: Empirical stress test on real data pipeline."""

    def test_real_data_pipeline_12_batches_forward_backward(self):
        """Fetch 12 real training batches, adapt targets, forward MyoPSNet, compute MyoPSLoss, backward."""
        loader = build_myops380_dataloader(batch_size=4, num_workers=0)
        model = MyoPSNet(in_chs=(5, 2, 2, 3), out_chs=(3, 3, 3, 3))
        criterion = MyoPSLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        batch_count = 0
        for i, batch in enumerate(loader):
            if batch_count >= 12:
                break

            C0 = batch["C0"]
            LGE = batch["LGE"]
            T2 = batch["T2"]
            T1m = batch["T1m"]
            T2starm = batch["T2starm"]
            label = batch["label"]

            # Shapes verification
            self.assertEqual(C0.shape, (4, 1, 128, 128))
            self.assertEqual(LGE.shape, (4, 1, 128, 128))
            self.assertEqual(T2.shape, (4, 1, 128, 128))
            self.assertEqual(T1m.shape, (4, 1, 128, 128))
            self.assertEqual(T2starm.shape, (4, 1, 128, 128))
            self.assertEqual(label.shape, (4, 128, 128))

            # Target adaptation
            targets = canonical_to_myops_targets(label)
            self.assertIn("cardiac", targets)
            self.assertIn("scar", targets)
            self.assertIn("edema", targets)

            self.assertEqual(targets["cardiac"].shape, (4, 3, 128, 128))
            self.assertEqual(targets["scar"].shape, (4, 3, 128, 128))
            self.assertEqual(targets["edema"].shape, (4, 3, 128, 128))

            # Forward pass
            optimizer.zero_grad()
            outputs = model(C0, LGE, T2, T1m, T2starm)
            self.assertEqual(len(outputs), 4)

            # Loss computation
            loss_seg, loss_inv, loss_inc, total_loss = criterion(outputs, targets)

            self.assertTrue(torch.isfinite(loss_seg).item(), f"Batch {i}: loss_seg non-finite")
            self.assertTrue(torch.isfinite(loss_inv).item(), f"Batch {i}: loss_inv non-finite")
            self.assertTrue(torch.isfinite(loss_inc).item(), f"Batch {i}: loss_inc non-finite")
            self.assertTrue(torch.isfinite(total_loss).item(), f"Batch {i}: total_loss non-finite")

            # Backward pass
            total_loss.backward()

            # Verify finite gradients on all parameters
            has_grads = False
            for name, param in model.named_parameters():
                if param.requires_grad:
                    self.assertIsNotNone(param.grad, f"Batch {i}: param {name} missing grad")
                    self.assertTrue(torch.isfinite(param.grad).all().item(), f"Batch {i}: param {name} non-finite grad")
                    has_grads = True
            self.assertTrue(has_grads)

            optimizer.step()
            batch_count += 1

        self.assertEqual(batch_count, 12)


class TestShapeInvarianceAdversary(unittest.TestCase):
    """Challenge 2.b: Shape Invariance on Target Adapter across dimensions."""

    def test_2d_slice_shape_handling(self):
        """2D single slice (H, W)."""
        label_2d = torch.randint(0, 4, (128, 128), dtype=torch.int64)
        targets_2d = canonical_to_myops_targets(label_2d)
        
        # Check target dictionary shapes
        for k in ["scar", "edema", "cardiac"]:
            self.assertIn(k, targets_2d)
            # Has 4 dims (1, 3, 128, 128)
            self.assertEqual(targets_2d[k].shape, (1, 3, 128, 128))

    def test_batched_2d_slice_shape_handling(self):
        """Batched 2D slices (B, H, W)."""
        label_batched = torch.randint(0, 4, (5, 128, 128), dtype=torch.int64)
        targets_batched = canonical_to_myops_targets(label_batched)
        
        for k in ["scar", "edema", "cardiac"]:
            self.assertEqual(targets_batched[k].shape, (5, 3, 128, 128))

    def test_3d_volume_batched_shape_handling(self):
        """3D batched volume (B, H, W, D)."""
        label_3d = torch.randint(0, 4, (2, 128, 128, 8), dtype=torch.int64)
        # Empirical test: does canonical_to_myops_targets support (B, H, W, D)?
        try:
            targets_3d = canonical_to_myops_targets(label_3d)
            # If supported, expected shape is (B, 3, H, W, D)
            self.assertEqual(targets_3d["scar"].shape, (2, 3, 128, 128, 8))
        except ValueError as e:
            # Capture exact error
            raise AssertionError(f"canonical_to_myops_targets failed on 3D volume (B, H, W, D): {e}")

    def test_3d_volume_unbatched_shape_handling(self):
        """3D unbatched volume (H, W, D)."""
        label_vol = torch.randint(0, 4, (128, 128, 8), dtype=torch.int64)
        targets_vol = canonical_to_myops_targets(label_vol)
        # If treated as (H, W, D), the channel dimension must be inserted at dim 0: (3, 128, 128, 8)
        # or with a batch dim: (1, 3, 128, 128, 8)
        # Current implementation assumes (B, H, W) and produces (128, 3, 128, 8)!
        self.assertNotEqual(
            targets_vol["scar"].shape,
            (128, 3, 128, 8),
            "Adapter incorrectly treated H (128) as batch dimension B!"
        )


class TestRoundTripConsistency(unittest.TestCase):
    """Challenge 2.c: Round-trip consistency between canonical_to_myops_targets and myops_outputs_to_canonical."""

    def test_round_trip_batched_2d(self):
        """Batched 2D (B, H, W) exact recovery."""
        B, H, W = 4, 128, 128
        label = torch.zeros((B, H, W), dtype=torch.int64)
        label[:, 20:60, 20:60] = 1
        label[:, 35:50, 35:50] = 2
        label[:, 40:45, 40:45] = 3

        targets = canonical_to_myops_targets(label)
        recovered = myops_outputs_to_canonical(targets)

        self.assertEqual(recovered.shape, label.shape)
        self.assertEqual(recovered.dtype, torch.int64)
        torch.testing.assert_close(recovered, label)

    def test_round_trip_2d_single_slice_shape_preservation(self):
        """2D single slice (H, W) shape preservation."""
        label_2d = torch.zeros((128, 128), dtype=torch.int64)
        label_2d[20:60, 20:60] = 1
        label_2d[35:50, 35:50] = 2
        label_2d[40:45, 40:45] = 3

        targets = canonical_to_myops_targets(label_2d)
        recovered = myops_outputs_to_canonical(targets)

        # Input was 2D (128, 128). Does recovered match (128, 128)?
        self.assertEqual(
            recovered.shape,
            label_2d.shape,
            f"Shape mismatch: input {label_2d.shape} vs recovered {recovered.shape}"
        )


if __name__ == "__main__":
    unittest.main()
