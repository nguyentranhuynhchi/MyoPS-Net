"""Empirical Adversarial Stress Test Suite for Milestone 2: Target Adapter & Loss Hardening.

Protocol: myops380_train304_test76_last_v1
Interface Contract: PROJECT.md § M2 Target Adapter & Loss Hardening

Challenges:
a. All absent-class permutations:
   - Only BG (0)
   - Normal myo only (1)
   - Edema only (2)
   - Scar only (3)
   - Edema + scar only (2 + 3)
   - No scar (1 + 2)
   - No edema (1 + 3)
   - All classes present (0, 1, 2, 3)
   - Single-pixel / isolated lesion anomalies
   Verify canonical_to_myops_targets produces valid one-hot tensors, and MyoPSLoss
   evaluates without ZeroDivisionError, without dimension mismatch, producing finite scalar loss.

b. Backward gradient propagation:
   - Full MyoPSNet model forward-backward pass across all absent-class permutations.
   - Verify loss.backward() produces strictly finite, non-NaN, non-Inf gradients across all trainable parameters.

c. Reverse mapping conflict stress testing:
   - Exhaustive 81-state truth table over all {0, 1, 2}^4 branch argmax permutations.
   - Competing high logits across branches (e.g. +1000.0 simultaneously).
   - Strict priority enforcement: Scar (3) > Edema (2) > Normal Myo (1) > BG (0).
   - Tuple, list, dict, and direct tensor input formats.

d. Device execution & agnosticism:
   - CPU execution and CUDA execution (if available).
   - Strict device preservation verification (inputs on device D yield outputs on device D).
   - Verify absence of hardcoded .cuda() calls.

e. Input preservation and immutability:
   - Verify input label tensors and prediction dictionaries are NEVER mutated in-place.
   - Arbitrary spatial dimensions (32x32, 128x128, 64x128) and batch sizes (1, 3, 7, 16).
"""

import os
import unittest
import inspect
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset.target_adapter import (
    canonical_to_myops_targets,
    myops_outputs_to_canonical,
    TargetAdapter,
)
from network.criterion.loss import (
    MyoPSLoss,
    SegmentationLoss,
    DiceLoss,
    BinaryDiceLoss,
    WCELoss,
    InvariantLoss,
    InclusiveLossScar,
    InclusiveLossEdema,
)
from network.model import MyoPSNet


class TestMilestone2AbsentClassPermutations(unittest.TestCase):
    """Adversarial Challenge A: Comprehensive Absent-Class Stress Testing."""

    def setUp(self):
        self.device = torch.device("cpu")
        self.criterion = MyoPSLoss()
        self.H, self.W = 128, 128

    def _generate_synthetic_model_outputs(self, B: int, H: int, W: int, seed: int = 42) -> dict[str, torch.Tensor]:
        torch.manual_seed(seed)
        return {
            "C0": F.softmax(torch.randn((B, 3, H, W), device=self.device), dim=1),
            "LGE": F.softmax(torch.randn((B, 3, H, W), device=self.device), dim=1),
            "T2": F.softmax(torch.randn((B, 3, H, W), device=self.device), dim=1),
            "mapping": F.softmax(torch.randn((B, 3, H, W), device=self.device), dim=1),
        }

    def _evaluate_permutation(self, label: torch.Tensor, perm_name: str):
        """Helper to run Target Adapter and MyoPSLoss on a given label permutation."""
        B, H, W = label.shape

        # 1. Forward Target Adapter
        targets = canonical_to_myops_targets(label)

        # Invariant checks on target adapter output
        self.assertIn("scar", targets)
        self.assertIn("edema", targets)
        self.assertIn("cardiac", targets)

        for name in ["scar", "edema", "cardiac"]:
            t = targets[name]
            self.assertEqual(t.shape, (B, 3, H, W), f"Shape mismatch for {name} under {perm_name}")
            self.assertEqual(t.dtype, torch.float32, f"Dtype mismatch for {name} under {perm_name}")
            self.assertEqual(t.device, label.device, f"Device mismatch for {name} under {perm_name}")
            self.assertFalse(torch.isnan(t).any(), f"NaN in targets['{name}'] under {perm_name}")
            self.assertFalse(torch.isinf(t).any(), f"Inf in targets['{name}'] under {perm_name}")

        # Mutual exclusivity / one-hot sum invariants:
        # Scar: ch0 (BG) + ch1 (Non-scar Myo) + ch2 (Scar) == 1.0 everywhere
        scar_sum = targets["scar"].sum(dim=1)
        torch.testing.assert_close(scar_sum, torch.ones((B, H, W), dtype=torch.float32),
                                   msg=f"Scar target not valid one-hot under {perm_name}")

        # Edema: ch0 (BG) + ch1 (Normal Myo) + ch2 (Edema inclusive) == 1.0 everywhere
        edema_sum = targets["edema"].sum(dim=1)
        torch.testing.assert_close(edema_sum, torch.ones((B, H, W), dtype=torch.float32),
                                   msg=f"Edema target not valid one-hot under {perm_name}")

        # Cardiac: ch0 (BG) + ch1 (Total Myo) == 1.0 everywhere, and ch2 is strictly 0.0 everywhere
        cardiac_sum_01 = targets["cardiac"][:, :2].sum(dim=1)
        torch.testing.assert_close(cardiac_sum_01, torch.ones((B, H, W), dtype=torch.float32),
                                   msg=f"Cardiac target ch0+ch1 sum != 1 under {perm_name}")
        self.assertTrue(torch.all(targets["cardiac"][:, 2] == 0.0),
                        f"Cardiac target ch2 is not strictly 0.0 under {perm_name}")

        # 2. Forward Loss Evaluation
        outputs = self._generate_synthetic_model_outputs(B, H, W)
        try:
            loss_seg, loss_inv, loss_inc, loss_tot = self.criterion(outputs, targets)
        except Exception as e:
            self.fail(f"MyoPSLoss crashed with {type(e).__name__}: {e} under permutation {perm_name}")

        # Loss invariants: scalar, strictly finite, positive
        for name, val in [("loss_seg", loss_seg), ("loss_inv", loss_inv),
                          ("loss_inc", loss_inc), ("loss_tot", loss_tot)]:
            self.assertEqual(val.dim(), 0, f"{name} is not a 0-dim scalar under {perm_name}")
            self.assertTrue(torch.isfinite(val), f"{name} is non-finite ({val.item()}) under {perm_name}")
            self.assertFalse(torch.isnan(val), f"{name} is NaN under {perm_name}")

        self.assertGreater(loss_tot.item(), 0.0, f"Total loss must be strictly positive under {perm_name}")
        torch.testing.assert_close(loss_tot, loss_seg + loss_inv + loss_inc,
                                   msg=f"Total loss does not match sum of components under {perm_name}")

    def test_perm_1_all_background(self):
        """Permutation 1: Pure Background (0 only) across entire batch."""
        B = 4
        label = torch.zeros((B, self.H, self.W), dtype=torch.int64, device=self.device)
        self._evaluate_permutation(label, "all_background")

    def test_perm_2_normal_myo_only(self):
        """Permutation 2: Background (0) + Normal Myocardium (1) only; Edema and Scar completely absent."""
        B = 4
        label = torch.zeros((B, self.H, self.W), dtype=torch.int64, device=self.device)
        label[:, 30:80, 30:80] = 1
        self._evaluate_permutation(label, "normal_myo_only")

    def test_perm_3_edema_only(self):
        """Permutation 3: Background (0) + Edema (2) only; Normal Myo and Scar completely absent."""
        B = 4
        label = torch.zeros((B, self.H, self.W), dtype=torch.int64, device=self.device)
        label[:, 40:70, 40:70] = 2
        self._evaluate_permutation(label, "edema_only")

    def test_perm_4_scar_only(self):
        """Permutation 4: Background (0) + Scar (3) only; Normal Myo and Edema completely absent."""
        B = 4
        label = torch.zeros((B, self.H, self.W), dtype=torch.int64, device=self.device)
        label[:, 45:65, 45:65] = 3
        self._evaluate_permutation(label, "scar_only")

    def test_perm_5_edema_and_scar_only(self):
        """Permutation 5: Edema (2) + Scar (3) only (no normal myocardium 1)."""
        B = 4
        label = torch.zeros((B, self.H, self.W), dtype=torch.int64, device=self.device)
        label[:, 30:60, 30:60] = 2
        label[:, 40:50, 40:50] = 3
        self._evaluate_permutation(label, "edema_and_scar_only")

    def test_perm_6_no_scar(self):
        """Permutation 6: Normal Myo (1) + Edema (2) only (Scar 3 completely absent)."""
        B = 4
        label = torch.zeros((B, self.H, self.W), dtype=torch.int64, device=self.device)
        label[:, 25:85, 25:85] = 1
        label[:, 40:60, 40:60] = 2
        self._evaluate_permutation(label, "no_scar")

    def test_perm_7_no_edema(self):
        """Permutation 7: Normal Myo (1) + Scar (3) only (Edema 2 completely absent)."""
        B = 4
        label = torch.zeros((B, self.H, self.W), dtype=torch.int64, device=self.device)
        label[:, 25:85, 25:85] = 1
        label[:, 45:55, 45:55] = 3
        self._evaluate_permutation(label, "no_edema")

    def test_perm_8_all_classes_present(self):
        """Permutation 8: All 4 canonical classes present simultaneously (0, 1, 2, 3)."""
        B = 4
        label = torch.zeros((B, self.H, self.W), dtype=torch.int64, device=self.device)
        label[:, 20:90, 20:90] = 1
        label[:, 35:75, 35:75] = 2
        label[:, 45:55, 45:55] = 3
        self._evaluate_permutation(label, "all_classes_present")

    def test_perm_9_single_pixel_lesion_extremes(self):
        """Permutation 9: Exactly 1 single voxel of scar or edema in the entire 128x128 volume."""
        B = 2
        label = torch.zeros((B, self.H, self.W), dtype=torch.int64, device=self.device)
        label[0, 64, 64] = 3 # 1 pixel scar in slice 0
        label[1, 64, 64] = 2 # 1 pixel edema in slice 1
        self._evaluate_permutation(label, "single_pixel_extremes")

    def test_perm_10_extreme_full_canvas_saturation(self):
        """Permutation 10: 100% canvas saturation (100% myo, 100% edema, 100% scar)."""
        for class_val, name in [(1, "100_percent_myo"), (2, "100_percent_edema"), (3, "100_percent_scar")]:
            label = torch.full((2, self.H, self.W), class_val, dtype=torch.int64, device=self.device)
            self._evaluate_permutation(label, name)


class TestMilestone2BackwardGradients(unittest.TestCase):
    """Adversarial Challenge B: Gradient Backward Propagation & Numerical Stability."""

    def setUp(self):
        self.device = torch.device("cpu")
        self.criterion = MyoPSLoss()
        self.H, self.W = 64, 64 # Use 64x64 for fast thorough end-to-end model checks

    def test_backward_pass_on_full_myopsnet_model(self):
        """Verify loss.backward() through full MyoPSNet across all absent-class permutations.
        Checks that EVERY trainable parameter receives non-NaN, non-Inf gradients.
        """
        permutations = [
            ("all_bg", torch.zeros((1, self.H, self.W), dtype=torch.int64)),
            ("myo_only", (torch.rand((1, self.H, self.W)) > 0.5).long()),
            ("scar_only", ((torch.rand((1, self.H, self.W)) > 0.8) * 3).long()),
            ("edema_only", ((torch.rand((1, self.H, self.W)) > 0.8) * 2).long()),
            ("mixed_all", torch.randint(0, 4, (1, self.H, self.W), dtype=torch.int64)),
        ]

        for perm_name, label in permutations:
            # Re-seed model for consistency
            torch.manual_seed(1234)
            model = MyoPSNet(in_chs=[5, 2, 2, 3], out_chs=[3, 3, 3, 3]).to(self.device)
            model.train()

            # Create mock inputs matching DataLoader contract
            C0 = torch.rand((1, 1, self.H, self.W), device=self.device)
            LGE = torch.rand((1, 1, self.H, self.W), device=self.device)
            T2 = torch.rand((1, 1, self.H, self.W), device=self.device)
            T1m = torch.zeros((1, 1, self.H, self.W), device=self.device)
            T2starm = torch.zeros((1, 1, self.H, self.W), device=self.device)

            # Forward pass through model
            seg_C0, seg_LGE, seg_T2, seg_mapping = model(C0, LGE, T2, T1m, T2starm)

            # Target adaptation
            targets = canonical_to_myops_targets(label)

            # Loss computation
            loss_seg, loss_inv, loss_inc, loss_tot = self.criterion(
                (seg_C0, seg_LGE, seg_T2, seg_mapping), targets
            )

            self.assertTrue(torch.isfinite(loss_tot), f"Total loss is non-finite for {perm_name}")

            # Backward pass
            model.zero_grad(set_to_none=True)
            loss_tot.backward()

            # Verify gradients across all trainable parameters
            nan_params = []
            inf_params = []
            missing_grad_params = []

            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                if param.grad is None:
                    missing_grad_params.append(name)
                    continue
                if torch.isnan(param.grad).any():
                    nan_params.append(name)
                if torch.isinf(param.grad).any():
                    inf_params.append(name)

            self.assertEqual(len(nan_params), 0,
                             f"Gradients contain NaN in {nan_params} under permutation {perm_name}")
            self.assertEqual(len(inf_params), 0,
                             f"Gradients contain Inf in {inf_params} under permutation {perm_name}")
            self.assertEqual(len(missing_grad_params), 0,
                             f"Missing gradients in {missing_grad_params} under permutation {perm_name}")

    def test_backward_gradient_with_extreme_prediction_values(self):
        """Stress test gradient stability when model predictions are saturated near 0.0 or 1.0."""
        B, H, W = 2, 32, 32
        label = torch.zeros((B, H, W), dtype=torch.int64)
        label[:, 10:25, 10:25] = 1
        label[:, 15:20, 15:20] = 3
        targets = canonical_to_myops_targets(label)

        # Extremely large positive and negative logits -> softmax saturated at 0 or 1
        raw_logits = {
            "C0": (torch.randn((B, 3, H, W)) * 50.0).detach().requires_grad_(True),
            "LGE": (torch.randn((B, 3, H, W)) * 50.0).detach().requires_grad_(True),
            "T2": (torch.randn((B, 3, H, W)) * 50.0).detach().requires_grad_(True),
            "mapping": (torch.randn((B, 3, H, W)) * 50.0).detach().requires_grad_(True),
        }

        seg = {k: F.softmax(v, dim=1) for k, v in raw_logits.items()}
        loss_seg, loss_inv, loss_inc, loss_tot = self.criterion(seg, targets)

        self.assertTrue(torch.isfinite(loss_tot))
        loss_tot.backward()

        for k, v in raw_logits.items():
            self.assertIsNotNone(v.grad, f"Gradient for {k} is None")
            self.assertFalse(torch.isnan(v.grad).any(), f"Gradient for {k} contains NaN")
            self.assertFalse(torch.isinf(v.grad).any(), f"Gradient for {k} contains Inf")


class TestMilestone2ReverseMappingConflicts(unittest.TestCase):
    """Adversarial Challenge C: Reverse Mapping Priority & Conflict Stress Testing."""

    def test_exhaustive_81_truth_table_permutations(self):
        """Exhaustively verify the priority Scar (3) > Edema (2) > Normal Myo (1) > BG (0)
        across all 3^4 = 81 combinations of (pred_scar, pred_edema, pred_cardiac, pred_mapping).
        """
        for p_scar in [0, 1, 2]:
            for p_edema in [0, 1, 2]:
                for p_cardiac in [0, 1, 2]:
                    for p_mapping in [0, 1, 2]:
                        # Synthesize 1-voxel logits
                        outputs = {
                            "scar": torch.zeros((1, 3, 1, 1)),
                            "edema": torch.zeros((1, 3, 1, 1)),
                            "cardiac": torch.zeros((1, 3, 1, 1)),
                            "mapping": torch.zeros((1, 3, 1, 1)),
                        }
                        outputs["scar"][0, p_scar, 0, 0] = 10.0
                        outputs["edema"][0, p_edema, 0, 0] = 10.0
                        outputs["cardiac"][0, p_cardiac, 0, 0] = 10.0
                        outputs["mapping"][0, p_mapping, 0, 0] = 10.0

                        canonical = myops_outputs_to_canonical(outputs)
                        result = canonical[0, 0, 0].item()

                        # Determine authoritative clinical ground truth priority:
                        if p_scar == 2 or p_mapping == 2:
                            expected = 3 # Scar
                        elif p_edema == 2:
                            expected = 2 # Edema
                        elif p_scar == 1 or p_edema == 1 or p_cardiac == 1:
                            expected = 1 # Normal Myo
                        else:
                            expected = 0 # BG

                        self.assertEqual(
                            result,
                            expected,
                            f"Mismatch for state (scar={p_scar}, edema={p_edema}, cardiac={p_cardiac}, mapping={p_mapping}): "
                            f"got {result}, expected {expected}",
                        )

    def test_conflicting_huge_logits(self):
        """Simultaneous competing huge logits (+1000.0) across all branches."""
        B, H, W = 1, 10, 10
        outputs = {
            "scar": torch.zeros((B, 3, H, W)),
            "edema": torch.zeros((B, 3, H, W)),
            "cardiac": torch.zeros((B, 3, H, W)),
            "mapping": torch.zeros((B, 3, H, W)),
        }

        # Voxel (0, 0): All branches predict class 2 (scar/edema) and class 1 (myo) simultaneously
        outputs["scar"][:, 2, 0, 0] = 1000.0
        outputs["edema"][:, 2, 0, 0] = 1000.0
        outputs["cardiac"][:, 1, 0, 0] = 1000.0
        outputs["mapping"][:, 2, 0, 0] = 1000.0

        # Voxel (1, 1): Edema branch predicts class 2 (1000.0), Scar predicts class 1 (1000.0)
        outputs["scar"][:, 1, 1, 1] = 1000.0
        outputs["edema"][:, 2, 1, 1] = 1000.0
        outputs["cardiac"][:, 1, 1, 1] = 1000.0

        # Voxel (2, 2): Only cardiac predicts class 1 (1000.0), scar/edema predict BG 0
        outputs["cardiac"][:, 1, 2, 2] = 1000.0
        outputs["scar"][:, 0, 2, 2] = 1000.0
        outputs["edema"][:, 0, 2, 2] = 1000.0

        # Voxel (3, 3): Only mapping branch predicts scar (class 2 = 1000.0), while scar branch predicts BG 0
        outputs["scar"][:, 0, 3, 3] = 1000.0
        outputs["mapping"][:, 2, 3, 3] = 1000.0
        outputs["edema"][:, 2, 3, 3] = 1000.0 # Edema also 1000.0, but mapping scar must override!

        canonical = myops_outputs_to_canonical(outputs)

        self.assertEqual(canonical[0, 0, 0].item(), 3, "Scar priority failed under competing +1000 logits")
        self.assertEqual(canonical[0, 1, 1].item(), 2, "Edema priority failed under competing +1000 logits")
        self.assertEqual(canonical[0, 2, 2].item(), 1, "Normal Myo priority failed under competing +1000 logits")
        self.assertEqual(canonical[0, 3, 3].item(), 3, "Mapping branch scar override failed")

    def test_reverse_mapping_input_formats(self):
        """Verify reverse mapping handles tuples of 3, tuples of 4, dicts, and direct tensors."""
        B, H, W = 2, 32, 32
        t_cardiac = torch.randn((B, 3, H, W))
        t_scar = torch.randn((B, 3, H, W))
        t_edema = torch.randn((B, 3, H, W))
        t_mapping = torch.randn((B, 3, H, W))

        # 1. Tuple of 4: (cardiac, scar, edema, mapping)
        res_tuple4 = myops_outputs_to_canonical((t_cardiac, t_scar, t_edema, t_mapping))
        self.assertEqual(res_tuple4.shape, (B, H, W))

        # 2. Tuple of 3: (cardiac, scar, edema)
        res_tuple3 = myops_outputs_to_canonical((t_cardiac, t_scar, t_edema))
        self.assertEqual(res_tuple3.shape, (B, H, W))

        # 3. Dict with standard keys
        res_dict = myops_outputs_to_canonical({"cardiac": t_cardiac, "scar": t_scar, "edema": t_edema})
        self.assertEqual(res_dict.shape, (B, H, W))

        # 4. Dict with alias keys (C0, LGE, T2)
        res_alias = myops_outputs_to_canonical({"C0": t_cardiac, "LGE": t_scar, "T2": t_edema})
        self.assertEqual(res_alias.shape, (B, H, W))

        # 5. Direct argmax tensor
        direct_argmax = torch.randint(0, 4, (B, H, W), dtype=torch.int64)
        res_direct = myops_outputs_to_canonical(direct_argmax)
        torch.testing.assert_close(res_direct, direct_argmax)


class TestMilestone2DeviceAndImmutability(unittest.TestCase):
    """Adversarial Challenge D & E: Device Agnosticism, Shape Generalization & Immutability."""

    def test_device_agnosticism_and_no_cuda_hardcoding(self):
        """Verify dataset/target_adapter.py and network/criterion/loss.py contain NO hardcoded .cuda() calls."""
        import dataset.target_adapter as ta
        import network.criterion.loss as loss_mod

        ta_source = inspect.getsource(ta)
        loss_source = inspect.getsource(loss_mod)

        # Check for hardcoded .cuda()
        self.assertNotIn(".cuda()", ta_source, "target_adapter.py contains hardcoded .cuda() call!")
        self.assertNotIn(".cuda()", loss_source, "network/criterion/loss.py contains hardcoded .cuda() call!")

        # Check device preservation on CPU
        cpu_label = torch.zeros((1, 32, 32), dtype=torch.int64, device=torch.device("cpu"))
        cpu_targets = canonical_to_myops_targets(cpu_label)
        for k, v in cpu_targets.items():
            self.assertEqual(v.device.type, "cpu")

        cpu_recon = myops_outputs_to_canonical(cpu_targets)
        self.assertEqual(cpu_recon.device.type, "cpu")

        # If CUDA is available, test on CUDA as well
        if torch.cuda.is_available():
            cuda_device = torch.device("cuda:0")
            cuda_label = cpu_label.to(cuda_device)
            cuda_targets = canonical_to_myops_targets(cuda_label)
            for k, v in cuda_targets.items():
                self.assertEqual(v.device.type, "cuda")
            cuda_recon = myops_outputs_to_canonical(cuda_targets)
            self.assertEqual(cuda_recon.device.type, "cuda")

    def test_input_immutability(self):
        """Verify canonical_to_myops_targets and myops_outputs_to_canonical do NOT mutate input tensors."""
        label = torch.zeros((2, 64, 64), dtype=torch.int64)
        label[:, 20:40, 20:40] = 1
        label[:, 25:35, 25:35] = 2
        label[:, 28:32, 28:32] = 3
        label_clone = label.clone()

        targets = canonical_to_myops_targets(label)
        torch.testing.assert_close(label, label_clone, msg="canonical_to_myops_targets mutated the input label!")

        targets_clone = {k: v.clone() for k, v in targets.items()}
        canonical = myops_outputs_to_canonical(targets)

        for k in targets:
            torch.testing.assert_close(targets[k], targets_clone[k],
                                       msg=f"myops_outputs_to_canonical mutated target tensor '{k}'!")

    def test_arbitrary_spatial_dimensions_and_batch_sizes(self):
        """Verify target adapter and loss operate seamlessly on non-square shapes and diverse batch sizes."""
        criterion = MyoPSLoss()
        shapes = [
            (1, 32, 32),
            (3, 64, 128),  # non-square
            (5, 128, 64),  # non-square
            (16, 64, 64),  # full effective batch size
        ]

        for B, H, W in shapes:
            label = torch.randint(0, 4, (B, H, W), dtype=torch.int64)
            targets = canonical_to_myops_targets(label)

            outputs = {
                "C0": F.softmax(torch.randn((B, 3, H, W)), dim=1),
                "LGE": F.softmax(torch.randn((B, 3, H, W)), dim=1),
                "T2": F.softmax(torch.randn((B, 3, H, W)), dim=1),
                "mapping": F.softmax(torch.randn((B, 3, H, W)), dim=1),
            }

            loss_seg, loss_inv, loss_inc, loss_tot = criterion(outputs, targets)
            self.assertTrue(torch.isfinite(loss_tot), f"Failed for shape {(B, H, W)}")

            recon = myops_outputs_to_canonical(outputs)
            self.assertEqual(recon.shape, (B, H, W), f"Shape mismatch in recon for {(B, H, W)}")


if __name__ == "__main__":
    unittest.main()
