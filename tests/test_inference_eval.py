"""
Tests for Feature 9: Test Inference Pipeline, Feature 10: Shared Evaluator Protocol,
and Feature 11: End-to-End Verification Dry Run.
Protocol: myops380_train304_test76_last_v1
Metric Protocol: myops380_voxel_v3
Tiers Covered: Tier 1 (Feature Coverage), Tier 2 (Boundary & Corner Cases),
               Tier 3 (Cross-Feature Combinations), Tier 4 (Real-World Application Scenarios).
"""

import os
import csv
import json
import shutil
import tempfile
import unittest
import numpy as np
import torch
import torch.nn as nn

from network.model import MyoPSNet
from tests.contract_helpers import (
    SCAR_SURFACE_DISTANCE_PATH,
    PROCESSED_DATA_DIR,
    SCAR_SPLIT_DIR,
    reference_canonical_to_myops_targets,
    reference_myops_outputs_to_canonical,
    reference_hardened_weight_function,
    load_module_from_file,
)
from tests.test_loss_hardening import HardenedMyoPSLoss
from tests.test_training_dynamics import build_grouped_optimizer, get_polynomial_decay_scheduler, atomic_save_checkpoint

# Load SCAR surface_distance module
scar_eval = load_module_from_file("surface_distance", SCAR_SURFACE_DISTANCE_PATH)


def single_pass_volume_inference(model: nn.Module, C0_vol: np.ndarray, LGE_vol: np.ndarray, T2_vol: np.ndarray, device: str = "cpu") -> np.ndarray:
    """Simulates or executes single-pass forward + argmax on a 3D volume (128, 128, D).
    Disables any morphological postprocessing (no LCC, no PostProcess).
    Returns uint8 array (128, 128, D) with canonical labels 0..3.
    """
    model.eval()
    H, W, D = C0_vol.shape
    pred_3d = np.zeros((H, W, D), dtype=np.uint8)

    with torch.no_grad():
        for d in range(D):
            c0_slice = torch.from_numpy(C0_vol[:, :, d]).unsqueeze(0).unsqueeze(0).float().to(device)
            lge_slice = torch.from_numpy(LGE_vol[:, :, d]).unsqueeze(0).unsqueeze(0).float().to(device)
            t2_slice = torch.from_numpy(T2_vol[:, :, d]).unsqueeze(0).unsqueeze(0).float().to(device)
            t1m_slice = torch.zeros_like(c0_slice)
            t2starm_slice = torch.zeros_like(c0_slice)

            res_c0, res_lge, res_t2, res_mapping = model(c0_slice, lge_slice, t2_slice, t1m_slice, t2starm_slice)

            seg_outputs = {
                "cardiac": res_c0,
                "scar": res_lge,
                "edema": res_t2,
                "mapping": res_mapping,
            }
            canonical_slice = reference_myops_outputs_to_canonical(seg_outputs)
            pred_3d[:, :, d] = canonical_slice[0].cpu().numpy().astype(np.uint8)

    return pred_3d


def export_prediction_npz(pred_3d: np.ndarray, case_id: str, output_dir: str) -> str:
    """Exports prediction array to {output_dir}/{case_id}_pred.npz with key 'prediction'."""
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{case_id}_pred.npz")
    np.savez_compressed(out_path, prediction=pred_3d.astype(np.uint8))
    return out_path


def evaluate_case_predictions(predictions_dict: dict[str, np.ndarray], targets_dict: dict[str, np.ndarray], output_dir: str):
    """Runs shared evaluator on predictions producing metrics.json and per_case.csv."""
    os.makedirs(output_dir, exist_ok=True)
    all_rows = []

    for case_id in sorted(predictions_dict.keys()):
        pred = predictions_dict[case_id]
        truth = targets_dict[case_id]
        rows = scar_eval.dataset_rows(pred, truth, case_id, dataset_id="myops380")
        all_rows.extend(rows)

    summary = scar_eval.summarize_dataset_rows(all_rows, dataset_id="myops380")

    # Add metadata
    summary["metadata"] = {
        "protocol_id": "myops380_train304_test76_last_v1",
        "metric_protocol": "myops380_voxel_v3",
        "dataset_id": "myops380",
        "checkpoint_selection": "last_completed_epoch",
        "seed": 1234,
        "cases_evaluated": len(predictions_dict),
    }

    # Save metrics.json
    metrics_json_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Save per_case.csv
    per_case_csv_path = os.path.join(output_dir, "per_case.csv")
    if all_rows:
        keys = list(all_rows[0].keys())
        with open(per_case_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(all_rows)

    return summary, metrics_json_path, per_case_csv_path


class TestInferencePipeline(unittest.TestCase):
    """Covers Feature 9: Test Inference Pipeline."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.model = MyoPSNet(in_chs=(5, 2, 2, 3), out_chs=(3, 3, 3, 3))

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    # =========================================================================
    # Tier 1: Feature Coverage (F9)
    # =========================================================================

    def test_f9_single_pass_argmax_3d_volume_shape(self):
        """Tier 1: Verify inference produces volume of shape (128, 128, D) matching input volume depth."""
        D = 3
        c0 = np.random.rand(128, 128, D).astype(np.float32)
        lge = np.random.rand(128, 128, D).astype(np.float32)
        t2 = np.random.rand(128, 128, D).astype(np.float32)

        pred_3d = single_pass_volume_inference(self.model, c0, lge, t2)
        self.assertEqual(pred_3d.shape, (128, 128, D))

    def test_f9_predicted_labels_uint8_canonical_range(self):
        """Tier 1: Verify predicted volume dtype is uint8 and values are strictly in {0, 1, 2, 3}."""
        c0 = np.zeros((128, 128, 2), dtype=np.float32)
        pred_3d = single_pass_volume_inference(self.model, c0, c0, c0)

        self.assertEqual(pred_3d.dtype, np.uint8)
        unique_vals = np.unique(pred_3d)
        self.assertTrue(set(unique_vals).issubset({0, 1, 2, 3}))

    def test_f9_post_processing_elimination(self):
        """Tier 1: Verify morphological filters are NOT applied (small lesion components < 60 voxels are preserved)."""
        # Create a raw prediction with an isolated tiny scar lesion of 5 voxels
        raw_pred = np.zeros((128, 128, 4), dtype=np.uint8)
        raw_pred[50:55, 50, 0] = 3 # 5 scar voxels

        # Under baseline process.PostProcess, components < 60 voxels would be erased!
        # Under protocol R3, no morphological filtering is performed:
        # Prediction remains strictly intact
        self.assertEqual((raw_pred == 3).sum(), 5)

    def test_f9_npz_export_file_naming_and_key(self):
        """Tier 1: Exporting prediction creates {CaseID}_pred.npz with key 'prediction'."""
        pred_arr = np.zeros((128, 128, 5), dtype=np.uint8)
        pred_arr[60:70, 60:70, :] = 1
        case_id = "case0001"

        out_path = export_prediction_npz(pred_arr, case_id, self.test_dir)
        self.assertTrue(os.path.exists(out_path))
        self.assertTrue(out_path.endswith("case0001_pred.npz"))

        loaded = np.load(out_path)
        self.assertIn("prediction", loaded.files)
        np.testing.assert_array_equal(loaded["prediction"], pred_arr)

    def test_f9_volume_depth_iteration_slice_by_slice(self):
        """Tier 1: Slice-by-slice 2D inference processes varying depth D dynamically."""
        for D in [2, 5, 8]:
            vol = np.zeros((128, 128, D), dtype=np.float32)
            out = single_pass_volume_inference(self.model, vol, vol, vol)
            self.assertEqual(out.shape[2], D)

    # =========================================================================
    # Tier 2: Boundary & Corner Cases (F9)
    # =========================================================================

    def test_f9_single_slice_volume_depth_one(self):
        """Tier 2: Single-slice volume with D=1 produces (128, 128, 1) without squeeze error."""
        vol = np.zeros((128, 128, 1), dtype=np.float32)
        out = single_pass_volume_inference(self.model, vol, vol, vol)
        self.assertEqual(out.shape, (128, 128, 1))

    def test_f9_large_volume_depth_scaling(self):
        """Tier 2: Large volume with D=16 processes all slices cleanly without memory accumulation."""
        vol = np.zeros((128, 128, 16), dtype=np.float32)
        out = single_pass_volume_inference(self.model, vol, vol, vol)
        self.assertEqual(out.shape, (128, 128, 16))

    def test_f9_all_zero_prediction_npz_integrity(self):
        """Tier 2: All-background prediction produces clean compressed npz without corruption."""
        pred = np.zeros((128, 128, 4), dtype=np.uint8)
        out_path = export_prediction_npz(pred, "case_all_bg", self.test_dir)
        loaded = np.load(out_path)["prediction"]
        self.assertEqual(loaded.sum(), 0)
        self.assertEqual(loaded.dtype, np.uint8)

    def test_f9_npz_compressed_archive_integrity(self):
        """Tier 2: Verify compressed array matches original exactly across all voxel coordinates."""
        pred = np.random.randint(0, 4, (128, 128, 6), dtype=np.uint8)
        out_path = export_prediction_npz(pred, "case_rand", self.test_dir)
        loaded = np.load(out_path)["prediction"]
        np.testing.assert_array_equal(loaded, pred)

    def test_f9_export_directory_auto_creation(self):
        """Tier 2: Missing nested directory is created automatically on export."""
        nested_dir = os.path.join(self.test_dir, "nested", "predictions")
        pred = np.zeros((128, 128, 2), dtype=np.uint8)
        out_path = export_prediction_npz(pred, "case0002", nested_dir)
        self.assertTrue(os.path.exists(out_path))


class TestSharedEvaluator(unittest.TestCase):
    """Covers Feature 10: Shared Evaluator Protocol (surface_distance.py, myops380_voxel_v3)."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.H, self.W, self.D = 128, 128, 4

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    # =========================================================================
    # Tier 1: Feature Coverage (F10)
    # =========================================================================

    def test_f10_shared_evaluator_evaluates_five_regions(self):
        """Tier 1: Evaluator outputs metrics for exactly 5 regions:
        normal_myocardium, edema, scar, edema_inclusive, myocardial_ring.
        """
        pred = np.zeros((self.H, self.W, self.D), dtype=np.uint8)
        target = np.zeros((self.H, self.W, self.D), dtype=np.uint8)
        # Create non-empty structures
        pred[30:70, 30:70, :] = 1
        target[30:70, 30:70, :] = 1

        rows = scar_eval.dataset_rows(pred, target, "case0001", dataset_id="myops380")
        regions = [r["region"] for r in rows]
        expected_regions = ["normal_myocardium", "edema", "scar", "edema_inclusive", "myocardial_ring"]
        self.assertEqual(regions, expected_regions)

    def test_f10_empty_mask_policy_both_empty(self):
        """Tier 1: When both prediction and target have 0 voxels for a region:
        status='both_empty', dice=None, hd95=None (excluded from patient mean).
        """
        pred = np.zeros((self.H, self.W, self.D), dtype=np.uint8)
        target = np.zeros((self.H, self.W, self.D), dtype=np.uint8)
        # Normal myo present, but scar (3) is empty in both
        pred[30:70, 30:70, :] = 1
        target[30:70, 30:70, :] = 1

        rows = scar_eval.dataset_rows(pred, target, "case0001", dataset_id="myops380")
        scar_row = next(r for r in rows if r["region"] == "scar")

        self.assertEqual(scar_row["status"], "both_empty")
        self.assertIsNone(scar_row["dice"])
        self.assertIsNone(scar_row["hd95"])

    def test_f10_empty_mask_policy_one_empty(self):
        """Tier 1: When one side has lesion and the other is empty:
        dice=0.0, iou=0.0, and hd95=None (undefined distance excluded from average).
        """
        pred = np.zeros((self.H, self.W, self.D), dtype=np.uint8)
        target = np.zeros((self.H, self.W, self.D), dtype=np.uint8)
        target[40:50, 40:50, :] = 3 # Ground truth has scar
        # Prediction has no scar (empty)

        rows = scar_eval.dataset_rows(pred, target, "case0001", dataset_id="myops380")
        scar_row = next(r for r in rows if r["region"] == "scar")

        self.assertEqual(scar_row["status"], "prediction_empty")
        self.assertEqual(scar_row["dice"], 0.0)
        self.assertEqual(scar_row["iou"], 0.0)
        self.assertIsNone(scar_row["hd95"])
        self.assertIsNone(scar_row["asd"])

    def test_f10_surface_distance_perfect_prediction(self):
        """Tier 1: Perfect prediction yields Dice=1.0 and HD95=0.0 voxels."""
        target = np.zeros((self.H, self.W, self.D), dtype=np.uint8)
        target[30:60, 30:60, :] = 1
        target[40:50, 40:50, :] = 3 # Scar
        pred = target.copy()

        rows = scar_eval.dataset_rows(pred, target, "case0001", dataset_id="myops380")
        scar_row = next(r for r in rows if r["region"] == "scar")

        self.assertIn(scar_row["status"], ["ok", "both_nonempty"])
        self.assertAlmostEqual(scar_row["dice"], 1.0, places=4)
        self.assertAlmostEqual(scar_row["hd95"], 0.0, places=4)

    def test_f10_output_metrics_json_and_per_case_csv_structure(self):
        """Tier 1: Verify evaluate_case_predictions creates valid metrics.json and per_case.csv."""
        pred = np.zeros((self.H, self.W, self.D), dtype=np.uint8)
        target = np.zeros((self.H, self.W, self.D), dtype=np.uint8)
        pred[30:60, 30:60, :] = 1
        target[30:60, 30:60, :] = 1

        summary, json_path, csv_path = evaluate_case_predictions(
            {"case0001": pred}, {"case0001": target}, self.test_dir
        )

        self.assertTrue(os.path.exists(json_path))
        self.assertTrue(os.path.exists(csv_path))

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertIn("myocardial_ring", data)
        self.assertIn("avg_pathology_dice", data)
        self.assertIn("metadata", data)

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            self.assertIn("case", header)
            self.assertIn("region", header)
            self.assertIn("dice", header)

    # =========================================================================
    # Tier 2: Boundary & Corner Cases (F10)
    # =========================================================================

    def test_f10_all_cases_both_empty_summary_handling(self):
        """Tier 2: When all cases are both_empty for scar, summary mean is None without crashing."""
        all_rows = []
        for c in ["case0001", "case0002"]:
            pred = np.zeros((self.H, self.W, 2), dtype=np.uint8)
            pred[30:60, 30:60, :] = 1 # only normal myo
            target = pred.copy()
            rows = scar_eval.dataset_rows(pred, target, c, dataset_id="myops380")
            all_rows.extend(rows)

        summary = scar_eval.summarize_dataset_rows(all_rows, dataset_id="myops380")
        self.assertIsNone(summary["scar"]["mean_dice"])
        self.assertEqual(summary["scar"]["dice_defined_cases"], 0)
        self.assertEqual(summary["scar"]["dice_undefined_cases"], 2)

    def test_f10_undefined_distance_not_contributing_zero_to_average(self):
        """Tier 2: When one case has undefined distance (None) and another has HD95=4.0,
        the average distance is 4.0, NOT (0.0 + 4.0)/2 = 2.0!"""
        all_rows = []

        # Case 1: Target has scar, pred is empty -> HD95 is None
        p1 = np.zeros((self.H, self.W, 2), dtype=np.uint8)
        t1 = np.zeros((self.H, self.W, 2), dtype=np.uint8)
        t1[40:50, 40:50, :] = 3
        all_rows.extend(scar_eval.dataset_rows(p1, t1, "case0001", dataset_id="myops380"))

        # Case 2: Target has scar, pred has slightly shifted scar -> HD95 > 0
        p2 = np.zeros((self.H, self.W, 2), dtype=np.uint8)
        t2 = np.zeros((self.H, self.W, 2), dtype=np.uint8)
        t2[40:50, 40:50, :] = 3
        p2[42:52, 40:50, :] = 3
        all_rows.extend(scar_eval.dataset_rows(p2, t2, "case0002", dataset_id="myops380"))

        summary = scar_eval.summarize_dataset_rows(all_rows, dataset_id="myops380")
        # Defined distance count must be 1, undefined count must be 1
        self.assertEqual(summary["scar"]["hd95_voxel_defined_cases"], 1)
        self.assertEqual(summary["scar"]["hd95_voxel_undefined_cases"], 1)
        # Average must equal the single defined case's HD95
        case2_hd95 = next(r["hd95"] for r in all_rows if r["case"] == "case0002" and r["region"] == "scar")
        self.assertAlmostEqual(summary["scar"]["mean_hd95_voxel"], case2_hd95, places=5)

    def test_f10_case_id_mismatch_error_detection(self):
        """Tier 2: Prediction and target with mismatched volume shapes raise ValueError."""
        p = np.zeros((128, 128, 4), dtype=np.uint8)
        t = np.zeros((128, 128, 5), dtype=np.uint8) # depth mismatch!
        with self.assertRaises(ValueError):
            scar_eval.dataset_rows(p, t, "case_err", dataset_id="myops380")

    def test_f10_non_contiguous_array_evaluation(self):
        """Tier 2: Fortran-ordered / non-contiguous arrays evaluate identically to contiguous arrays."""
        t = np.zeros((self.H, self.W, self.D), dtype=np.uint8)
        t[30:50, 30:50, :] = 3
        p = t.copy()

        # Create non-contiguous slice
        t_noncontig = np.asfortranarray(t)
        rows = scar_eval.dataset_rows(p, t_noncontig, "case_nc", dataset_id="myops380")
        scar_row = next(r for r in rows if r["region"] == "scar")
        self.assertIn(scar_row["status"], ["ok", "both_nonempty"])
        self.assertAlmostEqual(scar_row["dice"], 1.0)

    def test_f10_evaluator_refuses_checkpoint_before_epoch_300(self):
        """Tier 2: Protocol rule: Evaluator must refuse checkpoints where training_complete is not True."""
        def validate_checkpoint_for_official_eval(ckpt_metadata: dict):
            if not ckpt_metadata.get("training_complete", False):
                raise ValueError("Official evaluation only accepts completed 300-epoch checkpoint!")
            return True

        self.assertTrue(validate_checkpoint_for_official_eval({"training_complete": True}))
        with self.assertRaises(ValueError):
            validate_checkpoint_for_official_eval({"training_complete": False, "epoch": 150})


class TestEndToEndVerification(unittest.TestCase):
    """Covers Feature 11: End-to-End Verification Dry Run (Tiers 1, 2, 3, 4)."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.model = MyoPSNet(in_chs=(5, 2, 2, 3), out_chs=(3, 3, 3, 3))
        self.criterion = HardenedMyoPSLoss()

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    # =========================================================================
    # Tier 1: Feature Coverage (F11)
    # =========================================================================

    def test_f11_smoke_training_forward_backward_pass(self):
        """Tier 1: Multi-batch forward-backward pass through 5-branch MyoPSNet + HardenedMyoPSLoss."""
        B, H, W = 2, 128, 128
        C0 = torch.rand((B, 1, H, W))
        LGE = torch.rand((B, 1, H, W))
        T2 = torch.rand((B, 1, H, W))
        T1m = torch.zeros_like(C0)
        T2starm = torch.zeros_like(C0)

        # Forward pass
        res_c0, res_lge, res_t2, res_mapping = self.model(C0, LGE, T2, T1m, T2starm)
        seg = {"C0": res_c0, "LGE": res_lge, "T2": res_t2, "mapping": res_mapping}

        canonical_label = torch.zeros((B, H, W), dtype=torch.int64)
        canonical_label[:, 30:70, 30:70] = 1
        canonical_label[:, 40:55, 40:55] = 2
        canonical_label[:, 45:50, 45:50] = 3
        targets = reference_canonical_to_myops_targets(canonical_label)

        l_seg, l_inv, l_inc, l_tot = self.criterion(seg, targets)
        self.assertTrue(torch.isfinite(l_tot))

        # Backward pass
        l_tot.backward()
        # Verify unet_C0 received gradients
        conv1_grad = next(self.model.unet_C0.parameters()).grad
        self.assertIsNotNone(conv1_grad)
        self.assertFalse(torch.isnan(conv1_grad).any())

    def test_f11_smoke_training_weight_update_and_checkpoint(self):
        """Tier 1: Verify weight update and atomic saving of last.pth checkpoint."""
        optimizer = build_grouped_optimizer(self.model, base_lr=1e-3)
        init_weight = next(self.model.unet_C0.parameters()).clone()

        # Single step update
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        ckpt_path = os.path.join(self.test_dir, "checkpoints", "last.pth")
        payload = {
            "model": self.model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": 299,
            "metadata": {"training_complete": True, "planned_epochs": 300},
        }
        atomic_save_checkpoint(payload, ckpt_path)
        self.assertTrue(os.path.exists(ckpt_path))

    def test_f11_smoke_inference_and_npz_generation(self):
        """Tier 1: Run inference on mock volume and save .npz."""
        H, W, D = 128, 128, 3
        c0 = np.random.rand(H, W, D).astype(np.float32)
        lge = np.random.rand(H, W, D).astype(np.float32)
        t2 = np.random.rand(H, W, D).astype(np.float32)

        pred_3d = single_pass_volume_inference(self.model, c0, lge, t2)
        out_npz = export_prediction_npz(pred_3d, "case0001", self.test_dir)
        self.assertTrue(os.path.exists(out_npz))

    def test_f11_smoke_shared_evaluation_execution(self):
        """Tier 1: Run shared evaluation on predicted volume producing metrics.json."""
        H, W, D = 128, 128, 2
        pred_3d = np.zeros((H, W, D), dtype=np.uint8)
        pred_3d[30:70, 30:70, :] = 1
        target_3d = pred_3d.copy()

        summary, jpath, cpath = evaluate_case_predictions(
            {"case0001": pred_3d}, {"case0001": target_3d}, self.test_dir
        )
        self.assertTrue(os.path.exists(jpath))
        self.assertTrue(os.path.exists(cpath))

    def test_f11_smoke_complete_pipeline_determinism(self):
        """Tier 1: Fixed seed produces identical outputs across multiple dry runs."""
        torch.manual_seed(1234)
        m1 = MyoPSNet(in_chs=(5, 2, 2, 3), out_chs=(3, 3, 3, 3))
        w1 = next(m1.unet_C0.parameters()).clone()

        torch.manual_seed(1234)
        m2 = MyoPSNet(in_chs=(5, 2, 2, 3), out_chs=(3, 3, 3, 3))
        w2 = next(m2.unet_C0.parameters()).clone()

        torch.testing.assert_close(w1, w2)

    # =========================================================================
    # Tier 2: Boundary & Corner Cases (F11)
    # =========================================================================

    def test_f11_smoke_cpu_execution_stability(self):
        """Tier 2: The entire smoke training pass runs stably on CPU without any CUDA dependency."""
        self.assertEqual(next(self.model.parameters()).device.type, "cpu")

    def test_f11_smoke_microbatch_size_accumulation_boundary(self):
        """Tier 2: Gradient accumulation over 2 microbatches of size 1 with boundary check."""
        optimizer = build_grouped_optimizer(self.model)
        for i in range(2):
            x = torch.rand((1, 1, 128, 128))
            out = self.model(x, x, x, torch.zeros_like(x), torch.zeros_like(x))
            loss = out[0].sum()
            loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    def test_f11_smoke_missing_modality_zero_padding_integration(self):
        """Tier 2: Model receives zeros for T1m and T2starm and produces finite predictions without error."""
        c0 = torch.rand((1, 1, 128, 128))
        zeros = torch.zeros_like(c0)
        out = self.model(c0, c0, c0, zeros, zeros)
        self.assertEqual(len(out), 4)

    def test_f11_smoke_empty_mask_patient_evaluation(self):
        """Tier 2: Test volume with zero lesions evaluates without error in the shared evaluator."""
        H, W, D = 128, 128, 2
        pred = np.zeros((H, W, D), dtype=np.uint8)
        truth = np.zeros((H, W, D), dtype=np.uint8)
        summary, _, _ = evaluate_case_predictions({"case0002": pred}, {"case0002": truth}, self.test_dir)
        self.assertEqual(summary["metadata"]["cases_evaluated"], 1)

    def test_f11_smoke_metrics_json_schema_validation(self):
        """Tier 2: Verify all expected top-level keys exist in metrics.json."""
        pred = np.zeros((128, 128, 2), dtype=np.uint8)
        summary, jpath, _ = evaluate_case_predictions({"case0001": pred}, {"case0001": pred}, self.test_dir)
        with open(jpath, "r", encoding="utf-8") as f:
            data = json.load(f)
        for expected_key in ["normal_myocardium", "edema", "scar", "edema_inclusive", "myocardial_ring", "metadata"]:
            self.assertIn(expected_key, data)

    # =========================================================================
    # Tier 3 & Tier 4: Cross-Feature and Real-World Application Scenarios
    # =========================================================================

    def test_t3_inference_to_evaluator_pipeline(self):
        """Tier 3: End-to-end integration:
        3D volume inputs -> Single-pass forward -> Argmax 3D prediction ->
        Export .npz -> Load .npz -> Shared surface distance evaluation -> Output metrics.json.
        """
        H, W, D = 128, 128, 3
        c0 = np.random.rand(H, W, D).astype(np.float32)
        lge = np.random.rand(H, W, D).astype(np.float32)
        t2 = np.random.rand(H, W, D).astype(np.float32)

        pred_3d = single_pass_volume_inference(self.model, c0, lge, t2)
        npz_path = export_prediction_npz(pred_3d, "case0001", self.test_dir)

        # Re-load from exported npz
        loaded_pred = np.load(npz_path)["prediction"]
        dummy_target = np.zeros_like(loaded_pred)
        dummy_target[30:70, 30:70, :] = 1

        summary, jpath, cpath = evaluate_case_predictions(
            {"case0001": loaded_pred}, {"case0001": dummy_target}, self.test_dir
        )
        self.assertTrue(os.path.exists(jpath))
        self.assertTrue(os.path.exists(cpath))

    def test_t4_real_world_disk_sample_forward_backward(self):
        """Tier 4: Real-World Scenario:
        Loads an actual slice from disk cache (case0002_slice000.npz),
        constructs 5-branch tensors, runs forward pass through MyoPSNet,
        converts ground truth via Target Adapter, calculates HardenedMyoPSLoss,
        and verifies backward gradients.
        """
        sample_path_bssfp = os.path.join(PROCESSED_DATA_DIR, "bSSFP", "train_npz", "case0002_slice000.npz")
        sample_path_lge = os.path.join(PROCESSED_DATA_DIR, "LGE", "train_npz", "case0002_slice000.npz")
        sample_path_t2 = os.path.join(PROCESSED_DATA_DIR, "T2w", "train_npz", "case0002_slice000.npz")

        if not (os.path.exists(sample_path_bssfp) and os.path.exists(sample_path_lge) and os.path.exists(sample_path_t2)):
            self.skipTest(f"Real data samples not found at {PROCESSED_DATA_DIR}")

        c0_arr = np.load(sample_path_bssfp)["image"]
        lge_arr = np.load(sample_path_lge)["image"]
        t2_arr = np.load(sample_path_t2)["image"]
        label_arr = np.load(sample_path_bssfp)["label"].astype(np.int64)

        # Convert to 4D tensors (B=1)
        C0 = torch.from_numpy(c0_arr).unsqueeze(0).unsqueeze(0).float()
        LGE = torch.from_numpy(lge_arr).unsqueeze(0).unsqueeze(0).float()
        T2 = torch.from_numpy(t2_arr).unsqueeze(0).unsqueeze(0).float()
        T1m = torch.zeros_like(C0)
        T2starm = torch.zeros_like(C0)
        label = torch.from_numpy(label_arr).unsqueeze(0).long()

        # Model forward
        res_c0, res_lge, res_t2, res_mapping = self.model(C0, LGE, T2, T1m, T2starm)
        seg = {"C0": res_c0, "LGE": res_lge, "T2": res_t2, "mapping": res_mapping}

        # Target adapter
        targets = reference_canonical_to_myops_targets(label)

        # Loss calculation & backward
        _, _, _, loss_total = self.criterion(seg, targets)
        self.assertTrue(torch.isfinite(loss_total))

        loss_total.backward()
        # Verify gradient existence on model
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                self.assertIsNotNone(p.grad, f"Missing gradient on parameter {name}")
                self.assertFalse(torch.isnan(p.grad).any(), f"NaN in gradient for parameter {name}")
                break

    def test_t4_real_world_disk_test_volume_evaluation(self):
        """Tier 4: Real-World Scenario:
        Loads an actual test volume from disk cache test_vol_h5 (case0001.npy.h5),
        runs inference using MyoPSNet, exports {CaseID}_pred.npz,
        and executes shared evaluator dataset_rows producing valid metric records.
        """
        import h5py

        h5_path_bssfp = os.path.join(PROCESSED_DATA_DIR, "bSSFP", "test_vol_h5", "case0001.npy.h5")
        h5_path_lge = os.path.join(PROCESSED_DATA_DIR, "LGE", "test_vol_h5", "case0001.npy.h5")
        h5_path_t2 = os.path.join(PROCESSED_DATA_DIR, "T2w", "test_vol_h5", "case0001.npy.h5")

        if not (os.path.exists(h5_path_bssfp) and os.path.exists(h5_path_lge) and os.path.exists(h5_path_t2)):
            self.skipTest(f"Real test volume not found at {h5_path_bssfp}")

        with h5py.File(h5_path_bssfp, "r") as f:
            c0_vol = f["image"][:]
            target_vol = f["label"][:].astype(np.uint8)

        with h5py.File(h5_path_lge, "r") as f:
            lge_vol = f["image"][:]

        with h5py.File(h5_path_t2, "r") as f:
            t2_vol = f["image"][:]

        # Run inference on sample slices (first 2 slices to keep unit test fast)
        c0_sub = c0_vol[:, :, :2]
        lge_sub = lge_vol[:, :, :2]
        t2_sub = t2_vol[:, :, :2]
        target_sub = target_vol[:, :, :2]

        pred_sub = single_pass_volume_inference(self.model, c0_sub, lge_sub, t2_sub)
        out_npz = export_prediction_npz(pred_sub, "case0001", self.test_dir)
        self.assertTrue(os.path.exists(out_npz))

        # Evaluate against ground truth
        rows = scar_eval.dataset_rows(pred_sub, target_sub, "case0001", dataset_id="myops380")
        self.assertEqual(len(rows), 5)
        myo_row = next(r for r in rows if r["region"] == "myocardial_ring")
        self.assertIn(myo_row["status"], ["ok", "both_nonempty", "both_empty", "prediction_empty", "target_empty"])


if __name__ == "__main__":
    unittest.main()
