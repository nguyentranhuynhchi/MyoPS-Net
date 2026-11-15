"""
tests/test_multi_dataset.py
===========================
Comprehensive Unit Test Suite for Multi-Dataset Support (MyoPS380 & MyoPS++),
Double-Swap Prevention, Efficiency Metrics, and Core Invariants in MyoPS-Net.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch
import torch.nn as nn

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from criterion.loss import MyoPSLoss
from dataset.augmentation import RandomGenerator
from dataset.myops380_dataset import (
    EXPECTED_TEST_VOLUMES,
    EXPECTED_TRAIN_PATIENTS,
    EXPECTED_TRAIN_SLICES,
    MYOPS380_EXPECTED_TEST_VOLUMES,
    MYOPS380_EXPECTED_TRAIN_PATIENTS,
    MYOPS380_EXPECTED_TRAIN_SLICES,
    MYOPSPP_EXPECTED_TEST_VOLUMES,
    MYOPSPP_EXPECTED_TRAIN_PATIENTS,
    MYOPSPP_EXPECTED_TRAIN_SLICES,
    MyoPS380TestDataset,
    MyoPS380TrainDataset,
    MyoPSTestDataset,
    MyoPSTrainDataset,
    normalize_dataset_name,
    patient_id,
    read_manifest_lines,
    resolve_split_file,
)
from evaluate_predictions import (
    BENCHMARK_PROTOCOLS,
    CANONICAL_LUT,
    binary_metrics,
    compute_cohort_summary,
    evaluate_single_volume,
)
from network.model import MyoPSNetL
from training.checkpoint import (
    capture_rng_state,
    load_checkpoint,
    save_checkpoint,
    validate_checkpoint_payload,
)
from training.optimizer import (
    OPTIMIZATION_PROTOCOL,
    calculate_total_updates,
)
from utils.efficiency import (
    build_efficiency_payload,
    check_rule_compliance,
    count_parameters,
    measure_flops,
    measure_inference_efficiency,
    summarize_training_efficiency,
)
from utils.tools import (
    create_multi_head_targets,
    fuse_predictions,
)


class TestMultiDatasetSplits(unittest.TestCase):
    """Test data split manifests and line counts for both MyoPS380 and MyoPS++."""

    def test_normalize_dataset_name(self):
        self.assertEqual(normalize_dataset_name("myops380"), "myops380")
        self.assertEqual(normalize_dataset_name("myopspp"), "myopspp")
        self.assertEqual(normalize_dataset_name("myopspp_unet2d_roi128_1mm"), "myopspp")
        self.assertEqual(normalize_dataset_name("MYOPSPP"), "myopspp")
        self.assertEqual(normalize_dataset_name("MYOPS380"), "myops380")
        with self.assertRaises(ValueError):
            normalize_dataset_name("unknown_dataset")

    def test_myops380_split_manifests(self):
        train_file = PROJECT_ROOT / "preprocessing" / "splits" / "myops380_no_val" / "train.txt"
        test_file = PROJECT_ROOT / "preprocessing" / "splits" / "myops380_no_val" / "test_vol.txt"

        self.assertTrue(train_file.is_file(), f"Missing {train_file}")
        self.assertTrue(test_file.is_file(), f"Missing {test_file}")

        train_lines = read_manifest_lines(train_file)
        test_lines = read_manifest_lines(test_file)

        self.assertEqual(len(train_lines), EXPECTED_TRAIN_SLICES, f"Expected {EXPECTED_TRAIN_SLICES} slices")
        train_patients = {patient_id(line) for line in train_lines}
        self.assertEqual(len(train_patients), EXPECTED_TRAIN_PATIENTS, f"Expected {EXPECTED_TRAIN_PATIENTS} patients")

        self.assertEqual(len(test_lines), EXPECTED_TEST_VOLUMES, f"Expected {EXPECTED_TEST_VOLUMES} test volumes")
        test_patients = {patient_id(line) for line in test_lines}
        self.assertEqual(len(test_patients), EXPECTED_TEST_VOLUMES)

        # Zero patient leakage
        self.assertEqual(len(train_patients & test_patients), 0, "Data leakage between train and test!")

    def test_myopspp_split_manifests(self):
        train_file = PROJECT_ROOT / "lists" / "lists_MyoPSpp" / "train.txt"
        test_file = PROJECT_ROOT / "lists" / "lists_MyoPSpp" / "test_vol.txt"

        self.assertTrue(train_file.is_file(), f"Missing {train_file}")
        self.assertTrue(test_file.is_file(), f"Missing {test_file}")

        train_lines = read_manifest_lines(train_file)
        test_lines = read_manifest_lines(test_file)

        self.assertEqual(len(train_lines), MYOPSPP_EXPECTED_TRAIN_SLICES, f"Expected {MYOPSPP_EXPECTED_TRAIN_SLICES} slices")
        train_patients = {patient_id(line) for line in train_lines}
        self.assertEqual(len(train_patients), MYOPSPP_EXPECTED_TRAIN_PATIENTS, f"Expected {MYOPSPP_EXPECTED_TRAIN_PATIENTS} patients")

        self.assertEqual(len(test_lines), MYOPSPP_EXPECTED_TEST_VOLUMES, f"Expected {MYOPSPP_EXPECTED_TEST_VOLUMES} test volumes")
        test_patients = {patient_id(line) for line in test_lines}
        self.assertEqual(len(test_patients), MYOPSPP_EXPECTED_TEST_VOLUMES)

        # Zero patient leakage
        self.assertEqual(len(train_patients & test_patients), 0, "Data leakage between train and test in MyoPS++!")

        # Verify case casing preserved (e.g., Case2001, Case2011)
        uppercase_cases = [p for p in test_patients if p.startswith("Case")]
        self.assertTrue(len(uppercase_cases) > 0, "Case ID casing not preserved (expected CaseXXXX)")


class TestLabelTransformationAndDoubleSwapPrevention(unittest.TestCase):
    """Test multi-head target generation and double-swap prevention."""

    def test_target_generation_legacy_mode(self):
        # Legacy label order: 0: BG, 1: Normal Myo, 2: Scar, 3: Edema-exclusive
        label = torch.tensor([0, 1, 2, 3], dtype=torch.long)
        target_C0, target_LGE, target_T2 = create_multi_head_targets(label, mode="legacy")

        # C0: 0 is BG, {1,2,3} is Whole Myo -> [0, 1, 1, 1]
        self.assertTrue(torch.equal(target_C0, torch.tensor([0, 1, 1, 1])))
        # LGE: 0 is BG, 1 is Non-scar Myo, 2 is Scar -> [0, 1, 2, 1]
        self.assertTrue(torch.equal(target_LGE, torch.tensor([0, 1, 2, 1])))
        # T2: 0 is BG, 1 is Normal Myo, 2 is Edema-inclusive -> [0, 1, 2, 2]
        self.assertTrue(torch.equal(target_T2, torch.tensor([0, 1, 2, 2])))

    def test_target_generation_canonical_mode(self):
        # Canonical label order: 0: BG, 1: Normal Myo, 2: Edema-exclusive, 3: Scar
        label = torch.tensor([0, 1, 2, 3], dtype=torch.long)
        target_C0, target_LGE, target_T2 = create_multi_head_targets(label, mode="canonical")

        # C0: 0 is BG, {1,2,3} is Whole Myo -> [0, 1, 1, 1]
        self.assertTrue(torch.equal(target_C0, torch.tensor([0, 1, 1, 1])))
        # LGE: 0 is BG, 1 is Non-scar Myo, 2 is Scar -> [0, 1, 1, 2]
        self.assertTrue(torch.equal(target_LGE, torch.tensor([0, 1, 1, 2])))
        # T2: 0 is BG, 1 is Normal Myo, 2 is Edema-inclusive -> [0, 1, 2, 2]
        self.assertTrue(torch.equal(target_T2, torch.tensor([0, 1, 2, 2])))

    def test_fuse_predictions(self):
        # Create synthetic multi-head probabilities
        B, H, W = 1, 4, 4
        seg_C0 = torch.zeros(B, 2, H, W)
        seg_LGE = torch.zeros(B, 3, H, W)
        seg_T2 = torch.zeros(B, 3, H, W)

        # Default all background
        seg_C0[:, 0, :, :] = 1.0
        seg_LGE[:, 0, :, :] = 1.0
        seg_T2[:, 0, :, :] = 1.0

        # Pixel (0, 0): Normal Myo (C0=1, LGE=1, T2=1)
        seg_C0[:, :, 0, 0] = torch.tensor([0.0, 1.0])
        seg_LGE[:, :, 0, 0] = torch.tensor([0.0, 1.0, 0.0])
        seg_T2[:, :, 0, 0] = torch.tensor([0.0, 1.0, 0.0])

        # Pixel (0, 1): Edema-exclusive (C0=1, LGE=1, T2=2)
        seg_C0[:, :, 0, 1] = torch.tensor([0.0, 1.0])
        seg_LGE[:, :, 0, 1] = torch.tensor([0.0, 1.0, 0.0])
        seg_T2[:, :, 0, 1] = torch.tensor([0.0, 0.0, 1.0])

        # Pixel (0, 2): Scar (C0=1, LGE=2, T2=2)
        seg_C0[:, :, 0, 2] = torch.tensor([0.0, 1.0])
        seg_LGE[:, :, 0, 2] = torch.tensor([0.0, 0.0, 1.0])
        seg_T2[:, :, 0, 2] = torch.tensor([0.0, 0.0, 1.0])

        fused = fuse_predictions(seg_C0, seg_LGE, seg_T2)
        self.assertEqual(fused[0, 0, 0].item(), 1)  # Normal Myocardium
        self.assertEqual(fused[0, 0, 1].item(), 2)  # Edema-exclusive
        self.assertEqual(fused[0, 0, 2].item(), 3)  # Scar
        self.assertEqual(fused[0, 0, 3].item(), 0)  # Background

    def test_double_swap_prevention_in_evaluation(self):
        # Synthetic GT volume: shape (10, 10, 2)
        # In canonical order: 1 is Myo, 2 is Edema, 3 is Scar
        gt_canonical = np.zeros((10, 10, 2), dtype=np.uint8)
        gt_canonical[1:4, 1:4, :] = 1  # Myo
        gt_canonical[4:7, 4:7, :] = 2  # Edema
        gt_canonical[7:9, 7:9, :] = 3  # Scar

        # Synthetic prediction in canonical order matching gt_canonical perfectly
        pred_canonical = gt_canonical.copy()

        # 1. On MyoPS++, GT is ALREADY Canonical:
        # evaluate_single_volume with dataset="myopspp" must produce perfect Dice 1.0 for all regions
        detailed_rows_pp, summary_pp = evaluate_single_volume(
            prediction=pred_canonical,
            target_raw=gt_canonical,
            case_id="CaseTestPP",
            dataset="myopspp",
        )
        self.assertAlmostEqual(summary_pp["edema_dice"], 1.0, places=5)
        self.assertAlmostEqual(summary_pp["scar_dice"], 1.0, places=5)
        self.assertAlmostEqual(summary_pp["myocardial_ring_dice"], 1.0, places=5)

        # 2. On MyoPS380, GT is LEGACY:
        # 1 is Myo, 2 is Scar, 3 is Edema
        gt_legacy = np.zeros((10, 10, 2), dtype=np.uint8)
        gt_legacy[1:4, 1:4, :] = 1  # Myo
        gt_legacy[4:7, 4:7, :] = 3  # Edema is ID 3 in Legacy
        gt_legacy[7:9, 7:9, :] = 2  # Scar is ID 2 in Legacy

        detailed_rows_380, summary_380 = evaluate_single_volume(
            prediction=pred_canonical,
            target_raw=gt_legacy,
            case_id="case0001",
            dataset="myops380",
        )
        self.assertAlmostEqual(summary_380["edema_dice"], 1.0, places=5)
        self.assertAlmostEqual(summary_380["scar_dice"], 1.0, places=5)
        self.assertAlmostEqual(summary_380["myocardial_ring_dice"], 1.0, places=5)


class TestDistanceMetricsAndSpacing(unittest.TestCase):
    """Test physical spacing handling and mm vs voxel distance calculation."""

    def test_binary_metrics_mm_vs_voxel(self):
        pred = np.zeros((20, 20, 5), dtype=bool)
        truth = np.zeros((20, 20, 5), dtype=bool)

        pred[5:15, 5:15, 1:4] = True
        truth[6:16, 6:16, 1:4] = True

        # Voxel distance (spacing=None)
        m_voxel = binary_metrics(pred, truth, spacing=None, compute_distance=True)
        self.assertIsNotNone(m_voxel["hd95"])
        self.assertIsNotNone(m_voxel["asd"])

        # Physical spacing in mm (e.g. 2.0, 2.0, 5.0)
        spacing = [2.0, 2.0, 5.0]
        m_mm = binary_metrics(pred, truth, spacing=spacing, compute_distance=True)
        self.assertIsNotNone(m_mm["hd95"])
        self.assertIsNotNone(m_mm["asd"])

        # With 2x spacing, physical distance should be roughly ~2x voxel distance for in-plane shifts
        self.assertGreater(m_mm["hd95"], m_voxel["hd95"])

    def test_cohort_summary_mm_keys(self):
        detailed_rows = [
            {
                "case": "Case2001",
                "region": "edema",
                "dice": 0.85,
                "iou": 0.74,
                "hd95": 4.5,
                "asd": 1.2,
                "status": "ok",
                "precision": 0.88,
                "recall": 0.82,
                "hd95_unit": "mm",
                "hd95_mm": 4.5,
                "asd_mm": 1.2,
            },
            {
                "case": "Case2001",
                "region": "scar",
                "dice": 0.90,
                "iou": 0.82,
                "hd95": 3.0,
                "asd": 0.8,
                "status": "ok",
                "precision": 0.92,
                "recall": 0.88,
                "hd95_unit": "mm",
                "hd95_mm": 3.0,
                "asd_mm": 0.8,
            },
            {
                "case": "Case2001",
                "region": "edema_inclusive",
                "dice": 0.88,
                "iou": 0.78,
                "hd95": 4.0,
                "asd": 1.0,
                "status": "ok",
                "precision": 0.90,
                "recall": 0.86,
                "hd95_unit": "mm",
                "hd95_mm": 4.0,
                "asd_mm": 1.0,
            },
            {
                "case": "Case2001",
                "region": "myocardial_ring",
                "dice": 0.95,
                "iou": 0.90,
                "hd95": 2.0,
                "asd": 0.5,
                "status": "ok",
                "precision": 0.96,
                "recall": 0.94,
                "hd95_unit": "mm",
                "hd95_mm": 2.0,
                "asd_mm": 0.5,
            },
            {
                "case": "Case2001",
                "region": "normal_myocardium",
                "dice": 0.92,
                "iou": 0.85,
                "hd95": 2.5,
                "asd": 0.6,
                "status": "ok",
                "precision": 0.93,
                "recall": 0.91,
                "hd95_unit": "mm",
                "hd95_mm": 2.5,
                "asd_mm": 0.6,
            },
        ]
        case_summaries = [{
            "case_id": "Case2001",
            "edema_dice": 0.85,
            "scar_dice": 0.90,
            "edema_inclusive_dice": 0.88,
            "myocardial_ring_dice": 0.95,
            "normal_myocardium_dice": 0.92,
            "avg_pathology_dice": 0.875,
            "avg_pathology_inclusive_dice": 0.89,
        }]

        summary, summary_rows = compute_cohort_summary(
            detailed_rows=detailed_rows,
            case_summaries=case_summaries,
            distance_unit="mm",
            protocol=BENCHMARK_PROTOCOLS["myopspp"],
        )

        self.assertIn("mean_hd95_mm", summary["edema"])
        self.assertEqual(summary["edema"]["mean_hd95_mm"], 4.5)
        self.assertIn("avg_pathology_hd95_mm", summary)
        self.assertEqual(summary["avg_pathology_hd95_mm"], 3.75)


class TestTrainingUpdatesAndSchedule(unittest.TestCase):
    """Test dynamic total update calculation for both datasets."""

    def test_total_updates_myops380(self):
        # 1,972 slices / batch_size 16 = 124 batches per epoch
        # 300 epochs * 124 batches = 37,200 updates
        batches_per_epoch = int(np.ceil(1972 / 16))
        self.assertEqual(batches_per_epoch, 124)
        updates = calculate_total_updates(epochs=300, batches_per_epoch=batches_per_epoch, accum_steps=1)
        self.assertEqual(updates, 37200)

    def test_total_updates_myopspp(self):
        # 296 slices / batch_size 16 = 19 batches per epoch (18 full batches + 1 batch of 8)
        # 300 epochs * 19 batches = 5,700 updates
        batches_per_epoch = int(np.ceil(296 / 16))
        self.assertEqual(batches_per_epoch, 19)
        updates = calculate_total_updates(epochs=300, batches_per_epoch=batches_per_epoch, accum_steps=1)
        self.assertEqual(updates, 5700)


class TestCoreInvariants(unittest.TestCase):
    """Test that core model architecture, loss, and augmentation are fully preserved."""

    def test_model_forward_pass_variants(self):
        model = MyoPSNetL()
        model.eval()

        # Variant 1: 3 separate tensors (B, 1, 128, 128)
        c0 = torch.randn(2, 1, 128, 128)
        lge = torch.randn(2, 1, 128, 128)
        t2 = torch.randn(2, 1, 128, 128)

        with torch.no_grad():
            s_c0, s_lge, s_t2 = model(c0, lge, t2)

        self.assertEqual(s_c0.shape, (2, 2, 128, 128))
        self.assertEqual(s_lge.shape, (2, 3, 128, 128))
        self.assertEqual(s_t2.shape, (2, 3, 128, 128))

        # Variant 2: Single combined tensor (B, 3, 128, 128)
        combined = torch.cat([c0, lge, t2], dim=1)
        with torch.no_grad():
            s_c0_c, s_lge_c, s_t2_c = model(combined)

        self.assertEqual(s_c0_c.shape, (2, 2, 128, 128))
        self.assertEqual(s_lge_c.shape, (2, 3, 128, 128))
        self.assertEqual(s_t2_c.shape, (2, 3, 128, 128))

    def test_myops_loss(self):
        criterion = MyoPSLoss()
        s_c0 = torch.randn(2, 2, 128, 128, requires_grad=True)
        s_lge = torch.randn(2, 3, 128, 128, requires_grad=True)
        s_t2 = torch.randn(2, 3, 128, 128, requires_grad=True)

        t_c0 = torch.randint(0, 2, (2, 128, 128), dtype=torch.long)
        t_lge = torch.randint(0, 3, (2, 128, 128), dtype=torch.long)
        t_t2 = torch.randint(0, 3, (2, 128, 128), dtype=torch.long)

        loss_total, loss_seg, loss_inv, loss_inc = criterion(s_c0, s_lge, s_t2, t_c0, t_lge, t_t2)

        self.assertTrue(torch.isfinite(loss_total))
        self.assertTrue(torch.isfinite(loss_seg))
        self.assertTrue(torch.isfinite(loss_inv))
        self.assertTrue(torch.isfinite(loss_inc))

        # Backward test
        loss_total.backward()
        self.assertIsNotNone(s_c0.grad)
        self.assertIsNotNone(s_lge.grad)
        self.assertIsNotNone(s_t2.grad)

    def test_augmentation_contract(self):
        transform = RandomGenerator(output_size=[128, 128])
        sample = {
            "bSSFP": np.random.rand(128, 128).astype(np.float32),
            "LGE": np.random.rand(128, 128).astype(np.float32),
            "T2w": np.random.rand(128, 128).astype(np.float32),
            "label": np.random.randint(0, 4, (128, 128)).astype(np.uint8),
        }
        augmented = transform(sample)
        self.assertEqual(augmented["bSSFP"].shape, torch.Size([1, 128, 128]))
        self.assertEqual(augmented["LGE"].shape, torch.Size([1, 128, 128]))
        self.assertEqual(augmented["T2w"].shape, torch.Size([1, 128, 128]))
        self.assertEqual(augmented["label"].shape, torch.Size([128, 128]))


class TestEfficiencyMetrics(unittest.TestCase):
    """Test compliance with Quy tắc đo hiệu suất tính toán.txt."""

    def test_parameter_counting(self):
        model = MyoPSNetL()
        params = count_parameters(model)
        self.assertEqual(params["trainable_params"], 119284360)
        self.assertAlmostEqual(params["trainable_params_m"], 119.28, places=2)

    def test_flop_counting(self):
        model = MyoPSNetL()
        flops_info = measure_flops(model, input_shape=(1, 3, 128, 128), device="cpu")
        self.assertGreater(flops_info["flops_total"], 100_000_000_000)
        self.assertAlmostEqual(flops_info["flops_g"], 112.49, delta=0.5)

    def test_rule_compliance_checker(self):
        compliant, warnings, hw = check_rule_compliance(
            device=torch.device("cpu"),
            amp_mode="none",
            num_workers=0,
            pin_memory=True,
            batch_size=16,
            planned_epochs=300,
        )
        # On local CPU, warning about non-A100 GPU is expected
        self.assertFalse(compliant)
        self.assertTrue(any("A100" in w for w in warnings))


class TestCheckpointValidation(unittest.TestCase):
    """Test checkpoint payload validation for both datasets."""

    def test_checkpoint_payload_validation_myopspp(self):
        payload = {
            "epoch": 300,
            "completed_epochs": 300,
            "planned_epochs": 300,
            "global_step": 5700,
            "training_complete": True,
            "model_state_dict": {},
            "optimizer_state_dict": {},
            "scheduler_state_dict": {},
            "scaler_state_dict": {},
            "seed": 1234,
            "opt_protocol": OPTIMIZATION_PROTOCOL,
            "rng_state": capture_rng_state(),
            "protocol_id": "myopspp_train64_test16_last_v1",
            "model_contract": {"id": "myopsnet_binary_mpc_v1"},
            "loss_contract": {"id": "myops_loss_ce_dice_inv_inc_v1"},
        }
        # Validation should succeed without raising
        validate_checkpoint_payload(payload, require_complete=True)

        with self.assertRaises(ValueError):
            bad_payload = dict(payload, protocol_id="unknown_protocol_xyz")
            validate_checkpoint_payload(bad_payload, require_complete=True)

    def test_checkpoint_payload_validation_myops380(self):
        payload = {
            "epoch": 300,
            "completed_epochs": 300,
            "planned_epochs": 300,
            "global_step": 37200,
            "training_complete": True,
            "model_state_dict": {},
            "optimizer_state_dict": {},
            "scheduler_state_dict": {},
            "scaler_state_dict": {},
            "seed": 1234,
            "opt_protocol": OPTIMIZATION_PROTOCOL,
            "rng_state": capture_rng_state(),
            "protocol_id": "myops380_train304_test76_last_v1",
            "model_contract": {"id": "myopsnet_binary_mpc_v1"},
            "loss_contract": {"id": "myops_loss_ce_dice_inv_inc_v1"},
        }
        # Validation should succeed without raising
        validate_checkpoint_payload(payload, require_complete=True)

        with self.assertRaises(ValueError):
            bad_payload = dict(payload, training_complete=False)
            validate_checkpoint_payload(bad_payload, require_complete=True)

    def test_efficiency_metrics_schema_fields(self):
        """Ensure latency_ms and epoch_times_1_to_5 exist in efficiency payload."""
        import train
        model = MyoPSNetL()
        inf_info = measure_inference_efficiency(model, input_shape=(1, 3, 128, 128), warmup=1, repeats=2)
        self.assertIn("latency_ms", inf_info)
        self.assertIn("latency_ms_per_slice", inf_info)
        self.assertEqual(inf_info["latency_ms"], inf_info["latency_ms_per_slice"])

        trn_info = summarize_training_efficiency(
            epoch_times_s=[10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
            total_wall_clock_s=91.0,
            peak_vram_bytes=1024 * 1024 * 100,
            planned_epochs=7,
            warmup_epochs=5,
        )
        self.assertIn("epoch_times_1_to_5", trn_info)
        self.assertEqual(len(trn_info["epoch_times_1_to_5"]), 5)
        self.assertEqual(trn_info["epoch_times_1_to_5"], [10.0, 11.0, 12.0, 13.0, 14.0])

    def test_evaluate_predictions_composite_fallbacks(self):
        """Ensure mm distance summary produces fallback voxel keys for legacy parsers."""
        detailed_rows = [
            {"case": "CaseTest", "region": r, "dice": 0.9, "iou": 0.8, "hd95": 2.5, "asd": 0.5,
             "status": "ok", "precision": 0.9, "recall": 0.9, "hd95_unit": "mm", "hd95_mm": 2.5, "asd_mm": 0.5}
            for r in ("scar", "edema", "edema_inclusive", "myocardial_ring", "normal_myocardium")
        ]
        case_summaries = [{
            "case_id": "CaseTest", "scar_dice": 0.9, "edema_dice": 0.9, "edema_inclusive_dice": 0.9,
            "myocardial_ring_dice": 0.9, "normal_myocardium_dice": 0.9, "avg_pathology_dice": 0.9,
            "avg_pathology_inclusive_dice": 0.9,
        }]
        summary, _ = compute_cohort_summary(
            detailed_rows=detailed_rows,
            case_summaries=case_summaries,
            distance_unit="mm",
            protocol=BENCHMARK_PROTOCOLS["myopspp"],
        )
        self.assertIn("avg_pathology_inclusive_hd95_mm", summary)
        self.assertIn("avg_pathology_inclusive_hd95_voxel", summary)
        self.assertEqual(summary["avg_pathology_inclusive_hd95_voxel"], summary["avg_pathology_inclusive_hd95_mm"])

    def test_myops_test_dataset_synthetic_h5_loading(self):
        """Test MyoPSTestDataset against synthetic H5 files with both spacing and labels."""
        import h5py
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            split_file = tmp_path / "test_vol.txt"
            split_file.write_text("Case2001\n", encoding="utf-8")

            # Create test_vol_h5 layout
            for mod in ("bSSFP", "LGE", "T2w"):
                mod_dir = tmp_path / mod / "test_vol_h5"
                mod_dir.mkdir(parents=True, exist_ok=True)
                h5_file = mod_dir / "Case2001.npy.h5"
                with h5py.File(h5_file, "w") as f:
                    f.create_dataset("image", data=np.zeros((128, 128, 4), dtype=np.float32))
                    f.create_dataset("label", data=np.ones((128, 128, 4), dtype=np.int64))
                    f.create_dataset("spacing", data=np.array([1.0, 1.0, 2.5], dtype=np.float32))

            ds = MyoPSTestDataset(split_file=split_file, data_root=tmp_path, dataset="myopspp")
            self.assertEqual(len(ds), 1)
            sample = ds[0]
            self.assertEqual(sample["case_id"], "Case2001")
            self.assertIn("spacing", sample)
            np.testing.assert_allclose(sample["spacing"].numpy(), [1.0, 1.0, 2.5])


if __name__ == "__main__":
    unittest.main()
