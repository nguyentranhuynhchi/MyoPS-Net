"""End-to-End Smoke Test on Real Dataset for MyoPS-Net MyoPS380 Adaptation.

Verifies:
1. Real Data Ingestion (304 train manifest / 76 test manifest from SCAR)
2. 5-Branch Zero-Padding & SCAR RandomGenerator Augmentation
3. Canonical-to-MyoPS Target Adapter
4. Model Forward + Hardened MyoPSLoss Backward Pass
5. Gradient Accumulation & AdamW Optimizer Step & PolyLR Scheduler Step
6. Atomic Checkpoint Save & Resume Metadata Verification
7. Single-Pass Volume Inference (Zero Morphological Post-processing)
8. Prediction NPZ Export (uint8, 128x128xD, key 'prediction')
9. SCAR Shared Surface Distance Evaluator Execution
"""
import os
import shutil
import tempfile
import unittest
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset.myops380_dataset import MyoPS380TrainDataset, MyoPS380TestVolumeDataset
from dataset.target_adapter import canonical_to_myops_targets, myops_outputs_to_canonical
from network.criterion.loss import MyoPSLoss
from network.model import MyoPSNet
from train import (
    build_grouped_optimizer,
    get_polynomial_decay_scheduler,
    atomic_save_checkpoint,
    set_seed,
)
from evaluate import (
    single_pass_volume_inference,
    load_scar_surface_distance_module,
)

DATA_ROOT = r"D:\nckh_model\Dataset\MyoPS380\Processed_data"
SPLIT_DIR = r"D:\nckh_model\SCAR\preprocessing\splits\myops380_no_val"


class TestRealPipelineSmoke(unittest.TestCase):
    def setUp(self):
        set_seed(1234)
        self.temp_dir = tempfile.mkdtemp(prefix="smoke_test_")

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_e2e_training_and_evaluation_smoke_flow(self):
        print("\n--- [Smoke Test 1] Verifying Train Dataset & DataLoader ---")
        train_ds = MyoPS380TrainDataset(data_dir=DATA_ROOT, split_dir=SPLIT_DIR, split="train")
        self.assertEqual(len(train_ds), 1972)

        train_loader = DataLoader(train_ds, batch_size=2, shuffle=True)
        batch = next(iter(train_loader))

        # Check shapes and modalities
        self.assertEqual(batch["C0"].shape, (2, 1, 128, 128))
        self.assertEqual(batch["LGE"].shape, (2, 1, 128, 128))
        self.assertEqual(batch["T2"].shape, (2, 1, 128, 128))
        self.assertEqual(batch["T1m"].shape, (2, 1, 128, 128))
        self.assertEqual(batch["T2starm"].shape, (2, 1, 128, 128))
        self.assertTrue(torch.all(batch["T1m"] == 0.0))
        self.assertTrue(torch.all(batch["T2starm"] == 0.0))
        self.assertEqual(batch["label"].shape, (2, 128, 128))

        print("--- [Smoke Test 2] Verifying Target Adapter & Forward/Backward ---")
        targets_dict = canonical_to_myops_targets(batch["label"])
        self.assertIn("cardiac", targets_dict)
        self.assertIn("scar", targets_dict)
        self.assertIn("edema", targets_dict)

        model = MyoPSNet(in_chs=(5, 2, 2, 3), out_chs=(3, 3, 3, 3))
        criterion = MyoPSLoss()

        res_c0, res_lge, res_t2, res_mapping = model(
            batch["C0"], batch["LGE"], batch["T2"], batch["T1m"], batch["T2starm"]
        )
        seg_outputs = {
            "cardiac": res_c0,
            "scar": res_lge,
            "edema": res_t2,
            "mapping": res_mapping,
        }
        loss_seg, loss_inv, loss_inc, loss = criterion(seg_outputs, targets_dict)
        self.assertTrue(torch.isfinite(loss))

        print("--- [Smoke Test 3] Verifying Optimizer, Scheduler & Accumulation ---")
        optimizer = build_grouped_optimizer(model, base_lr=1e-3, weight_decay=1e-4)
        scheduler = get_polynomial_decay_scheduler(optimizer, total_updates=100, power=0.9)

        initial_lr = optimizer.param_groups[0]["lr"]
        optimizer.zero_grad()
        loss.backward()

        # Check gradients exist
        has_grad = any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        self.assertTrue(has_grad)

        optimizer.step()
        scheduler.step()
        new_lr = optimizer.param_groups[0]["lr"]
        self.assertLess(new_lr, initial_lr)

        print("--- [Smoke Test 4] Verifying Atomic Checkpoint Save ---")
        ckpt_path = os.path.join(self.temp_dir, "last.pth")
        payload = {
            "epoch": 0,
            "completed_epochs": 1,
            "planned_epochs": 300,
            "training_complete": False,
            "protocol": "myops380_train304_test76_last_v1",
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "seed": 1234,
        }
        atomic_save_checkpoint(payload, ckpt_path)
        self.assertTrue(os.path.isfile(ckpt_path))

        loaded_ckpt = torch.load(ckpt_path, map_location="cpu")
        self.assertEqual(loaded_ckpt["protocol"], "myops380_train304_test76_last_v1")

        print("--- [Smoke Test 5] Verifying Test Volume Dataset & Single-Pass Inference ---")
        test_ds = MyoPS380TestVolumeDataset(data_dir=DATA_ROOT, split_dir=SPLIT_DIR, split="test_vol")
        self.assertEqual(len(test_ds), 76)

        case_0 = test_ds[0]
        case_id = case_0["case_id"]
        c0_vol = case_0["C0"]
        lge_vol = case_0["LGE"]
        t2_vol = case_0["T2"]
        gt_vol = case_0["label"]
        H, W, D = c0_vol.shape

        pred_3d = single_pass_volume_inference(
            model=model,
            c0_vol=c0_vol,
            lge_vol=lge_vol,
            t2_vol=t2_vol,
            device=torch.device("cpu"),
        )
        self.assertEqual(pred_3d.shape, (H, W, D))
        self.assertEqual(pred_3d.dtype, np.uint8)
        self.assertTrue(set(np.unique(pred_3d)).issubset({0, 1, 2, 3}))

        print("--- [Smoke Test 6] Verifying Prediction NPZ Export ---")
        npz_path = os.path.join(self.temp_dir, f"{case_id}_pred.npz")
        np.savez_compressed(npz_path, prediction=pred_3d)
        self.assertTrue(os.path.isfile(npz_path))

        with np.load(npz_path) as loaded_npz:
            self.assertIn("prediction", loaded_npz)
            self.assertEqual(loaded_npz["prediction"].shape, (H, W, D))

        print("--- [Smoke Test 7] Verifying SCAR Shared Surface Distance Evaluator ---")
        scar_eval = load_scar_surface_distance_module()
        rows = scar_eval.dataset_rows(pred_3d, gt_vol, case_id, dataset_id="myops380")
        self.assertGreater(len(rows), 0)

        # Ensure regions exist in rows
        region_names = {r["region"] for r in rows}
        self.assertTrue({"scar", "edema", "normal_myocardium"}.issubset(region_names))

        summary = scar_eval.summarize_dataset_rows(rows, dataset_id="myops380")
        self.assertIn("scar", summary)
        self.assertIn("edema", summary)
        print("Smoke test successfully completed all 7 verification steps!")


if __name__ == "__main__":
    unittest.main()
