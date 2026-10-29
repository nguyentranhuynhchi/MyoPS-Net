"""Production Inference & Shared Evaluation for MyoPS-Net on MyoPS380 Benchmark.

Strictly adheres to:
- Benchmark Protocol: myops380_train304_test76_last_v1
- Metric Protocol: myops380_voxel_v3
- Shared Evaluator: Reuses SCAR surface_distance.py (dataset_rows, summarize_dataset_rows)
- Requirements:
  - Loads atomic last.pth at epoch 300
  - Single-pass forward + argmax on all 76 test patients from test_vol.txt
  - Zero morphological postprocessing (no LCC, no connected components filtering)
  - Exports test_predictions/{CaseID}_pred.npz (uint8, 128x128xD, key 'prediction')
  - Produces metrics.json and per_case.csv
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import h5py
import numpy as np
import torch
import torch.nn as nn

from dataset.myops380_dataset import MyoPS380TestVolumeDataset, extract_patient_id
from dataset.target_adapter import myops_outputs_to_canonical
from network.model import MyoPSNet


def load_scar_surface_distance_module(scar_path: str | Path | None = None):
    """Dynamically loads SCAR's training/metrics/surface_distance.py."""
    candidates = []
    if scar_path is not None:
        candidates.append(Path(scar_path))
    candidates.extend([
        Path(r"D:\nckh_model\SCAR\training\metrics\surface_distance.py"),
        Path(r"../SCAR/training/metrics/surface_distance.py"),
    ])

    for p in candidates:
        if p.is_file():
            spec = importlib.util.spec_from_file_location("scar_surface_distance", str(p.resolve()))
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                sys.modules["scar_surface_distance"] = mod
                spec.loader.exec_module(mod)
                return mod

    raise FileNotFoundError(
        f"Cannot locate SCAR surface_distance.py in candidates: {[str(c) for c in candidates]}. "
        "Please provide --scar_root or ensure D:\\nckh_model\\SCAR exists."
    )


def compute_file_sha256(path: str | Path) -> str:
    """Computes SHA256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def single_pass_volume_inference(
    model: nn.Module,
    c0_vol: np.ndarray,
    lge_vol: np.ndarray,
    t2_vol: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Runs single-pass forward + argmax on a 3D volume (128, 128, D).

    Strictly disables all morphological postprocessing (no LCC, no small CC filtering).
    Returns uint8 array (128, 128, D) with canonical labels in {0, 1, 2, 3}.
    """
    model.eval()
    H, W, D = c0_vol.shape
    pred_3d = np.zeros((H, W, D), dtype=np.uint8)

    with torch.no_grad():
        for d in range(D):
            # Shape (1, 1, H, W)
            slice_c0 = torch.from_numpy(c0_vol[:, :, d]).unsqueeze(0).unsqueeze(0).float().to(device)
            slice_lge = torch.from_numpy(lge_vol[:, :, d]).unsqueeze(0).unsqueeze(0).float().to(device)
            slice_t2 = torch.from_numpy(t2_vol[:, :, d]).unsqueeze(0).unsqueeze(0).float().to(device)
            slice_t1m = torch.zeros_like(slice_c0)
            slice_t2starm = torch.zeros_like(slice_c0)

            res_c0, res_lge, res_t2, res_mapping = model(
                slice_c0, slice_lge, slice_t2, slice_t1m, slice_t2starm
            )
            seg_outputs = {
                "cardiac": res_c0,
                "scar": res_lge,
                "edema": res_t2,
                "mapping": res_mapping,
            }
            canonical_slice = myops_outputs_to_canonical(seg_outputs)  # shape (1, H, W)
            pred_3d[:, :, d] = canonical_slice[0].cpu().numpy().astype(np.uint8)

    return pred_3d


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    """Executes full evaluation on the 76 test volumes."""
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("================================================================================")
    print("  MyoPS-Net Shared Evaluator — Protocol: myops380_train304_test76_last_v1")
    print(f"  Checkpoint Path : {checkpoint_path}")
    print(f"  Data Root       : {args.data_root}")
    print(f"  Split Dir       : {args.split_dir}")
    print(f"  Output Dir      : {output_dir}")
    print(f"  Device          : {device}")
    print("================================================================================")

    # 1. Validate checkpoint
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    ckpt_hash = compute_file_sha256(checkpoint_path)

    # Enforce protocol compliance
    completed_epochs = checkpoint.get("completed_epochs", checkpoint.get("epoch", -1))
    planned_epochs = checkpoint.get("planned_epochs", 300)
    protocol = checkpoint.get("protocol", "unknown")
    training_complete = checkpoint.get("training_complete", False)

    print(f"Checkpoint metadata: protocol={protocol}, epochs={completed_epochs}/{planned_epochs}, complete={training_complete}")

    if not args.ignore_incomplete:
        if not training_complete or completed_epochs < 300:
            raise ValueError(
                f"Evaluator rejected checkpoint: Expected completed 300 epochs under protocol "
                f"'myops380_train304_test76_last_v1', but got completed_epochs={completed_epochs}, "
                f"training_complete={training_complete}. Only the final epoch 300 checkpoint is eligible."
            )

    # 2. Load model
    model = MyoPSNet(in_chs=(5, 2, 2, 3), out_chs=(3, 3, 3, 3)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # 3. Load SCAR surface_distance module
    scar_eval = load_scar_surface_distance_module(args.scar_surface_distance)

    # 4. Load test dataset (76 test volumes)
    test_dataset = MyoPS380TestVolumeDataset(
        data_dir=args.data_root,
        split_dir=args.split_dir,
        split="test_vol",
    )
    print(f"Loaded {len(test_dataset)} test patient volumes from {args.split_dir}/test_vol.txt.")

    # 5. Run inference and shared evaluation
    all_rows = []
    cases_processed = 0

    print("\n--- Starting Single-Pass Test Volume Inference & Metric Calculation ---")
    start_eval_time = time.time()

    for idx in range(len(test_dataset)):
        case_data = test_dataset[idx]
        case_id = case_data["case_id"]

        c0_vol = case_data["C0"]
        lge_vol = case_data["LGE"]
        t2_vol = case_data["T2"]
        gt_vol = case_data["label"]

        # Run inference
        pred_3d = single_pass_volume_inference(model, c0_vol, lge_vol, t2_vol, device)

        # Export prediction NPZ
        npz_path = output_dir / f"{case_id}_pred.npz"
        np.savez_compressed(npz_path, prediction=pred_3d.astype(np.uint8))

        # Run shared evaluator row calculation
        rows = scar_eval.dataset_rows(pred_3d, gt_vol, case_id, dataset_id="myops380")
        all_rows.extend(rows)
        cases_processed += 1

        if (idx + 1) % 10 == 0 or (idx + 1) == len(test_dataset):
            print(f"  Processed [{idx + 1:02d}/{len(test_dataset):02d}] cases: {case_id} -> {npz_path.name}")

    eval_duration = time.time() - start_eval_time
    print(f"\nInference & metric calculation completed in {eval_duration:.2f}s.")

    # 6. Summarize rows
    summary = scar_eval.summarize_dataset_rows(all_rows, dataset_id="myops380")

    # Add protocol metadata
    summary["metadata"] = {
        "protocol_id": "myops380_train304_test76_last_v1",
        "metric_protocol": "myops380_voxel_v3",
        "dataset_id": "myops380",
        "checkpoint_selection": "last_completed_epoch",
        "checkpoint_selection_metric": None,
        "completed_epochs": completed_epochs,
        "planned_epochs": planned_epochs,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": ckpt_hash,
        "seed": checkpoint.get("seed", 1234),
        "cases_evaluated": cases_processed,
        "evaluation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    # 7. Write metrics.json
    metrics_json_path = output_dir / "metrics.json"
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved metrics summary to: {metrics_json_path}")

    # 8. Write per_case.csv
    per_case_csv_path = output_dir / "per_case.csv"
    if all_rows:
        fieldnames = list(all_rows[0].keys())
        with open(per_case_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"Saved per-case detailed results to: {per_case_csv_path}")

    # 9. Print formatted benchmark summary
    print("\n================================================================================")
    print("                    MYOPS380 BENCHMARK EVALUATION RESULTS")
    print("================================================================================")
    for region_name in ["scar", "edema", "edema_inclusive", "normal_myocardium", "myocardial_ring"]:
        if region_name in summary:
            reg = summary[region_name]
            dice_mean = reg.get("dice_patient_mean", reg.get("dice_mean", 0.0))
            hd95_mean = reg.get("hd95_patient_mean", reg.get("hd95_mean", "undefined"))
            asd_mean = reg.get("asd_patient_mean", reg.get("asd_mean", "undefined"))
            hd_str = f"{hd95_mean:.2f}" if isinstance(hd95_mean, (int, float)) else str(hd95_mean)
            asd_str = f"{asd_mean:.2f}" if isinstance(asd_mean, (int, float)) else str(asd_mean)
            print(f"  Region [{region_name:<18}] | Dice: {dice_mean:.4f} | HD95 (vox): {hd_str:<6} | ASD (vox): {asd_str}")

    if "avg_pathology_inclusive_dice" in summary:
        print("--------------------------------------------------------------------------------")
        print(f"  Avg Pathology Inclusive Dice (Report): {summary['avg_pathology_inclusive_dice']:.4f}")
    if "avg_pathology_dice" in summary:
        print(f"  Avg Pathology Exclusive Dice (Report): {summary['avg_pathology_dice']:.4f}")
    print("================================================================================")

    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    """Builds CLI argument parser for evaluation."""
    parser = argparse.ArgumentParser(
        description="Run evaluation of MyoPS-Net on MyoPS380 76 test volumes."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=r"checkpoints\last.pth",
        help="Path to final model checkpoint (last.pth).",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=r"D:\nckh_model\Dataset\MyoPS380\Processed_data",
        help="Path to preprocessed MyoPS380 dataset directory.",
    )
    parser.add_argument(
        "--split_dir",
        type=str,
        default=r"D:\nckh_model\SCAR\preprocessing\splits\myops380_no_val",
        help="Path to split manifests directory (test_vol.txt).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=r"test_predictions",
        help="Directory to save prediction .npz files and evaluation results.",
    )
    parser.add_argument(
        "--scar_surface_distance",
        type=str,
        default=r"D:\nckh_model\SCAR\training\metrics\surface_distance.py",
        help="Path to SCAR surface_distance.py metric module.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to evaluate on ('cuda' or 'cpu').",
    )
    parser.add_argument(
        "--ignore_incomplete",
        action="store_true",
        default=False,
        help="Allow evaluating checkpoints with fewer than 300 epochs (for testing only).",
    )
    return parser


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()
    evaluate(args)
