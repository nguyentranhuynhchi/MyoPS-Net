"""
predict.py
==========
Deterministic Single-Pass Volume Inference for MyoPS-Net Baseline.
Protocol: myops380_train304_test76_last_v1
Benchmark: MyoPS380 (76 Held-Out Test Volumes / Unit Voxel Surface Distance)

Key Requirements:
  - Checkpoint loading: <run_dir>/checkpoints/myopsnet_seed<seed>.pt
  - Single-pass model.eval(), torch.inference_mode()
  - Process all 76 cases from test_vol.txt (or subset in --dry_run)
  - Slice batching with bilinear spatial resizing (if H, W != img_size)
  - Strict NO test-time augmentation (TTA)
  - Strict NO post-processing / largest connected component (LCC)
  - Export: <run_dir>/test_predictions/<CaseID>_pred.npz with key 'prediction' (uint8, [0..3])
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset.myops380_dataset import (
    EXPECTED_TEST_VOLUMES,
    MYOPSPP_EXPECTED_TEST_VOLUMES,
    MyoPS380TestDataset,
    MyoPSTestDataset,
    normalize_dataset_name,
    read_manifest_lines,
    resolve_split_file,
)
from network.model import MyoPSNetL
from training.checkpoint import load_checkpoint, validate_checkpoint_payload
from utils.tools import fuse_predictions


CLASS_NAMES = ["background", "normal_myocardium", "edema", "scar"]
BENCHMARK_PROTOCOL_ID = "myops380_voxel_v3"
TRAINING_PROTOCOL_ID = "myops380_train304_test76_last_v1"


def setup_inference_logger(log_file: Optional[Path] = None) -> logging.Logger:
    """Configures console and optional file logging for inference."""
    logger = logging.getLogger("myopsnet.predict")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream_h = logging.StreamHandler(sys.stdout)
    stream_h.setFormatter(formatter)
    logger.addHandler(stream_h)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_h = logging.FileHandler(log_file, encoding="utf-8")
        file_h.setFormatter(formatter)
        logger.addHandler(file_h)

    return logger


def resolve_device(device_name: str = "auto") -> torch.device:
    """Resolves execution device."""
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def parse_amp_mode(amp_arg: Union[str, bool]) -> str:
    """Normalizes AMP mode argument."""
    if isinstance(amp_arg, bool):
        return "auto" if amp_arg else "none"
    s = str(amp_arg).strip().lower()
    if s in ("true", "1", "yes", "auto"):
        return "auto"
    if s in ("false", "0", "no", "none"):
        return "none"
    if s in ("fp16", "bf16"):
        return s
    return "auto"


def resolve_amp_dtype(amp_mode: str, device: torch.device) -> Optional[torch.dtype]:
    """Resolves AMP autocast floating point dtype."""
    if amp_mode == "none" or device.type == "cpu":
        return None
    if amp_mode == "auto":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if amp_mode == "fp16":
        return torch.float16
    if amp_mode == "bf16":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return None


@torch.inference_mode()
def predict_volume(
    model: nn.Module,
    images: Sequence[Union[np.ndarray, torch.Tensor]],
    img_size: int = 128,
    batch_size: int = 8,
    device: Optional[Union[str, torch.device]] = None,
    amp_dtype: Optional[torch.dtype] = None,
) -> np.ndarray:
    """Deterministic, batched single-pass 3D volume inference with MyoPS-Net-L.

    Args:
        model: MyoPSNetL model instance (must be in eval mode).
        images: 3 aligned 3D volumes [bSSFP, LGE, T2w], each with shape (H, W, D).
        img_size: Model spatial input resolution (default 128).
        batch_size: Number of adjacent slices to process per forward pass.
        device: Hardware device for inference.
        amp_dtype: AMP autocast dtype (None for float32).

    Returns:
        prediction: uint8 NumPy array of shape (H, W, D) with canonical values in {0, 1, 2, 3}.
    """
    arrays = [np.asarray(val, dtype=np.float32) for val in images]
    if len(arrays) != 3:
        raise ValueError(f"Expected exactly 3 aligned modality arrays, got {len(arrays)}")
    if any(a.ndim != 3 for a in arrays):
        raise ValueError(f"Expected 3D arrays of shape (H, W, D), got shapes {[a.shape for a in arrays]}")
    if any(a.shape != arrays[0].shape for a in arrays):
        raise ValueError(f"Modalities must have identical shapes, got {[a.shape for a in arrays]}")
    if not all(np.isfinite(a).all() for a in arrays):
        raise ValueError("Non-finite values detected in input volume arrays")
    if batch_size < 1:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    dev = torch.device(device or next(model.parameters()).device)
    height, width, depth = arrays[0].shape
    prediction = np.empty((height, width, depth), dtype=np.uint8)

    was_training = model.training
    model.eval()

    try:
        for start in range(0, depth, batch_size):
            stop = min(start + batch_size, depth)

            # Extract slices: (H, W, B_slice) -> transpose -> (B_slice, H, W) -> unsqueeze -> (B_slice, 1, H, W)
            inputs = [
                torch.from_numpy(
                    np.ascontiguousarray(a[:, :, start:stop].transpose(2, 0, 1))
                ).unsqueeze(1).to(dev, non_blocking=True)
                for a in arrays
            ]

            # Spatial resize to model input dimension if needed
            if height != img_size or width != img_size:
                inputs = [
                    F.interpolate(x, size=(img_size, img_size), mode="bilinear", align_corners=False)
                    for x in inputs
                ]

            if dev.type == "cuda" and amp_dtype is not None:
                autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=amp_dtype)
            else:
                autocast_ctx = contextlib.nullcontext()

            with autocast_ctx:
                seg_C0, seg_LGE, seg_T2 = model(inputs[0], inputs[1], inputs[2])

            if not (torch.isfinite(seg_C0).all() and torch.isfinite(seg_LGE).all() and torch.isfinite(seg_T2).all()):
                raise FloatingPointError(f"Non-finite output detected in slice range {start}:{stop}")

            # Resize continuous probability maps back to original grid if needed
            if seg_C0.shape[-2:] != (height, width):
                seg_C0 = F.interpolate(seg_C0.float(), size=(height, width), mode="bilinear", align_corners=False)
                seg_LGE = F.interpolate(seg_LGE.float(), size=(height, width), mode="bilinear", align_corners=False)
                seg_T2 = F.interpolate(seg_T2.float(), size=(height, width), mode="bilinear", align_corners=False)

            # Hierarchical canonical fusion (strict NO LCC, NO TTA)
            canonical_batch = fuse_predictions(seg_C0, seg_LGE, seg_T2)  # (B_slice, H, W) uint8

            # Place into full volume prediction: (B_slice, H, W) -> (H, W, B_slice)
            prediction[:, :, start:stop] = canonical_batch.cpu().numpy().transpose(1, 2, 0)
    finally:
        model.train(was_training)

    return prediction


def build_parser() -> argparse.ArgumentParser:
    """Constructs the CLI argument parser for volume inference."""
    parser = argparse.ArgumentParser(
        description="Deterministic Single-Pass Volume Inference for MyoPS-Net Baseline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Dataset selection
    parser.add_argument(
        "--dataset",
        default="myops380",
        choices=["myops380", "myopspp", "myopspp_unet2d_roi128_1mm"],
        help="Dataset choice: myops380 or myopspp",
    )
    # Checkpoint & directories
    parser.add_argument("--checkpoint", default=None, help="Path to checkpoint .pt file (defaults to <run_dir>/checkpoints/myopsnet_seed<seed>.pt)")
    parser.add_argument("--run_dir", default=None, help="Root directory of training run (defaults to runs/myopsnet_seed<seed>)")
    parser.add_argument("--seed", type=int, default=1234, help="Seed identifier matching checkpoint name")
    parser.add_argument(
        "--pred_dir", "--output_pred_dir", "--output_dir",
        default=None,
        dest="pred_dir",
        help="Output directory for predictions (defaults to <run_dir>/test_predictions)",
    )

    # Data paths
    parser.add_argument("--data_root", "--data-root", default=None, help="Dataset root containing bSSFP, LGE, T2w test_vol_h5")
    parser.add_argument("--split_dir", "--list_dir", default=None, help="Directory containing test_vol.txt")
    parser.add_argument("--split", default="test_vol", help="Split name to evaluate (default: test_vol)")
    parser.add_argument("--test_manifest", default=None, help="Direct path to test manifest file (overrides --split_dir/--split.txt)")

    # Inference settings
    parser.add_argument("--batch_size", "--batch-size", type=int, default=8, help="Number of slices per forward pass batch")
    parser.add_argument("--device", default="auto", help="Execution device: auto, cuda, cpu, cuda:0")
    parser.add_argument("--amp", default="auto", help="AMP autocast mode: auto, none, fp16, bf16")
    parser.add_argument("--img_size", type=int, default=128, help="Model grid resolution")

    # Options & validation
    parser.add_argument("--require_complete", action=argparse.BooleanOptionalAction, default=False, help="Require checkpoint to have completed all 300 epochs")
    parser.add_argument("--strict", action=argparse.BooleanOptionalAction, default=False, help="Strictly enforce expected test volume count")
    parser.add_argument("--save_nii", action=argparse.BooleanOptionalAction, default=False, help="Also export NIfTI (.nii.gz) format alongside .npz")
    parser.add_argument("--dry_run", "--smoke_test", action="store_true", help="Run fast test inference on first 2 cases only")
    parser.add_argument("--cases", nargs="+", default=None, help="Explicit list of CaseIDs to predict (e.g. --cases case0001 case0006)")

    return parser


def resolve_paths(args: argparse.Namespace) -> Tuple[Path, Path, Path, Path, Path]:
    """Resolves checkpoint path, run directory, predictions output directory, manifest path, and data root."""
    if args.run_dir is not None:
        run_dir = Path(args.run_dir).resolve()
    elif args.checkpoint is not None:
        cp = Path(args.checkpoint).resolve()
        if cp.parent.name == "checkpoints":
            run_dir = cp.parents[1]
        else:
            run_dir = cp.parent
    else:
        run_dir = Path(f"runs/myopsnet_seed{args.seed}").resolve()

    if args.checkpoint is not None:
        checkpoint_path = Path(args.checkpoint).resolve()
    else:
        checkpoint_path = run_dir / "checkpoints" / f"myopsnet_seed{args.seed}.pt"

    if args.pred_dir is not None:
        pred_dir = Path(args.pred_dir).resolve()
    else:
        pred_dir = run_dir / "test_predictions"

    dataset = normalize_dataset_name(getattr(args, "dataset", "myops380"))

    if args.test_manifest is not None:
        manifest_path = Path(args.test_manifest).resolve()
    else:
        if args.split_dir is not None:
            split_p = Path(args.split_dir).resolve()
        else:
            if dataset == "myopspp":
                candidates = [
                    PROJECT_ROOT / "lists" / "lists_MyoPSpp",
                    PROJECT_ROOT / "preprocessing" / "splits" / "myopspp_bc80_no_val",
                    Path("D:/NCKH/SCAR/preprocessing/splits/myopspp_bc80_no_val"),
                ]
            else:
                candidates = [
                    PROJECT_ROOT / "preprocessing" / "splits" / "myops380_no_val",
                    Path("D:/NCKH/SCAR/preprocessing/splits/myops380_no_val"),
                ]
            split_p = next((c for c in candidates if c.is_dir()), candidates[0])

        if split_p.is_file():
            manifest_path = split_p
        else:
            manifest_path = split_p / f"{args.split}.txt"

    if args.data_root is not None:
        data_root = Path(args.data_root).resolve()
    else:
        if dataset == "myopspp":
            data_root = Path(os.environ.get("MYOPSPP_DATA_ROOT", "D:/NCKH/Dataset/MyoPSpp_UNet2D_ROI128_1mm")).resolve()
        else:
            data_root = Path(os.environ.get("MYOPS380_DATA_ROOT", "D:/NCKH/Dataset/MyoPS380/Processed_data")).resolve()

    return checkpoint_path, run_dir, pred_dir, manifest_path, data_root


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Executes standardized volume inference pipeline on MyoPS test cases."""
    parser = build_parser()
    args = parser.parse_args(argv)

    dataset = normalize_dataset_name(args.dataset)
    if dataset == "myopspp":
        training_protocol_id = "myopspp_train64_test16_last_v1"
        benchmark_protocol_id = "myopspp_unet2d_roi128_1mm_predicted_roi_mm_v1"
    else:
        training_protocol_id = TRAINING_PROTOCOL_ID
        benchmark_protocol_id = BENCHMARK_PROTOCOL_ID

    checkpoint_path, run_dir, pred_dir, manifest_path, data_root = resolve_paths(args)
    pred_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_inference_logger(pred_dir / "predict.log")

    logger.info("=" * 75)
    logger.info("MyoPS-Net Standardized Volume Inference Pipeline")
    logger.info("Dataset: %s | Protocol: %s | Benchmark: %s", dataset, training_protocol_id, benchmark_protocol_id)
    logger.info("Checkpoint: %s", checkpoint_path)
    logger.info("Manifest: %s", manifest_path)
    logger.info("Data Root: %s", data_root)
    logger.info("Prediction Output Dir: %s", pred_dir)
    logger.info("=" * 75)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Test manifest file not found: {manifest_path}")

    # 1. Device and Precision
    device = resolve_device(args.device)
    amp_mode = parse_amp_mode(args.amp)
    amp_dtype = resolve_amp_dtype(amp_mode, device)
    logger.info("Execution Device: %s | AMP: %s (dtype=%s) | Batch Size: %d", device, amp_mode, amp_dtype, args.batch_size)

    # 2. Instantiate Model and Load Checkpoint
    model = MyoPSNetL().to(device)
    logger.info("Loading model weights from checkpoint ...")
    checkpoint_payload = load_checkpoint(checkpoint_path, model=model, map_location=device)

    if args.require_complete:
        validate_checkpoint_payload(checkpoint_payload, require_complete=True)

    completed_epochs = checkpoint_payload.get("completed_epochs", checkpoint_payload.get("epoch", "unknown"))
    planned_epochs = checkpoint_payload.get("planned_epochs", 300)
    is_complete = checkpoint_payload.get("training_complete", False)
    logger.info(
        "Checkpoint loaded successfully: completed_epochs=%s/%s, training_complete=%s",
        completed_epochs, planned_epochs, is_complete
    )
    model.eval()

    # 3. Load Test Dataset
    test_dataset = MyoPSTestDataset(
        split_file=manifest_path,
        data_root=data_root,
        canonical_label=False,
        strict=args.strict,
        dataset=dataset,
    )
    all_cases = test_dataset.cases
    logger.info("Found %d test volumes in manifest", len(all_cases))

    # Handle case filtering (--cases or --dry_run)
    if args.cases is not None:
        target_cases = set(args.cases)
        selected_indices = [i for i, c in enumerate(all_cases) if c in target_cases]
        if not selected_indices:
            raise ValueError(f"None of the specified cases {args.cases} found in manifest")
    elif args.dry_run:
        selected_indices = list(range(min(2, len(all_cases))))
        logger.info("Dry-run mode activated: processing first %d cases", len(selected_indices))
    else:
        selected_indices = list(range(len(all_cases)))

    # 4. Volume Prediction Loop
    case_records = []
    total_slices = 0
    start_total_time = time.perf_counter()

    for idx, case_idx in enumerate(selected_indices):
        sample = test_dataset[case_idx]
        case_id = sample["case_id"]

        # Extract 3D volumes (H, W, D)
        bssfp_vol = sample["bSSFP"][0].numpy()
        lge_vol = sample["LGE"][0].numpy()
        t2w_vol = sample["T2w"][0].numpy()
        height, width, depth = bssfp_vol.shape

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()

        prediction = predict_volume(
            model=model,
            images=[bssfp_vol, lge_vol, t2w_vol],
            img_size=args.img_size,
            batch_size=args.batch_size,
            device=device,
            amp_dtype=amp_dtype,
        )

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - t0

        # Validate prediction format
        if prediction.shape != (height, width, depth):
            raise ValueError(f"{case_id}: prediction shape {prediction.shape} != input shape {(height, width, depth)}")
        if prediction.dtype != np.uint8:
            raise TypeError(f"{case_id}: expected uint8 prediction, got {prediction.dtype}")
        unique_vals = np.unique(prediction).tolist()
        if not set(unique_vals).issubset({0, 1, 2, 3}):
            raise ValueError(f"{case_id}: unexpected label values {unique_vals}, must be in [0..3]")

        # Save canonical .npz file
        npz_out = pred_dir / f"{case_id}_pred.npz"
        np.savez_compressed(
            npz_out,
            prediction=prediction,
            case_id=case_id,
            class_names=np.asarray(CLASS_NAMES),
            shape=np.asarray(prediction.shape),
            protocol_id=training_protocol_id,
            benchmark_protocol=benchmark_protocol_id,
            dataset=dataset,
            spacing=np.asarray(sample.get("spacing", [1.0, 1.0, 1.0])),
        )

        # Optional NIfTI export
        if args.save_nii:
            try:
                import nibabel as nib
                nii = nib.Nifti1Image(prediction, np.eye(4, dtype=np.float32))
                nib.save(nii, pred_dir / f"{case_id}_pred.nii.gz")
            except ImportError:
                logger.warning("nibabel not installed; skipping .nii.gz export")

        total_slices += depth
        case_records.append({
            "case_id": case_id,
            "shape": list(prediction.shape),
            "slices": depth,
            "seconds": round(elapsed, 4),
            "unique_labels": unique_vals,
            "output_npz": str(npz_out),
        })

        logger.info(
            "[%02d/%02d] %s | Shape: %s (%d slices) | Time: %.3fs (%.1f sl/s) | Labels: %s",
            idx + 1, len(selected_indices), case_id, prediction.shape, depth,
            elapsed, depth / max(elapsed, 1e-6), unique_vals
        )

    total_duration = time.perf_counter() - start_total_time
    throughput = total_slices / max(total_duration, 1e-6)

    logger.info("=" * 75)
    logger.info(
        "Inference Completed: %d volumes (%d slices) in %.2fs (%.2f slices/s)",
        len(selected_indices), total_slices, total_duration, throughput
    )
    logger.info("Predictions saved in: %s", pred_dir)
    logger.info("=" * 75)

    # 5. Export Execution Summary Metadata
    summary_record = {
        "dataset": dataset,
        "training_protocol_id": training_protocol_id,
        "benchmark_protocol_id": benchmark_protocol_id,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_epoch": completed_epochs,
        "planned_epochs": planned_epochs,
        "training_complete": is_complete,
        "model_contract": checkpoint_payload.get("model_contract"),
        "initialization": checkpoint_payload.get("initialization"),
        "loss_contract": checkpoint_payload.get("loss_contract"),
        "device": str(device),
        "amp_mode": amp_mode,
        "amp_dtype": str(amp_dtype),
        "batch_size": args.batch_size,
        "total_volumes": len(selected_indices),
        "total_slices": total_slices,
        "total_duration_seconds": round(total_duration, 4),
        "slices_per_second": round(throughput, 2),
        "cases": case_records,
    }
    with open(pred_dir / "inference_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_record, f, indent=2)

    return summary_record


if __name__ == "__main__":
    main()
