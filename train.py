"""Production Training Engine for MyoPS-Net on MyoPS380 Benchmark.

Strictly adheres to:
- Benchmark Protocol: myops380_train304_test76_last_v1
- Reference Benchmark: SCAR (D:\\nckh_model\\SCAR)
- Features: F1-F8, F11 (PROJECT.md)
  - F1: 304 train patients (1,972 slices) from train.txt, zero overlap with 76 test cases
  - F2: [0.0, 1.0] direct ingestion, 5-branch input with T1m/T2starm zero-padding
  - F3: SCAR RandomGenerator augmentation (128x128)
  - F4: Strict no-validation protocol (no val loader, no val loop, no best.pth)
  - F5: Target Adapter (canonical 4-class -> cardiac, scar, edema)
  - F6: Hardened MyoPSLoss (WCE zero-division protection)
  - F7: Grouped AdamW (wd=1e-4), step-level Polynomial LR Decay (power=0.9), seed=1098
  - F8: Effective batch size 16 with sample-weighted gradient accumulation, atomic last.pth
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from dataset.myops380_dataset import MyoPS380TrainDataset
from dataset.target_adapter import canonical_to_myops_targets
from network.criterion.loss import MyoPSLoss
from network.model import MyoPSNet
def weights_init(m: nn.Module) -> None:
    """Initializes Conv2d, BatchNorm2d, and Linear layer weights."""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            nn.init.constant_(m.weight, 1)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif classname.find('Linear') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            nn.init.xavier_uniform_(m.weight)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias, 0)


def set_seed(seed: int = 1098, deterministic: bool = True) -> None:
    """Sets random seeds across Python, NumPy, PyTorch, and CUDA for deterministic execution."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def seed_worker(worker_id: int) -> None:
    """Seeds DataLoader worker processes using PyTorch initial seed modulo 2**32."""
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def build_grouped_optimizer(model: nn.Module, base_lr: float = 1e-3, weight_decay: float = 1e-4) -> AdamW:
    """Standard grouped AdamW optimizer:
    - 2D/4D weights (Conv2d, Linear): weight_decay = 1e-4
    - 1D weights, biases, Normalization layers (BatchNorm, GroupNorm, LayerNorm): weight_decay = 0.0
    """
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() >= 2 and not any(norm_name in name.lower() for norm_name in ["norm", "bn", "gn", "ln"]):
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    # Invariant: zero overlap
    decay_ids = {id(p) for p in decay_params}
    no_decay_ids = {id(p) for p in no_decay_params}
    assert len(decay_ids.intersection(no_decay_ids)) == 0, "Parameter groups have overlapping parameters!"

    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    return AdamW(param_groups, lr=base_lr)


def get_polynomial_decay_scheduler(optimizer: AdamW, total_updates: int, power: float = 0.9) -> LambdaLR:
    """Step-level Polynomial LR Decay scheduler:
    lr = base_lr * (max(0.0, 1.0 - step / total_updates)) ** power
    """
    lr_lambda = lambda step: max(0.0, 1.0 - step / float(max(1, total_updates))) ** power
    return LambdaLR(optimizer, lr_lambda)


def atomic_save_checkpoint(payload: dict[str, Any], target_path: str | Path) -> None:
    """Atomically saves checkpoint using a temporary file and os.replace to prevent corruption."""
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target_path.with_suffix(target_path.suffix + ".tmp")
    torch.save(payload, temp_path)
    os.replace(temp_path, target_path)


def restore_rng_states(checkpoint: dict[str, Any], generator: torch.Generator | None = None) -> None:
    """Restores Python, NumPy, PyTorch, CUDA, and DataLoader generator RNG states from checkpoint."""
    # Python random state
    if "random_rng_state" in checkpoint and checkpoint["random_rng_state"] is not None:
        random.setstate(checkpoint["random_rng_state"])
    elif "rng" in checkpoint and isinstance(checkpoint["rng"], dict) and "python" in checkpoint["rng"]:
        if checkpoint["rng"]["python"] is not None:
            random.setstate(checkpoint["rng"]["python"])

    # NumPy random state
    if "numpy_rng_state" in checkpoint and checkpoint["numpy_rng_state"] is not None:
        np.random.set_state(checkpoint["numpy_rng_state"])
    elif "rng" in checkpoint and isinstance(checkpoint["rng"], dict) and "numpy" in checkpoint["rng"]:
        if checkpoint["rng"]["numpy"] is not None:
            np.random.set_state(checkpoint["rng"]["numpy"])

    # PyTorch RNG state
    if "torch_rng_state" in checkpoint and checkpoint["torch_rng_state"] is not None:
        torch.set_rng_state(checkpoint["torch_rng_state"])
    elif "rng" in checkpoint and isinstance(checkpoint["rng"], dict) and "torch" in checkpoint["rng"]:
        if checkpoint["rng"]["torch"] is not None:
            torch.set_rng_state(checkpoint["rng"]["torch"])

    # CUDA RNG state
    if torch.cuda.is_available():
        cuda_state = checkpoint.get("cuda_rng_state", None)
        if cuda_state is None and "rng" in checkpoint and isinstance(checkpoint["rng"], dict):
            cuda_state = checkpoint["rng"].get("cuda", None)
        if cuda_state is not None:
            try:
                torch.cuda.set_rng_state_all(cuda_state)
            except Exception:
                pass

    # DataLoader Generator state
    gen_state = checkpoint.get("generator_rng_state", checkpoint.get("loader_rng", None))
    if gen_state is None and "rng" in checkpoint and isinstance(checkpoint["rng"], dict):
        gen_state = checkpoint["rng"].get("generator", None)
    if gen_state is not None and generator is not None:
        try:
            generator.set_state(gen_state)
        except Exception:
            pass


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: AdamW,
    scheduler: LambdaLR,
    scaler: torch.cuda.amp.GradScaler | None,
    accum_steps: int,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    global_step: int,
    skipped_amp_steps: int,
    log_interval: int = 20,
) -> tuple[float, int, int]:
    """Executes one training epoch with sample-weighted gradient accumulation.

    Returns:
        (epoch_loss, global_step, skipped_amp_steps)
    """
    model.train()
    total_loss = 0.0
    total_samples = 0
    group_samples = 0
    accum_loss = 0.0

    optimizer.zero_grad(set_to_none=True)
    num_batches = len(train_loader)

    for batch_idx, batch in enumerate(train_loader):
        c0 = batch["C0"].to(device, non_blocking=True)
        lge = batch["LGE"].to(device, non_blocking=True)
        t2 = batch["T2"].to(device, non_blocking=True)
        t1m = batch["T1m"].to(device, non_blocking=True)
        t2starm = batch["T2starm"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)

        batch_count = label.shape[0]
        group_samples += batch_count
        total_samples += batch_count

        # Adapt canonical 4-class label to MyoPS multi-branch targets
        targets_dict = canonical_to_myops_targets(label)

        # Forward pass with AMP if scaler present
        use_amp = scaler is not None and device.type == "cuda"
        with torch.cuda.amp.autocast(enabled=use_amp):
            res_c0, res_lge, res_t2, res_mapping = model(c0, lge, t2, t1m, t2starm)
            seg_outputs = {
                "cardiac": res_c0,
                "scar": res_lge,
                "edema": res_t2,
                "mapping": res_mapping,
            }
            loss_seg, loss_inv, loss_inc, loss = criterion(seg_outputs, targets_dict)
            weighted_loss = loss * batch_count

        total_loss += loss.item() * batch_count
        accum_loss += loss.item() * batch_count

        # Backward pass
        if use_amp and scaler is not None:
            scaler.scale(weighted_loss).backward()
        else:
            weighted_loss.backward()

        # Check accumulation boundary
        boundary = ((batch_idx + 1) % accum_steps == 0) or ((batch_idx + 1) == num_batches)
        if boundary:
            if use_amp and scaler is not None:
                scaler.unscale_(optimizer)

            # Sample-weighted normalization across accumulated microbatches
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.div_(group_samples)

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Step optimizer & scheduler
            if use_amp and scaler is not None:
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                scale_after = scaler.get_scale()
                if scale_after >= scale_before:
                    scheduler.step()
                    global_step += 1
                else:
                    skipped_amp_steps += 1
            else:
                optimizer.step()
                scheduler.step()
                global_step += 1

            optimizer.zero_grad(set_to_none=True)
            group_samples = 0

        if (batch_idx + 1) % log_interval == 0 or (batch_idx + 1) == num_batches:
            current_lr = optimizer.param_groups[0]["lr"]
            avg_b_loss = total_loss / max(1, total_samples)
            print(
                f"Epoch [{epoch + 1:03d}/{total_epochs:03d}] | "
                f"Iter [{batch_idx + 1:03d}/{num_batches:03d}] | "
                f"LR: {current_lr:.6f} | "
                f"Loss: {avg_b_loss:.4f} | "
                f"Step: {global_step}"
            )

    epoch_avg_loss = total_loss / max(1, total_samples)
    return epoch_avg_loss, global_step, skipped_amp_steps


def train(args: argparse.Namespace) -> None:
    """Main training routine under no-validation protocol."""
    set_seed(args.seed)

    # Device configuration
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"Using device: {device} (CUDA available: {torch.cuda.is_available()})")

    # Paths
    data_root_path = getattr(args, "data_root", None) or getattr(args, "data_dir", None) or r"D:\nckh_model\Dataset\MyoPS380\Processed_data"
    data_root = Path(data_root_path)
    split_dir = Path(args.split_dir)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = save_dir / "last.pth"

    print("================================================================================")
    print("  MyoPS-Net Training Engine — Protocol: myops380_train304_test76_last_v1")
    print(f"  Data Root     : {data_root}")
    print(f"  Split Dir     : {split_dir}")
    print(f"  Output Dir    : {save_dir}")
    print(f"  Total Epochs  : {args.epochs}")
    print(f"  Micro-Batch   : {args.batch_size} | Accumulation Steps: {args.accum_steps} | Effective BS: {args.batch_size * args.accum_steps}")
    print(f"  Base LR       : {args.lr} | Weight Decay: {args.weight_decay}")
    print(f"  Seed          : {args.seed} | Validation: STRICTLY DISABLED")
    print("================================================================================")

    # 1. Dataset & DataLoader
    train_dataset = MyoPS380TrainDataset(
        data_dir=data_root,
        split_dir=split_dir,
        split="train",
        img_size=(args.dim, args.dim),
    )
    print(f"Loaded training dataset: {len(train_dataset)} slices from 304 patients.")

    # Generator for reproducible batch shuffling
    generator = torch.Generator()
    generator.manual_seed(args.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False,  # Keep all 1,972 slices
        generator=generator,
        worker_init_fn=seed_worker,
        pin_memory=(device.type == "cuda"),
    )

    # 2. Model definition
    # in_chs: C0(5), LGE(2), T2(2), mapping(3) -> 5 modalities with zero-padding
    model = MyoPSNet(in_chs=(5, 2, 2, 3), out_chs=(3, 3, 3, 3)).to(device)
    model.apply(weights_init)

    # 3. Loss & Optimizer
    criterion = MyoPSLoss().to(device)
    optimizer = build_grouped_optimizer(model, base_lr=args.lr, weight_decay=args.weight_decay)

    # 4. Total updates & Polynomial LR Decay
    updates_per_epoch = math.ceil(len(train_loader) / args.accum_steps)
    total_updates = args.epochs * updates_per_epoch
    scheduler = get_polynomial_decay_scheduler(optimizer, total_updates=total_updates, power=0.9)

    # 5. AMP GradScaler
    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and device.type == "cuda"))

    # 6. Resume handling
    start_epoch = 0
    global_step = 0
    skipped_amp_steps = 0

    if args.resume and checkpoint_path.is_file():
        print(f"Resuming from existing checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if scaler and "scaler_state_dict" in checkpoint and checkpoint["scaler_state_dict"]:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = checkpoint.get("completed_epochs", checkpoint.get("epoch", 0))
        global_step = checkpoint.get("global_step", 0)
        skipped_amp_steps = checkpoint.get("skipped_amp_steps", 0)
        restore_rng_states(checkpoint, generator=generator)
        print(f"Resumed at epoch {start_epoch}, global step {global_step} with RNG states restored.")

    if start_epoch >= args.epochs:
        print(f"Training already complete ({start_epoch}/{args.epochs} epochs). Checkpoint: {checkpoint_path}")
        return

    # Compute dataset manifest split hashes for provenance verification
    split_hashes: dict[str, str] = {}
    for manifest in ["train.txt", "test_vol.txt"]:
        mf_path = split_dir / manifest
        if mf_path.is_file():
            try:
                split_hashes[manifest.replace(".txt", "")] = hashlib.sha256(mf_path.read_bytes()).hexdigest()
            except Exception:
                pass

    # 7. Training loop
    start_time = time.time()
    for epoch in range(start_epoch, args.epochs):
        epoch_loss, global_step, skipped_amp_steps = train_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            accum_steps=args.accum_steps,
            device=device,
            epoch=epoch,
            total_epochs=args.epochs,
            global_step=global_step,
            skipped_amp_steps=skipped_amp_steps,
            log_interval=args.log_interval,
        )

        completed_epochs = epoch + 1
        is_final_epoch = (completed_epochs == args.epochs)

        # Atomic checkpoint save at end of each epoch (overwriting last.pth)
        gen_state = generator.get_state() if generator else None
        checkpoint_payload = {
            "epoch": epoch,
            "completed_epochs": completed_epochs,
            "planned_epochs": args.epochs,
            "training_complete": is_final_epoch,
            "checkpoint_selection": "last_completed_epoch",
            "checkpoint_selection_metric": None,
            "protocol": "myops380_train304_test76_last_v1",
            "format_version": 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler else None,
            "global_step": global_step,
            "total_updates": total_updates,
            "skipped_amp_steps": skipped_amp_steps,
            "epoch_loss": epoch_loss,
            "seed": args.seed,
            "base_lr": args.lr,
            "batch_size": args.batch_size,
            "accum_steps": args.accum_steps,
            "effective_batch_size": args.batch_size * args.accum_steps,
            "loss_identifier": "MyoPSLoss_Dice_WCE_Invariant_Inclusive",
            "data_root": str(data_root),
            "split_dir": str(split_dir),
            "split_hashes": split_hashes,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy_rng_state": np.random.get_state(),
            "random_rng_state": random.getstate(),
            "loader_rng": gen_state,
            "generator_rng_state": gen_state,
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "generator": gen_state,
            },
        }
        atomic_save_checkpoint(checkpoint_payload, checkpoint_path)

        if is_final_epoch:
            print("================================================================================")
            print(f"  Training officially COMPLETE! Reached final budget epoch {completed_epochs}/{args.epochs}.")
            print(f"  Official evaluation checkpoint saved to: {checkpoint_path}")
            print(f"  Total time elapsed: {time.time() - start_time:.2f}s")
            print("================================================================================")


def build_arg_parser() -> argparse.ArgumentParser:
    """Builds CLI argument parser for training."""
    parser = argparse.ArgumentParser(
        description="Train MyoPS-Net on MyoPS380 under no_val protocol (myops380_train304_test76_last_v1)."
    )
    parser.add_argument(
        "--data_root",
        "--data_dir",
        dest="data_root",
        type=str,
        default=r"D:\nckh_model\Dataset\MyoPS380\Processed_data",
        help="Path to preprocessed MyoPS380 dataset directory (alias: --data_dir).",
    )
    local_splits = Path(__file__).resolve().parent / "dataset" / "splits" / "myops380_no_val"
    default_split_dir = str(local_splits) if local_splits.is_dir() else r"D:\nckh_model\SCAR\preprocessing\splits\myops380_no_val"
    parser.add_argument(
        "--split_dir",
        type=str,
        default=default_split_dir,
        help="Path to split manifests directory (train.txt, test_vol.txt).",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=r"checkpoints",
        help="Directory to save checkpoint (last.pth).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=300,
        help="Total training epochs budget (fixed to 300).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Micro-batch size per iteration.",
    )
    parser.add_argument(
        "--accum_steps",
        type=int,
        default=4,
        help="Gradient accumulation steps (effective batch size = batch_size * accum_steps = 16).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Base learning rate for CNN backbone.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="Weight decay for AdamW (1e-4).",
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=128,
        help="Spatial dimension (128x128).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1098,
        help="Random seed for reproducibility (default: 1098).",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="Number of DataLoader worker processes.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to train on ('cuda' or 'cpu').",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        default=True,
        help="Enable Automatic Mixed Precision (AMP).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Resume from existing last.pth if present.",
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=20,
        help="Iterations between console logging.",
    )
    return parser


# Backward-compatible function aliases
MyoPSNetTrain = train
CrossModalSegNetTrain = train

__all__ = [
    "train",
    "MyoPSNetTrain",
    "CrossModalSegNetTrain",
    "build_arg_parser",
    "build_grouped_optimizer",
    "get_polynomial_decay_scheduler",
    "atomic_save_checkpoint",
    "restore_rng_states",
    "set_seed",
    "seed_worker",
    "weights_init",
]


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()
    train(args)
