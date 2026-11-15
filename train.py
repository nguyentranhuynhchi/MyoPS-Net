"""
train.py
========
Standardized 300-Epoch Training Pipeline for MyoPS-Net Baseline.
Protocol: myops380_train304_test76_last_v1
Benchmark: MyoPS380 (304 Train / 76 Test / 300 Epochs / Strict NO Validation)
"""

from __future__ import annotations
import argparse
import csv
import json
import logging
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from criterion.loss import MyoPSLoss
from dataset.augmentation import RandomGenerator
from dataset.myops380_dataset import (
    create_myops380_train_loader,
    create_myops_train_loader,
    normalize_dataset_name,
)
from network.model import MyoPSNetL
from training.checkpoint import (
    atomic_save_checkpoint,
    capture_rng_state,
    load_checkpoint,
    restore_rng_state,
    resume_checkpoint,
    resume_training_state,
    safe_atomic_save,
    save_checkpoint,
    validate_checkpoint_payload,
)
from training.optimizer import (
    OPTIMIZATION_PROTOCOL,
    build_optimizer,
    build_scheduler,
    calculate_total_updates,
    compute_gradient_norm,
)
from training.provenance import (
    INITIALIZATION, LOSS_CONTRACT, MODEL_CONTRACT, RNG_POLICY, build_resume_contract,
)
from utils.efficiency import (
    build_efficiency_payload,
    check_rule_compliance,
    count_parameters,
    measure_flops,
    measure_inference_efficiency,
    save_efficiency_report,
    summarize_training_efficiency,
    sync_if_cuda,
)
from utils.tools import create_multi_head_targets, weights_init


def setup_logger(log_file: Path) -> logging.Logger:
    """Configure console and file logging."""
    logger = logging.getLogger("myopsnet.train")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_h = logging.FileHandler(log_file, encoding="utf-8")
    file_h.setFormatter(formatter)
    stream_h = logging.StreamHandler(sys.stdout)
    stream_h.setFormatter(formatter)
    logger.addHandler(file_h)
    logger.addHandler(stream_h)
    return logger


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Set deterministic random seed across all RNG backends."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic


def resolve_device(device_name: str = "auto") -> torch.device:
    """Resolve hardware execution device."""
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def parse_amp_mode(amp_arg: Union[str, bool]) -> str:
    """Normalize AMP string or boolean argument."""
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
    """Resolve AMP autocast floating point dtype."""
    if amp_mode == "none" or device.type == "cpu":
        return None
    if amp_mode == "auto":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if amp_mode == "fp16":
        return torch.float16
    if amp_mode == "bf16":
        if not torch.cuda.is_bf16_supported():
            return torch.float16
        return torch.bfloat16
    return None


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for training pipeline."""
    parser = argparse.ArgumentParser(
        description="Standardized Training Pipeline for MyoPS-Net",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Dataset selection
    parser.add_argument(
        "--dataset",
        default="myops380",
        choices=["myops380", "myopspp", "myopspp_unet2d_roi128_1mm"],
        help="Dataset choice: myops380 or myopspp",
    )
    # Data paths
    parser.add_argument("--data_root", "--data-root", default=None, help="Dataset root directory")
    parser.add_argument("--split_dir", "--list_dir", default=None, help="Split manifest directory")
    parser.add_argument("--run_dir", "--output_dir", default=None)

    # Training recipe
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--epochs", "--max_epochs", type=int, default=300)
    parser.add_argument("--batch_size", "--batch-size", type=int, default=16)
    parser.add_argument("--base_lr", "--lr", type=float, default=0.0003333)
    parser.add_argument("--weight_decay", "--weight-decay", type=float, default=0.0001)
    parser.add_argument("--clip_grad", type=float, default=0.0)
    parser.add_argument("--accum_steps", type=int, default=1)

    # Hardware & precision
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", default="auto", help="AMP mode: auto, fp16, bf16, none, or boolean True/False")
    parser.add_argument("--no-amp", dest="amp", action="store_const", const="none", help="Disable AMP")
    parser.add_argument("--num_workers", "--num-workers", type=int, default=4)
    parser.add_argument("--pin_memory", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)

    # Efficiency benchmarking
    parser.add_argument("--measure_efficiency", action="store_true", help="Measure and export 7 computational efficiency metrics")
    parser.add_argument("--efficiency_output", default=None, help="Output path for efficiency_metrics.json")
    parser.add_argument("--efficiency_only", action="store_true", help="Only measure complexity and inference, without full training")

    # Checkpoint & logging
    parser.add_argument("--resume", default=None, help="Path to checkpoint file or 'auto'")
    parser.add_argument("--save_interval", type=int, default=1, help="Periodic checkpoint save frequency (epochs)")
    parser.add_argument("--epochs_per_run", "--epochs-per-run", type=int, default=0,
                        help="Pause after this many epochs without changing the planned LR budget (0 = finish)")
    parser.add_argument("--log_interval", type=int, default=20, help="Batch logging frequency")
    parser.add_argument("--tensorboard", action=argparse.BooleanOptionalAction, default=True)

    # Fast verification / testing
    parser.add_argument("--strict", action=argparse.BooleanOptionalAction, default=False, help="Strictly enforce expected slice/patient counts")
    parser.add_argument("--dry_run", "--smoke_test", action="store_true", help="Run 1 epoch of 2 batches for fast CI check")
    parser.add_argument("--max_batches", type=int, default=0, help="Max batches per epoch (0 = unlimited)")

    return parser


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Main execution function for MyoPS-Net standardized training."""
    parser = build_parser()
    args = parser.parse_args(argv)
    for key in ("epochs", "batch_size", "save_interval", "log_interval"):
        if getattr(args, key) < 1:
            parser.error(f"--{key} must be positive")
    if args.num_workers < 0 or args.epochs_per_run < 0 or args.max_batches < 0:
        parser.error("num_workers, epochs_per_run and max_batches must be nonnegative")
    if args.accum_steps != 1:
        parser.error("This multi-head training recipe requires --accum_steps 1")
    resuming = args.resume is not None and str(args.resume).strip() != ""

    dataset = normalize_dataset_name(args.dataset)

    # Protocol & Label settings
    if dataset == "myopspp":
        protocol_id = "myopspp_train64_test16_last_v1"
        target_mode = "canonical"
    else:
        protocol_id = "myops380_train304_test76_last_v1"
        target_mode = "legacy"

    # Resolve Data paths
    if args.data_root is not None:
        data_root = Path(args.data_root)
    else:
        if dataset == "myopspp":
            data_root = Path(os.environ.get("MYOPSPP_DATA_ROOT", "D:/NCKH/Dataset/MyoPSpp_UNet2D_ROI128_1mm"))
        else:
            data_root = Path(os.environ.get("MYOPS380_DATA_ROOT", "D:/NCKH/Dataset/MyoPS380/Processed_data"))

    if args.split_dir is not None:
        split_dir = Path(args.split_dir)
    else:
        if dataset == "myopspp":
            candidates = [
                PROJECT_ROOT / "lists" / "lists_MyoPSpp",
                PROJECT_ROOT / "preprocessing" / "splits" / "myopspp_bc80_no_val",
                Path("D:/NCKH/SCAR/preprocessing/splits/myopspp_bc80_no_val"),
            ]
            split_dir = next((c for c in candidates if c.is_dir()), candidates[0])
        else:
            candidates = [
                PROJECT_ROOT / "preprocessing" / "splits" / "myops380_no_val",
                Path("D:/NCKH/SCAR/preprocessing/splits/myops380_no_val"),
            ]
            split_dir = next((c for c in candidates if c.is_dir()), candidates[0])

    args.data_root = str(data_root)
    args.split_dir = str(split_dir)
    args.dataset = dataset

    # 1. Output directory setup
    if args.run_dir is None:
        run_dir = Path("runs") / f"myopsnet_seed{args.seed}"
    else:
        run_dir = Path(args.run_dir)
    run_dir = run_dir.resolve()
    if not resuming and run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError("Run directory is not empty; choose a new --run_dir or use --resume")

    ckpt_dir = run_dir / "checkpoints"
    log_dir = run_dir / "logs"
    tb_dir = run_dir / "tensorboard"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(log_dir / "train.log")
    checkpoint_path = ckpt_dir / f"myopsnet_seed{args.seed}.pt"

    # 2. Hardware and seeding
    seed_everything(args.seed, deterministic=args.deterministic)
    device = resolve_device(args.device)
    amp_mode = parse_amp_mode(args.amp)
    amp_dtype = resolve_amp_dtype(amp_mode, device)
    use_amp = (amp_dtype is not None)
    pin_memory = (device.type == "cuda") if args.pin_memory is None else args.pin_memory

    logger.info("=" * 70)
    logger.info("MyoPS-Net Standardized Training Pipeline")
    logger.info("Dataset: %s | Protocol: %s | Opt Protocol: %s", dataset, protocol_id, OPTIMIZATION_PROTOCOL["id"])
    logger.info("Target Mode: %s | Split dir: %s", target_mode, split_dir)
    logger.info("Run directory: %s", run_dir)
    logger.info("Device: %s | AMP: %s (%s)", device, amp_mode, amp_dtype)
    logger.info("Seed: %d | Batch size: %d | Planned epochs: %d", args.seed, args.batch_size, args.epochs)
    logger.info("=" * 70)

    # 3. DataLoader creation
    split_file = split_dir / "train.txt"
    transform = RandomGenerator(output_size=[128, 128])

    loader_generator = torch.Generator()
    loader_generator.manual_seed(args.seed)

    train_loader = create_myops_train_loader(
        split_file=split_file,
        data_root=data_root,
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        transform=transform,
        strict=args.strict,
        generator=loader_generator,
    )
    batches_per_epoch = len(train_loader)
    logger.info("Loaded training data: %d slices across %d batches per epoch", len(train_loader.dataset), batches_per_epoch)
    logger.info("Fingerprinting training cache and source for resume compatibility ...")
    resume_contract = build_resume_contract(
        args, train_loader.dataset, device=device, amp_dtype=amp_dtype, pin_memory=pin_memory,
    )

    # 4. Model, Criterion, Optimizer, Scheduler, Scaler
    model = MyoPSNetL().to(device)
    if not resuming:
        model.apply(weights_init)
    criterion = MyoPSLoss().to(device)
    optimizer = build_optimizer(model, base_lr=args.base_lr, weight_decay=args.weight_decay)
    total_updates = calculate_total_updates(args.epochs, batches_per_epoch, args.accum_steps)
    scheduler = build_scheduler(optimizer, total_updates=total_updates)
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and amp_dtype == torch.float16))

    # Efficiency benchmarking initialization (if requested)
    eff_params_info = None
    eff_flops_info = None
    eff_inf_info = None
    eff_rule_ok = False
    eff_warnings: list[str] = []
    eff_hw_info: dict[str, Any] = {}
    epoch_times_measured: list[float] = []

    if args.measure_efficiency or args.efficiency_only:
        logger.info("Running complexity and inference efficiency measurements ...")
        eff_params_info = count_parameters(model)
        eff_flops_info = measure_flops(model, input_shape=(1, 3, 128, 128), device=device)
        inf_warmup = 100 if device.type == "cuda" else 2
        inf_repeats = 300 if device.type == "cuda" else 5
        eff_inf_info = measure_inference_efficiency(
            model, input_shape=(1, 3, 128, 128), device=device, warmup=inf_warmup, repeats=inf_repeats
        )
        eff_rule_ok, eff_warnings, eff_hw_info = check_rule_compliance(
            device=device,
            amp_mode=amp_mode,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
            batch_size=args.batch_size,
            planned_epochs=args.epochs,
        )
        if args.efficiency_only:
            payload = build_efficiency_payload(
                model_name="MyoPS-Net",
                model_id="myopsnet",
                seed=args.seed,
                dataset=dataset,
                params_info=eff_params_info,
                flops_info=eff_flops_info,
                inference_info=eff_inf_info,
                training_info=None,
                rule_compliant=eff_rule_ok,
                compliance_warnings=eff_warnings,
                hardware_info=eff_hw_info,
            )
            eff_paths = [log_dir / "efficiency_metrics.json", run_dir / "efficiency_metrics.json"]
            if args.efficiency_output:
                eff_paths.append(Path(args.efficiency_output).resolve())
            std_out = PROJECT_ROOT / "outputs" / "runs" / "myopsnet" / "efficiency_metrics.json"
            if std_out not in eff_paths:
                eff_paths.append(std_out)
            save_efficiency_report(payload, eff_paths, summary_text_paths=[log_dir / "efficiency_summary.txt"], log=logger)
            return {"training_complete": False, "efficiency_metrics": payload}

    # 5. Handle Resumption
    start_epoch = 0
    completed_epochs = 0
    global_step = 0
    amp_skipped_updates = 0

    if resuming:
        resume_target = checkpoint_path if str(args.resume).strip() in ("auto", "True", "true") else Path(args.resume)
        if resume_target.is_file():
            logger.info("Resuming training state from %s ...", resume_target)
            res = resume_training_state(
                checkpoint_path=resume_target,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                loader_generator=loader_generator,
                device=device,
                expected_resume_contract=resume_contract,
            )
            completed_epochs = res["completed_epochs"]
            start_epoch = res["start_epoch"]
            global_step = res["global_step"]
            amp_skipped_updates = res["amp_skipped_updates"]
            logger.info(
                "Resumed state: completed_epochs=%d (next epoch=%d), global_step=%d, status=%s",
                completed_epochs, start_epoch + 1, global_step, res["status"]
            )
            if res["status"] == "already_completed" or completed_epochs >= args.epochs:
                logger.info("Training already complete (%d/%d epochs). Exiting cleanly.", completed_epochs, args.epochs)
                return {
                    "completed_epochs": completed_epochs,
                    "global_step": global_step,
                    "training_complete": True,
                    "checkpoint_path": str(resume_target),
                }
        else:
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_target}")

    # Write config metadata
    model_contract = dict(MODEL_CONTRACT)
    if dataset == "myopspp":
        model_contract["source_label_order"] = "canonical_background_normal_edema_scar"
    config_record = {
        "dataset": dataset,
        "protocol_id": protocol_id,
        "optimization_protocol": OPTIMIZATION_PROTOCOL,
        "args": vars(args),
        "total_updates": total_updates,
        "model_parameters": sum(p.numel() for p in model.parameters()),
        "device": str(device),
        "amp_mode": amp_mode,
        "amp_dtype": str(amp_dtype),
        "model_contract": model_contract,
        "initialization": INITIALIZATION,
        "loss_contract": LOSS_CONTRACT,
        "rng_policy": RNG_POLICY,
        "resume_contract": resume_contract,
    }
    with open(log_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config_record, f, indent=2)

    # TensorBoard initialization
    writer = None
    if args.tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(log_dir=str(tb_dir / "epochs"))
        except ImportError:
            logger.warning("TensorBoard not installed. Telemetry written to CSV and JSONL only.")

    target_epochs = 1 if args.dry_run else args.epochs
    if args.epochs_per_run:
        target_epochs = min(target_epochs, start_epoch + args.epochs_per_run)

    # 6. Training Loop (Strict NO Validation)
    logger.info("Starting training loop: epoch %d -> %d (target nominal: %d) ...", start_epoch + 1, target_epochs, args.epochs)

    for epoch in range(start_epoch, target_epochs):
        model.train()
        epoch_start_time = time.perf_counter()
        running_loss_total = 0.0
        running_loss_seg = 0.0
        running_loss_inv = 0.0
        running_loss_inc = 0.0
        sample_count = 0

        for batch_idx, batch in enumerate(train_loader):
            if args.dry_run and batch_idx >= 2:
                break
            if args.max_batches > 0 and batch_idx >= args.max_batches:
                break

            bSSFP = batch["bSSFP"].to(device, non_blocking=pin_memory)
            LGE   = batch["LGE"].to(device, non_blocking=pin_memory)
            T2w   = batch["T2w"].to(device, non_blocking=pin_memory)
            label = batch["label"].to(device, non_blocking=pin_memory)
            B = label.shape[0]

            target_C0, target_LGE, target_T2 = create_multi_head_targets(label, mode=target_mode)

            amp_enabled = (use_amp and device.type == "cuda")
            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                seg_C0, seg_LGE, seg_T2 = model(bSSFP, LGE, T2w)
                loss_total, loss_seg, loss_inv, loss_inc = criterion(
                    seg_C0, seg_LGE, seg_T2, target_C0, target_LGE, target_T2
                )

            if not torch.isfinite(loss_total):
                raise FloatingPointError(
                    f"Non-finite loss detected at epoch {epoch + 1}, batch {batch_idx + 1}: {loss_total.item()}"
                )

            scaler.scale(loss_total).backward()

            is_boundary = ((batch_idx + 1) % args.accum_steps == 0) or ((batch_idx + 1) == batches_per_epoch)
            if is_boundary:
                scaler.unscale_(optimizer)
                norm = compute_gradient_norm(
                    model,
                    clip_grad=args.clip_grad,
                    error_if_nonfinite=not scaler.is_enabled(),
                )

                old_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()

                skipped = (scaler.get_scale() < old_scale)
                if not skipped:
                    scheduler.step()
                    global_step += 1
                else:
                    amp_skipped_updates += 1
                    logger.warning(
                        "Epoch [%03d/%03d] Batch [%03d/%03d] | Non-finite gradient norm (%.4f) detected with AMP scaler enabled; step skipped (scale reduced: %.1f -> %.1f, total skipped: %d).",
                        epoch + 1, target_epochs, batch_idx + 1, batches_per_epoch,
                        norm, old_scale, scaler.get_scale(), amp_skipped_updates,
                    )

                optimizer.zero_grad(set_to_none=True)

            running_loss_total += loss_total.item() * B
            running_loss_seg += loss_seg.item() * B
            running_loss_inv += loss_inv.item() * B
            running_loss_inc += loss_inc.item() * B
            sample_count += B

            if (batch_idx + 1) % args.log_interval == 0 or (batch_idx + 1) == batches_per_epoch or args.dry_run:
                current_lr = optimizer.param_groups[0]["lr"]
                logger.info(
                    "Epoch [%03d/%03d] Batch [%03d/%03d] | Total Loss: %.5f (Seg: %.4f, Inv: %.4f, Inc: %.4f) | LR: %.7f | Step: %d",
                    epoch + 1, target_epochs, batch_idx + 1, batches_per_epoch,
                    loss_total.item(), loss_seg.item(), loss_inv.item(), loss_inc.item(),
                    current_lr, global_step
                )

        sync_if_cuda(device)
        epoch_duration = time.perf_counter() - epoch_start_time
        epoch_times_measured.append(epoch_duration)
        completed_epochs = epoch + 1
        mean_loss = running_loss_total / max(sample_count, 1)
        mean_seg = running_loss_seg / max(sample_count, 1)
        mean_inv = running_loss_inv / max(sample_count, 1)
        mean_inc = running_loss_inc / max(sample_count, 1)
        current_lr = optimizer.param_groups[0]["lr"]

        logger.info(
            "--> Epoch [%03d/%03d] Finished in %.2fs | Mean Loss: %.5f | LR: %.7f",
            completed_epochs, target_epochs, epoch_duration, mean_loss, current_lr
        )

        # Also save at a session boundary so epochs_per_run can resume exactly.
        is_save_epoch = (
            (completed_epochs % args.save_interval == 0)
            or (completed_epochs == args.epochs)
            or (args.dry_run and completed_epochs == 1)
            or (completed_epochs == target_epochs)
        )
        if is_save_epoch:
            logger.info("Saving atomic checkpoint to %s (epoch %d) ...", checkpoint_path, completed_epochs)
            save_checkpoint(
                checkpoint_path=checkpoint_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=completed_epochs,
                completed_epochs=completed_epochs,
                planned_epochs=args.epochs,
                global_step=global_step,
                total_updates=total_updates,
                seed=args.seed,
                args=args,
                amp_skipped_updates=amp_skipped_updates,
                loader_generator=loader_generator,
                resume_contract=resume_contract,
                model_contract=model_contract,
                initialization=INITIALIZATION,
                loss_contract=LOSS_CONTRACT,
                rng_policy=RNG_POLICY,
                protocol_id=protocol_id,
                dataset=dataset,
            )

        # Record tabular metrics
        record = {
            "epoch": completed_epochs,
            "global_step": global_step,
            "train/loss": mean_loss,
            "train/loss_seg": mean_seg,
            "train/loss_inv": mean_inv,
            "train/loss_inc": mean_inc,
            "train/lr": current_lr,
            "epoch_seconds": epoch_duration,
            "amp_skipped_updates": amp_skipped_updates,
            "training_complete": (completed_epochs >= args.epochs),
        }

        with open(log_dir / "metrics.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        csv_path = log_dir / "metrics.csv"
        csv_exists = csv_path.is_file() and csv_path.stat().st_size > 0
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer_csv = csv.DictWriter(f, fieldnames=list(record.keys()))
            if not csv_exists:
                writer_csv.writeheader()
            writer_csv.writerow(record)

        if writer:
            for k, v in record.items():
                if isinstance(v, (int, float)):
                    writer.add_scalar(k, v, completed_epochs)

    if writer:
        writer.close()

    is_complete = (completed_epochs >= args.epochs)
    logger.info("Training session finished. Checkpoint: %s (completed: %s)", checkpoint_path, is_complete)

    # 7. Finalize and export efficiency metrics if requested
    if args.measure_efficiency:
        logger.info("Finalizing computational efficiency metrics report ...")
        peak_bytes = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        peak_reserved = int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0
        training_info = summarize_training_efficiency(
            epoch_times_s=epoch_times_measured,
            total_wall_clock_s=sum(epoch_times_measured),
            peak_vram_bytes=peak_bytes,
            planned_epochs=args.epochs,
            warmup_epochs=5,
            peak_reserved_bytes=peak_reserved,
        )
        eff_payload = build_efficiency_payload(
            model_name="MyoPS-Net",
            model_id="myopsnet",
            seed=args.seed,
            dataset=dataset,
            params_info=eff_params_info,
            flops_info=eff_flops_info,
            inference_info=eff_inf_info,
            training_info=training_info,
            rule_compliant=eff_rule_ok,
            compliance_warnings=eff_warnings,
            hardware_info=eff_hw_info,
        )
        eff_paths = [log_dir / "efficiency_metrics.json", run_dir / "efficiency_metrics.json"]
        if args.efficiency_output:
            eff_paths.append(Path(args.efficiency_output).resolve())
        std_out = PROJECT_ROOT / "outputs" / "runs" / "myopsnet" / "efficiency_metrics.json"
        if std_out not in eff_paths:
            eff_paths.append(std_out)
        save_efficiency_report(
            eff_payload,
            eff_paths,
            summary_text_paths=[log_dir / "efficiency_summary.txt", run_dir / "efficiency_summary.txt"],
            log=logger,
        )

    return {
        "completed_epochs": completed_epochs,
        "global_step": global_step,
        "training_complete": is_complete,
        "checkpoint_path": str(checkpoint_path),
    }


if __name__ == "__main__":
    main()
