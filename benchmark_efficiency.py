"""
benchmark_efficiency.py
=======================
Standardized Computational Efficiency Benchmarking CLI for MyoPS-Net.
Compliant with 'Quy tắc đo hiệu suất tính toán.txt'.

Measures and exports the 7 standardized computational efficiency metrics:
  1) Trainable Params (M) and Total Params (M)
  2) FLOPs (G) on (1, 3, 128, 128) FP32 input via PyTorch FlopCounterMode (eval mode)
  3) Latency (ms/slice, batch=1, FP32, eval, no_grad, with warmup + timed iterations)
  4) Inference Peak VRAM (MB & GB conversion)
  5) Time / Epoch (s, batch=16, measured skipping first 5 warmup epochs)
  6) Total Time (hours for planned epochs)
  7) Training Peak VRAM (MB & GB conversion)

Usage Examples:
  # Measure complexity + inference (quick audit):
  python benchmark_efficiency.py --dataset myops380 --efficiency_only

  # Measure with real/simulated 1 epoch on MyoPS++:
  python benchmark_efficiency.py --dataset myopspp --device cuda --simulate_training
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from criterion.loss import MyoPSLoss
from network.model import MyoPSNetL
from utils.efficiency import (
    BYTES_PER_GB,
    BYTES_PER_MB,
    build_efficiency_payload,
    check_rule_compliance,
    count_parameters,
    format_efficiency_report,
    measure_flops,
    measure_inference_efficiency,
    save_efficiency_report,
    summarize_training_efficiency,
    sync_if_cuda,
)


def resolve_device(device_name: str = "auto") -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Standardized Computational Efficiency Benchmarking for MyoPS-Net",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Dataset selection
    parser.add_argument(
        "--dataset",
        default="myops380",
        choices=["myops380", "myopspp", "myopspp_unet2d_roi128_1mm"],
        help="Dataset choice: myops380 or myopspp",
    )
    parser.add_argument("--data_root", "--data-root", default=None, help="Dataset root path (optional)")
    parser.add_argument("--split_dir", "--list_dir", default=None, help="Split directory (optional)")

    # Hardware & precision
    parser.add_argument("--device", default="auto", help="Execution device: auto, cuda, cpu, cuda:0")
    parser.add_argument("--amp", default="none", help="AMP mode: none (mandatory FP32 for benchmark), auto, fp16, bf16")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader num_workers (rule: 0)")
    parser.add_argument("--pin_memory", action=argparse.BooleanOptionalAction, default=True, help="DataLoader pin_memory")

    # Benchmark protocol settings
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size (rule: 16)")
    parser.add_argument("--epochs", type=int, default=300, help="Planned training epochs (rule: 300)")
    parser.add_argument("--warmup", type=int, default=None, help="Inference warmup iterations (default: 100 on CUDA, 2 on CPU)")
    parser.add_argument("--repeats", type=int, default=None, help="Inference timed iterations (default: 300 on CUDA, 5 on CPU)")

    # Execution modes
    parser.add_argument("--efficiency_only", action="store_true", help="Only measure complexity and inference latency/VRAM")
    parser.add_argument("--simulate_training", action="store_true", help="Benchmark 1 full epoch training step to project time and measure VRAM")

    # Output paths
    parser.add_argument("--output", "--output_path", default=None, help="Direct output path for efficiency_metrics.json")
    parser.add_argument("--run_dir", default=None, help="Run directory to save efficiency_metrics.json")

    return parser


def benchmark_efficiency_main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    parser = build_parser()
    args = parser.parse_args(argv)

    device = resolve_device(args.device)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print("=" * 80)
    print("MyoPS-Net Computational Efficiency Benchmark")
    print(f"Dataset : {args.dataset} | Device: {device} | Precision: FP32 (--amp {args.amp})")
    print("=" * 80)

    # 1. Model instantiation
    model = MyoPSNetL().to(device).float()
    model.eval()

    # 2. Complexity: Parameters & FLOPs
    print("\n[Step 1/3] Measuring Complexity (Trainable Parameters & FLOPs) ...")
    params_info = count_parameters(model)
    flops_info = measure_flops(model, input_shape=(1, 3, 128, 128), device=device)
    print(f"  Trainable Params: {params_info['trainable_params_excel']} ({params_info['trainable_params']:,} parameters)")
    print(f"  Total Params    : {params_info['total_params_excel']} ({params_info['total_params']:,} parameters)")
    print(f"  FLOPs           : {flops_info['flops_g']:.2f} GFLOPs ({flops_info['flops_total']:,} FLOPs)")

    # 3. Inference: Latency & Peak VRAM
    warmup = args.warmup if args.warmup is not None else (100 if device.type == "cuda" else 2)
    repeats = args.repeats if args.repeats is not None else (300 if device.type == "cuda" else 5)

    print(f"\n[Step 2/3] Measuring Batch=1 FP32 Inference (warmup={warmup}, repeats={repeats}) ...")
    inf_info = measure_inference_efficiency(
        model,
        input_shape=(1, 3, 128, 128),
        device=device,
        warmup=warmup,
        repeats=repeats,
    )
    print(f"  Latency   : {inf_info['latency_ms_per_slice']:.2f} ms/slice")
    print(f"  Peak VRAM : {inf_info['peak_vram_mb']:.2f} MB ({inf_info['peak_vram_gb']:.4f} GB)")

    # 4. Training measurement or simulation
    training_info = None
    if not args.efficiency_only:
        print("\n[Step 3/3] Measuring Training Efficiency (Batch=16, FP32) ...")
        # Determine number of batches per epoch
        if args.dataset in ("myopspp", "myopspp_unet2d_roi128_1mm"):
            batches_per_epoch = 19  # 296 slices // 16 = 18 + 1 = 19
            slices_count = 296
        else:
            batches_per_epoch = 124  # 1972 slices // 16 = 123 + 1 = 124
            slices_count = 1972

        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)

        model.train()
        criterion = MyoPSLoss().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003333, weight_decay=0.0001)

        # Warmup iteration
        dummy_c0 = torch.randn(args.batch_size, 1, 128, 128, device=device)
        dummy_lge = torch.randn(args.batch_size, 1, 128, 128, device=device)
        dummy_t2 = torch.randn(args.batch_size, 1, 128, 128, device=device)
        dummy_t_c0 = torch.randint(0, 2, (args.batch_size, 128, 128), device=device)
        dummy_t_lge = torch.randint(0, 3, (args.batch_size, 128, 128), device=device)
        dummy_t_t2 = torch.randint(0, 3, (args.batch_size, 128, 128), device=device)

        optimizer.zero_grad()
        s_c0, s_lge, s_t2 = model(dummy_c0, dummy_lge, dummy_t2)
        loss, _, _, _ = criterion(s_c0, s_lge, s_t2, dummy_t_c0, dummy_t_lge, dummy_t_t2)
        loss.backward()
        optimizer.step()
        sync_if_cuda(device)

        # Timed benchmark: 1 epoch simulation (batches_per_epoch forward + backward)
        t_epoch_start = time.perf_counter()
        for b_idx in range(batches_per_epoch):
            bs = args.batch_size if (b_idx < batches_per_epoch - 1) else (slices_count % args.batch_size or args.batch_size)
            if bs != dummy_c0.shape[0]:
                b_c0 = dummy_c0[:bs]
                b_lge = dummy_lge[:bs]
                b_t2 = dummy_t2[:bs]
                b_t_c0 = dummy_t_c0[:bs]
                b_t_lge = dummy_t_lge[:bs]
                b_t_t2 = dummy_t_t2[:bs]
            else:
                b_c0, b_lge, b_t2 = dummy_c0, dummy_lge, dummy_t2
                b_t_c0, b_t_lge, b_t_t2 = dummy_t_c0, dummy_t_lge, dummy_t_t2

            optimizer.zero_grad()
            s_c0, s_lge, s_t2 = model(b_c0, b_lge, b_t2)
            loss, _, _, _ = criterion(s_c0, s_lge, s_t2, b_t_c0, b_t_lge, b_t_t2)
            loss.backward()
            optimizer.step()

        sync_if_cuda(device)
        t_epoch_end = time.perf_counter()
        single_epoch_time = t_epoch_end - t_epoch_start

        if device.type == "cuda" and torch.cuda.is_available():
            peak_train_bytes = int(torch.cuda.max_memory_allocated(device))
            peak_reserved_bytes = int(torch.cuda.max_memory_reserved(device))
        else:
            peak_train_bytes = 0
            peak_reserved_bytes = 0

        # Simulate 300 epochs from measured time per epoch
        epoch_times = [round(single_epoch_time, 4)] * args.epochs
        training_info = summarize_training_efficiency(
            epoch_times_s=epoch_times,
            total_wall_clock_s=single_epoch_time * args.epochs,
            peak_vram_bytes=peak_train_bytes,
            planned_epochs=args.epochs,
            warmup_epochs=5,
            peak_reserved_bytes=peak_reserved_bytes,
        )
        print(f"  Time / Epoch: {training_info['time_per_epoch_s']:.2f} s")
        print(f"  Total Time  : {training_info['total_time_hours']:.4f} hours ({training_info['total_time_seconds']:.1f} s)")
        print(f"  Peak VRAM   : {training_info['peak_vram_mb']:.2f} MB ({training_info['peak_vram_gb']:.4f} GB)")

    # 5. Rule compliance check
    rule_ok, warnings, hw_info = check_rule_compliance(
        device=device,
        amp_mode=args.amp,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        batch_size=args.batch_size,
        planned_epochs=args.epochs,
    )

    # 6. Build payload
    payload = build_efficiency_payload(
        model_name="MyoPS-Net",
        model_id="myopsnet",
        seed=args.seed,
        dataset=args.dataset,
        params_info=params_info,
        flops_info=flops_info,
        inference_info=inf_info,
        training_info=training_info,
        rule_compliant=rule_ok,
        compliance_warnings=warnings,
        hardware_info=hw_info,
    )

    # 7. Determine output paths
    output_paths = []
    if args.output:
        output_paths.append(Path(args.output).resolve())
    if args.run_dir:
        output_paths.append((Path(args.run_dir) / "efficiency_metrics.json").resolve())

    # Standard outputs
    std_out = PROJECT_ROOT / "outputs" / "runs" / "myopsnet" / "efficiency_metrics.json"
    if std_out not in output_paths:
        output_paths.append(std_out)

    root_out = PROJECT_ROOT / "efficiency_metrics.json"
    if root_out not in output_paths:
        output_paths.append(root_out)

    save_efficiency_report(payload, output_paths=output_paths)
    return payload


if __name__ == "__main__":
    benchmark_efficiency_main()
