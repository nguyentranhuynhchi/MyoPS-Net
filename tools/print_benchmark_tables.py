"""Utility to print standardized benchmark tables (Efficiency & Segmentation metrics)
Compliant with MyoPS Benchmark Protocol and Quy tắc đo hiệu suất tính toán.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def print_tables(run_dir: Path) -> None:
    run_dir = Path(run_dir).resolve()
    eff_file = run_dir / "efficiency_metrics.json"
    metric_file = run_dir / "evaluation" / "metrics.json"

    # 1. Print Efficiency Table
    if eff_file.exists():
        with open(eff_file, "r", encoding="utf-8") as f:
            eff_data = json.load(f)
        print("\n" + "=" * 80)
        print("=== BẢNG 1: 7 CHỈ SỐ HIỆU NĂNG TÍNH TOÁN (A100 FP32) ===")
        print("=" * 80)
        ex = eff_data.get("excel_summary", {})
        for k, v in ex.items():
            print(f"  {k:<35}: {v}")
        print("=" * 80)
    else:
        print(f"[!] Warning: Efficiency metrics file not found: {eff_file}")

    # 2. Print Segmentation Metric Table
    if metric_file.exists():
        with open(metric_file, "r", encoding="utf-8") as f:
            m_data = json.load(f)
        dist_unit = m_data.get("distance_unit", "mm")
        print("\n" + "=" * 80)
        print(f"=== BẢNG 2: CHỈ SỐ PHÂN VÙNG ({dist_unit.upper()}) ===")
        print("=" * 80)
        print(f"Dataset Profile : {m_data.get('dataset_id', 'unknown')}")
        print(f"Metric Protocol : {m_data.get('metric_protocol', 'unknown')}")
        print(f"Khoảng cách     : {dist_unit}")
        print("-" * 80)
        print(f"{'Target Region':<25} | {'Mean Dice (%)':<15} | {f'Mean HD95 ({dist_unit})':<15}")
        print("-" * 80)

        regions = [
            ("Myocardium (Whole Ring)", "myocardial_ring"),
            ("Scar (LGE)", "scar"),
            ("Edema (Exclusive)", "edema"),
            ("Avg Pathology (Scar + Edema-Excl)", "avg_pathology_dice_pseudo"),
            ("--- Phụ trợ / Tham khảo ---", None),
            ("Edema-inclusive (LGE-abnormal)", "edema_inclusive"),
            ("Normal Myocardium", "normal_myocardium"),
        ]

        hd_key = f"mean_hd95_{dist_unit}"
        for name, key in regions:
            if key is None:
                print("-" * 80)
                continue
            if key == "avg_pathology_dice_pseudo":
                avg_exc = m_data.get("avg_pathology_dice")
                dsc_str = f"{avg_exc*100:.2f}%" if avg_exc is not None else "N/A"
                print(f"{name:<35} | {dsc_str:<15} | {'-':<15}")
                continue

            st = m_data.get(key, {})
            dsc = st.get("mean_dice")
            hd = st.get(hd_key)
            dsc_str = f"{dsc*100:.2f}%" if dsc is not None else "N/A"
            hd_str = f"{hd:.2f} {dist_unit}" if hd is not None else "N/A"
            print(f"{name:<35} | {dsc_str:<15} | {hd_str:<15}")

        print("=" * 80 + "\n")
    else:
        print(f"[!] Warning: Metrics evaluation file not found: {metric_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Print benchmark summary tables.")
    parser.add_argument("--run_dir", required=True, help="Path to experiment run directory")
    args = parser.parse_args()
    print_tables(args.run_dir)


if __name__ == "__main__":
    main()
