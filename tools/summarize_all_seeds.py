"""Summarize all 5 seeds for MyoPS++ segmentation benchmark.
Outputs Excel-ready tab-separated values (with comma decimal separator).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np


def summarize_seeds(base_dir: str | Path = "/content/drive/MyDrive/MyoPS_Results/MyoPS-Net/runs") -> None:
    base_dir = Path(base_dir).resolve()
    seeds = [1234, 1098, 1024, 2026, 6868]

    rows = []
    headers = [
        "Seed",
        "Scar_Dice", "Scar_HD",
        "Edema_Dice", "Edema_HD",
        "Myo_Dice", "Myo_HD",
        "Avg_Path_Dice", "Avg_Path_HD"
    ]

    print("\n" + "=" * 90)
    print("=== TỔNG HỢP KẾT QUẢ 5 SEEDS MYOPS++ (CHUẨN EDEMA_INCLUSIVE & PHYSICAL MM) ===")
    print("=" * 90)

    for seed in seeds:
        metric_file = base_dir / f"myopspp_seed{seed}" / "evaluation" / "metrics.json"
        if not metric_file.exists():
            print(f"[!] Seed {seed}: Chưa tìm thấy {metric_file}")
            continue

        with open(metric_file, "r", encoding="utf-8") as f:
            d = json.load(f)

        s = d.get("scar", {})
        e = d.get("edema_inclusive", {})
        m = d.get("myocardial_ring", {})

        s_d = s.get("mean_dice", 0.0) * 100
        s_hd = s.get("mean_hd95_mm", 0.0)
        e_d = e.get("mean_dice", 0.0) * 100
        e_hd = e.get("mean_hd95_mm", 0.0)
        m_d = m.get("mean_dice", 0.0) * 100
        m_hd = m.get("mean_hd95_mm", 0.0)

        avg_d = (s_d + e_d) / 2.0
        avg_hd = (s_hd + e_hd) / 2.0

        rows.append({
            "seed": seed,
            "s_d": s_d, "s_hd": s_hd,
            "e_d": e_d, "e_hd": e_hd,
            "m_d": m_d, "m_hd": m_hd,
            "avg_d": avg_d, "avg_hd": avg_hd,
        })

    # Print Table
    print(f"{'Seed':<8} | {'Scar Dice':<10} {'HD':<8} | {'Edema Dice':<10} {'HD':<8} | {'Myo Dice':<10} {'HD':<8} | {'Avg Path':<10} {'HD':<8}")
    print("-" * 90)
    for r in rows:
        print(f"{r['seed']:<8} | {r['s_d']:<9.2f}% {r['s_hd']:<7.2f} | {r['e_d']:<9.2f}% {r['e_hd']:<7.2f} | {r['m_d']:<9.2f}% {r['m_hd']:<7.2f} | {r['avg_d']:<9.2f}% {r['avg_hd']:<7.2f}")

    if rows:
        print("-" * 90)
        # Compute mean and std
        means = {k: np.mean([r[k] for r in rows]) for k in ["s_d", "s_hd", "e_d", "e_hd", "m_d", "m_hd", "avg_d", "avg_hd"]}
        stds = {k: np.std([r[k] for r in rows], ddof=1) if len(rows) > 1 else 0.0 for k in ["s_d", "s_hd", "e_d", "e_hd", "m_d", "m_hd", "avg_d", "avg_hd"]}
        print(f"{'Avg':<8} | {means['s_d']:<9.2f}% {means['s_hd']:<7.2f} | {means['e_d']:<9.2f}% {means['e_hd']:<7.2f} | {means['m_d']:<9.2f}% {means['m_hd']:<7.2f} | {means['avg_d']:<9.2f}% {means['avg_hd']:<7.2f}")
        print(f"{'Std':<8} | {stds['s_d']:<9.2f}% {stds['s_hd']:<7.2f} | {stds['e_d']:<9.2f}% {stds['e_hd']:<7.2f} | {stds['m_d']:<9.2f}% {stds['m_hd']:<7.2f} | {stds['avg_d']:<9.2f}% {stds['avg_hd']:<7.2f}")

    print("\n" + "=" * 90)
    print("📋 KHỐI DỮ LIỆU COPY THẲNG VÀO EXCEL (ĐỊNH DẠNG DẤU PHẨY, TAB-SEPARATED):")
    print("=" * 90)
    for r in rows:
        line = f"{r['s_d']:.2f}\t{r['s_hd']:.2f}\t{r['e_d']:.2f}\t{r['e_hd']:.2f}\t{r['m_d']:.2f}\t{r['m_hd']:.2f}\t{r['avg_d']:.2f}\t{r['avg_hd']:.2f}"
        print(line.replace(".", ","))
    if rows:
        avg_line = f"{means['s_d']:.2f}\t{means['s_hd']:.2f}\t{means['e_d']:.2f}\t{means['e_hd']:.2f}\t{means['m_d']:.2f}\t{means['m_hd']:.2f}\t{means['avg_d']:.2f}\t{means['avg_hd']:.2f}"
        print("-" * 40)
        print("Dòng Avg:")
        print(avg_line.replace(".", ","))
    print("=" * 90 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize 5 seeds for MyoPS++")
    parser.add_argument("--base_dir", default="/content/drive/MyDrive/MyoPS_Results/MyoPS-Net/runs", help="Base runs directory")
    args = parser.parse_args()
    summarize_seeds(args.base_dir)


if __name__ == "__main__":
    main()
