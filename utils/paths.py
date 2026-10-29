"""Robust path resolution for MyoPS-Net on local and cloud environments."""
from __future__ import annotations

import os
from pathlib import Path


def resolve_split_dir(candidate: str | Path | None = None) -> Path:
    """Resolves split directory containing train.txt and test_vol.txt."""
    if candidate and str(candidate).strip():
        p = Path(candidate)
        if p.is_dir() and (p / "train.txt").is_file():
            return p
    # Try bundled splits inside repository
    repo_splits = Path(__file__).resolve().parent.parent / "dataset" / "splits" / "myops380_no_val"
    if repo_splits.is_dir() and (repo_splits / "train.txt").is_file():
        return repo_splits
    # Common fallback paths
    fallbacks = [
        Path(r"/content/MyoPS-Net/dataset/splits/myops380_no_val"),
        Path(r"D:\nckh_model\SCAR\preprocessing\splits\myops380_no_val"),
        Path(r"../SCAR/preprocessing/splits/myops380_no_val"),
    ]
    for fb in fallbacks:
        if fb.is_dir() and (fb / "train.txt").is_file():
            return fb
    return repo_splits


def resolve_data_root(candidate: str | Path | None = None) -> Path:
    """Resolves dataset root containing bSSFP, LGE, T2w directories."""
    if candidate and str(candidate).strip():
        p = Path(candidate)
        if p.is_dir():
            if (p / "bSSFP").is_dir():
                return p
            if (p / "Processed_data" / "bSSFP").is_dir():
                return p / "Processed_data"

    # Known candidate paths
    search_dirs = [
        Path(r"/content/MyoPS-Net/MyoPS380_dataset/Processed_data"),
        Path(r"/content/MyoPS-Net/MyoPS380_dataset"),
        Path(r"/content/MyoPS380_dataset/Processed_data"),
        Path(r"/content/MyoPS380_dataset"),
        Path(r"/content/dataset_download/Processed_data"),
        Path(r"/content/dataset_download"),
        Path(r"/content/drive/MyDrive/MyoPS380_dataset/Processed_data"),
        Path(r"/content/drive/MyDrive/Processed_data"),
        Path(r"D:\nckh_model\Dataset\MyoPS380\Processed_data"),
        Path(r"../Dataset/MyoPS380/Processed_data"),
    ]
    for d in search_dirs:
        if d.is_dir():
            if (d / "bSSFP").is_dir():
                return d
            if (d / "Processed_data" / "bSSFP").is_dir():
                return d / "Processed_data"

    # Dynamic search under /content if available
    content_dir = Path("/content")
    if content_dir.is_dir():
        for bssfp in content_dir.glob("**/bSSFP"):
            if bssfp.is_dir() and (bssfp.parent / "LGE").is_dir():
                return bssfp.parent

    return Path(candidate) if candidate and str(candidate).strip() else Path(r"D:\nckh_model\Dataset\MyoPS380\Processed_data")


def resolve_save_dir(candidate: str | Path | None = None) -> Path:
    """Resolves directory for saving checkpoints."""
    if candidate and str(candidate).strip():
        p = Path(candidate)
    else:
        p = Path("checkpoints")
    p.mkdir(parents=True, exist_ok=True)
    return p
