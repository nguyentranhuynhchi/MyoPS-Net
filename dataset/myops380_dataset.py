"""Dataset and DataLoader implementations for MyoPS-Net on the MyoPS380 benchmark.

Conforms strictly to benchmark protocol myops380_train304_test76_last_v1:
- 304 train patients (1,972 slices) from train.txt
- 76 test patients from test_vol.txt
- Zero validation split (val.txt strictly prohibited, no validation loader)
- Direct ingestion of [0.0, 1.0] preprocessed cache (no double division by 255.0)
- Zero-padding for missing modalities T1m and T2starm downstream in __getitem__
- 5-modality dictionary contract: C0, LGE, T2, T1m, T2starm, label, case_id
"""
from __future__ import annotations

import os
from pathlib import Path
import random
import re
from typing import Any, Callable, Generator, Sequence

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from dataset.data_augmentation import RandomGenerator


def read_split_names(split_path: str | Path) -> list[str]:
    """Reads slice or volume case IDs from a manifest file.

    Args:
        split_path: Path to the manifest file (.txt).

    Returns:
        List of trimmed sample names without empty lines or file extensions.

    Raises:
        FileNotFoundError: If manifest file does not exist.
        ValueError: If manifest is empty, has duplicates, or contains extensions.
    """
    path = Path(split_path)
    if not path.is_file():
        raise FileNotFoundError(f"Split manifest does not exist: {path}")

    content = path.read_text(encoding="utf-8-sig")
    names = [line.strip() for line in content.splitlines() if line.strip()]

    if not names:
        raise ValueError(f"Split manifest is empty: {path}")

    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate entries found in manifest: {path}")

    for name in names:
        if Path(name).suffix != "":
            raise ValueError(
                f"Manifest entries must not contain file extensions: {name!r} in {path}"
            )

    return names


def extract_patient_id(sample_name: str) -> str:
    """Extracts base patient ID from a sample name or file path."""
    name = str(sample_name).strip()
    for ext in (".npy.h5", ".npz", ".h5", ".nii.gz", ".nii"):
        if name.endswith(ext):
            name = name[: -len(ext)]
    return re.sub(r"_slice\d+$", "", name)


def ingest_image_array(image: np.ndarray) -> np.ndarray:
    """Ingests and validates preprocessed image array into [0.0, 1.0] float32.

    If input data is already in [0.0, 1.0], it is clipped to [0.0, 1.0] without
    redundant division by 255.0. If input data is in [0, 255], it is scaled.
    """
    img = np.asarray(image, dtype=np.float32)
    if not np.isfinite(img).all():
        raise ValueError("Non-finite values (NaN or Inf) encountered in image array.")

    if img.max() > 1.0 + 1e-4:
        img = np.clip(img, 0.0, 255.0) / 255.0
    else:
        img = np.clip(img, 0.0, 1.0)

    return img.astype(np.float32)


class MyoPS380TrainDataset(Dataset):
    """2D Slice Training Dataset for MyoPS-Net on MyoPS380 under no_val protocol.

    Loads aligned bSSFP, LGE, and T2w slices from preprocessed NPZ files, applies
    SCAR RandomGenerator augmentation, synthesizes zero-padded T1m and T2starm
    channels, and outputs standardized batch dictionaries.
    """

    def __init__(
        self,
        data_dir: str | Path = r"D:\nckh_model\Dataset\MyoPS380\Processed_data",
        split_dir: str | Path = r"D:\nckh_model\SCAR\preprocessing\splits\myops380_no_val",
        split: str = "train",
        img_size: int | tuple[int, int] = (128, 128),
        transform: Callable | None = None,
        validate_files: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        # Handle aliases from kwargs (e.g. data_root, split_path)
        if "data_root" in kwargs and kwargs["data_root"] is not None:
            data_dir = kwargs["data_root"]
        if "split_path" in kwargs and kwargs["split_path"] is not None:
            resolved_split_path = Path(kwargs["split_path"])
        else:
            resolved_split_path = Path(split_dir) / f"{split}.txt" if Path(split_dir).is_dir() else Path(split_dir)

        # Enforce strict no-validation protocol
        split_lower = str(split).lower()
        if "val" in split_lower:
            raise ValueError(
                "Validation split is forbidden under benchmark protocol 'myops380_train304_test76_last_v1'. "
                "All 304 non-test patients are included in train.txt. Validation loaders and loops must not be created."
            )

        self.data_root = Path(data_dir)
        self.split_path = resolved_split_path
        self.img_size = (img_size, img_size) if isinstance(img_size, int) else tuple(img_size)

        # 1. Read and validate manifest
        self.sample_list = read_split_names(self.split_path)

        # 2. Modality directories
        self.modality_dirs = {
            "C0": self.data_root / "bSSFP" / "train_npz",
            "LGE": self.data_root / "LGE" / "train_npz",
            "T2": self.data_root / "T2w" / "train_npz",
        }

        # 3. Optional integrity check on initialization
        if validate_files:
            for name in self.sample_list:
                for mod, mod_dir in self.modality_dirs.items():
                    p = mod_dir / f"{name}.npz"
                    if not p.is_file():
                        raise FileNotFoundError(f"Missing {mod} file for {name}: {p}")

        # 4. Transform setup (SCAR RandomGenerator by default)
        if transform is None:
            self.transform = RandomGenerator(output_size=self.img_size, image_keys=("C0", "LGE", "T2"))
        else:
            self.transform = transform

    def __len__(self) -> int:
        return len(self.sample_list)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        slice_name = self.sample_list[idx]

        # 1. Read aligned modalities from NPZ
        p_c0 = self.modality_dirs["C0"] / f"{slice_name}.npz"
        p_lge = self.modality_dirs["LGE"] / f"{slice_name}.npz"
        p_t2 = self.modality_dirs["T2"] / f"{slice_name}.npz"

        with np.load(p_c0, allow_pickle=False) as d_c0, \
             np.load(p_lge, allow_pickle=False) as d_lge, \
             np.load(p_t2, allow_pickle=False) as d_t2:
            img_c0 = ingest_image_array(d_c0["image"])
            img_lge = ingest_image_array(d_lge["image"])
            img_t2 = ingest_image_array(d_t2["image"])
            label = np.asarray(d_c0["label"])

        # 2. Canonicalize label
        label = np.round(label).astype(np.int64)
        if label.min() < 0 or label.max() > 3:
            raise ValueError(f"Slice {slice_name} has invalid label classes: {np.unique(label)}")

        # 3. Apply SCAR RandomGenerator augmentation to real modalities and label
        sample = {
            "C0": img_c0,
            "LGE": img_lge,
            "T2": img_t2,
            "label": label,
            "case_id": slice_name,
        }
        augmented = self.transform(sample)

        # 4. Extract transformed tensors
        t_c0 = augmented.get("C0", augmented.get("image"))
        t_lge = augmented.get("LGE", augmented.get("image1"))
        t_t2 = augmented.get("T2", augmented.get("image2"))
        t_label = augmented["label"]

        # Ensure spatial dimensions format: image is (1, H, W) float32, label is (H, W) int64
        if not isinstance(t_c0, torch.Tensor):
            t_c0 = torch.from_numpy(np.ascontiguousarray(t_c0, dtype=np.float32))
        if t_c0.dim() == 2:
            t_c0 = t_c0.unsqueeze(0)

        if not isinstance(t_lge, torch.Tensor):
            t_lge = torch.from_numpy(np.ascontiguousarray(t_lge, dtype=np.float32))
        if t_lge.dim() == 2:
            t_lge = t_lge.unsqueeze(0)

        if not isinstance(t_t2, torch.Tensor):
            t_t2 = torch.from_numpy(np.ascontiguousarray(t_t2, dtype=np.float32))
        if t_t2.dim() == 2:
            t_t2 = t_t2.unsqueeze(0)

        if not isinstance(t_label, torch.Tensor):
            t_label = torch.from_numpy(np.ascontiguousarray(t_label, dtype=np.int64))
        if t_label.dim() == 3 and t_label.shape[0] == 1:
            t_label = t_label.squeeze(0)

        # 5. Synthesize zero-padded channels for missing modalities AFTER augmentation
        # This guarantees missing modalities remain pure zeros without artificial Gaussian noise.
        _, H, W = t_c0.shape
        t_t1m = torch.zeros((1, H, W), dtype=torch.float32)
        t_t2starm = torch.zeros((1, H, W), dtype=torch.float32)

        # 6. Return standard interface contract dictionary
        return {
            "C0": t_c0.float(),              # torch.Tensor (1, 128, 128) float32 in [0.0, 1.0]
            "LGE": t_lge.float(),            # torch.Tensor (1, 128, 128) float32 in [0.0, 1.0]
            "T2": t_t2.float(),              # torch.Tensor (1, 128, 128) float32 in [0.0, 1.0]
            "T1m": t_t1m,                    # torch.Tensor (1, 128, 128) float32 zeros
            "T2starm": t_t2starm,            # torch.Tensor (1, 128, 128) float32 zeros
            "label": t_label.long(),         # torch.Tensor (128, 128) int64 in {0, 1, 2, 3}
            "case_id": slice_name,           # str e.g. "case0002_slice000"
        }


class MyoPS380TestVolumeDataset(Dataset):
    """3D Volume Testing Dataset for MyoPS-Net on MyoPS380.

    Loads 3D volumes (128, 128, D) from HDF5 (.npy.h5) files for the 76 held-out
    test cases. Provides whole-volume numpy arrays and helper methods for slice-by-slice
    or batched inference.
    """

    def __init__(
        self,
        data_dir: str | Path = r"D:\nckh_model\Dataset\MyoPS380\Processed_data",
        split_dir: str | Path = r"D:\nckh_model\SCAR\preprocessing\splits\myops380_no_val",
        split: str = "test_vol",
        validate_files: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        if "data_root" in kwargs and kwargs["data_root"] is not None:
            data_dir = kwargs["data_root"]
        if "split_path" in kwargs and kwargs["split_path"] is not None:
            resolved_split_path = Path(kwargs["split_path"])
        else:
            resolved_split_path = Path(split_dir) / f"{split}.txt" if Path(split_dir).is_dir() else Path(split_dir)

        self.data_root = Path(data_dir)
        self.split_path = resolved_split_path

        # 1. Read test patient IDs
        self.sample_list = read_split_names(self.split_path)

        # 2. Modality paths
        self.modality_dirs = {
            "C0": self.data_root / "bSSFP" / "test_vol_h5",
            "LGE": self.data_root / "LGE" / "test_vol_h5",
            "T2": self.data_root / "T2w" / "test_vol_h5",
        }

        # 3. Optional validation
        if validate_files:
            for name in self.sample_list:
                for mod, mod_dir in self.modality_dirs.items():
                    p = mod_dir / f"{name}.npy.h5"
                    if not p.is_file():
                        raise FileNotFoundError(f"Missing {mod} test volume for {name}: {p}")

    def __len__(self) -> int:
        return len(self.sample_list)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        case_id = self.sample_list[idx]

        p_c0 = self.modality_dirs["C0"] / f"{case_id}.npy.h5"
        p_lge = self.modality_dirs["LGE"] / f"{case_id}.npy.h5"
        p_t2 = self.modality_dirs["T2"] / f"{case_id}.npy.h5"

        with h5py.File(p_c0, "r") as f_c0, \
             h5py.File(p_lge, "r") as f_lge, \
             h5py.File(p_t2, "r") as f_t2:
            img_c0 = ingest_image_array(f_c0["image"][:])
            img_lge = ingest_image_array(f_lge["image"][:])
            img_t2 = ingest_image_array(f_t2["image"][:])
            label = np.asarray(f_c0["label"][:], dtype=np.int64)

        if not (img_c0.shape == img_lge.shape == img_t2.shape == label.shape):
            raise ValueError(f"Shape mismatch in test volume {case_id}: {img_c0.shape}")

        height, width, depth = img_c0.shape

        # Synthesize zero-padded 3D channels for missing modalities
        img_t1m = np.zeros_like(img_c0)
        img_t2starm = np.zeros_like(img_c0)

        return {
            "C0": img_c0,            # np.ndarray (128, 128, D) float32 in [0.0, 1.0]
            "LGE": img_lge,          # np.ndarray (128, 128, D) float32 in [0.0, 1.0]
            "T2": img_t2,            # np.ndarray (128, 128, D) float32 in [0.0, 1.0]
            "T1m": img_t1m,          # np.ndarray (128, 128, D) float32 zeros
            "T2starm": img_t2starm,  # np.ndarray (128, 128, D) float32 zeros
            "label": label,          # np.ndarray (128, 128, D) int64 in {0, 1, 2, 3}
            "case_id": case_id,      # str e.g. "case0001"
            "depth": depth,          # int D
            "spacing": np.array([1.0, 1.0, 1.0], dtype=np.float32),
            "affine": np.eye(4, dtype=np.float32),
        }

    def iter_slice_batches(
        self,
        case_idx_or_name: int | str,
        batch_size: int = 4,
        device: torch.device | str = "cpu",
    ) -> Generator[tuple[dict[str, torch.Tensor], int, int], None, None]:
        """Generator yielding batched 2D slices along the depth axis for model forward passes.

        Yields:
            inputs: dict with keys 'C0', 'LGE', 'T2', 'T1m', 'T2starm'
                    each of shape (B_slice, 1, 128, 128) on specified device.
            start: start slice index along depth.
            stop: stop slice index along depth.
        """
        if isinstance(case_idx_or_name, str):
            idx = self.sample_list.index(case_idx_or_name)
        else:
            idx = case_idx_or_name

        sample = self[idx]
        depth = sample["depth"]

        for start in range(0, depth, batch_size):
            stop = min(start + batch_size, depth)

            def slice_to_tensor(arr: np.ndarray) -> torch.Tensor:
                chunk = arr[:, :, start:stop]  # (128, 128, B_slice)
                tensor = torch.from_numpy(np.ascontiguousarray(chunk.transpose(2, 0, 1)))
                return tensor.unsqueeze(1).to(device)  # (B_slice, 1, 128, 128)

            inputs = {
                "C0": slice_to_tensor(sample["C0"]),
                "LGE": slice_to_tensor(sample["LGE"]),
                "T2": slice_to_tensor(sample["T2"]),
                "T1m": slice_to_tensor(sample["T1m"]),
                "T2starm": slice_to_tensor(sample["T2starm"]),
            }
            yield inputs, start, stop


# Compatibility aliases
MyoPS380Dataset = MyoPS380TrainDataset
MyoPS380VolumeDataset = MyoPS380TestVolumeDataset


def seed_worker(worker_id: int) -> None:
    """Worker initialization function ensuring independent deterministic seeds across workers."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_myops380_dataloader(
    dataset_or_args: Any = None,
    split: str = "train",
    data_dir: str | Path | None = None,
    split_dir: str | Path | None = None,
    batch_size: int | None = None,
    num_workers: int | None = None,
    pin_memory: bool | None = None,
    seed: int = 1234,
    **kwargs: Any,
) -> DataLoader:
    """Builds a standardized PyTorch DataLoader for MyoPS380.

    Ensures drop_last=False so all 1,972 slices are retained across epochs.

    Can be invoked with:
        build_myops380_dataloader(args)
        build_myops380_dataloader(dataset, batch_size=16)
        build_myops380_dataloader(data_dir=..., split_dir=..., batch_size=16)
    """
    # 1. Parse configuration from args object if provided
    args = dataset_or_args if hasattr(dataset_or_args, "batch_size") else None

    if args is not None:
        eff_data_dir = getattr(args, "data_dir", getattr(args, "path", data_dir))
        eff_split_dir = getattr(args, "split_dir", split_dir)
        eff_batch_size = getattr(args, "batch_size", batch_size or 16)
        eff_num_workers = getattr(args, "num_workers", num_workers if num_workers is not None else 4)
        eff_pin_memory = getattr(args, "pin_memory", pin_memory if pin_memory is not None else True)
        eff_seed = getattr(args, "seed", seed)
    else:
        eff_data_dir = data_dir or r"D:\nckh_model\Dataset\MyoPS380\Processed_data"
        eff_split_dir = split_dir or r"D:\nckh_model\SCAR\preprocessing\splits\myops380_no_val"
        eff_batch_size = batch_size or 16
        eff_num_workers = num_workers if num_workers is not None else 4
        eff_pin_memory = pin_memory if pin_memory is not None else True
        eff_seed = seed

    # 2. Resolve dataset
    if isinstance(dataset_or_args, Dataset):
        dataset = dataset_or_args
    else:
        dataset = MyoPS380TrainDataset(
            data_dir=eff_data_dir,
            split_dir=eff_split_dir,
            split=split,
            **kwargs,
        )

    # 3. Deterministic generator
    generator = torch.Generator()
    generator.manual_seed(eff_seed)

    return DataLoader(
        dataset=dataset,
        batch_size=eff_batch_size,
        shuffle=(split == "train"),
        drop_last=False,  # Strictly retain all 1,972 slices
        num_workers=eff_num_workers,
        pin_memory=(torch.cuda.is_available() and eff_pin_memory),
        generator=generator,
        worker_init_fn=seed_worker,
        persistent_workers=(eff_num_workers > 0),
    )
