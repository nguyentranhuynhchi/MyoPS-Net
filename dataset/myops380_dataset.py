"""MyoPS Dataset Classes for Standardized Benchmark Protocols.
============================================================
Protocols:
  - MyoPS380: myops380_train304_test76_last_v1
  - MyoPS++:  myopspp_train64_test16_last_v1 (myopspp_unet2d_roi128_1mm)

Interface contracts:
  - MyoPSTrainDataset (MyoPS380TrainDataset alias):
      Input: split_file (train.txt), data_root (Processed_data), transform=None, dataset="myops380"
      Yields: dict with keys:
        'bSSFP': torch.FloatTensor (1, 128, 128)
        'LGE':   torch.FloatTensor (1, 128, 128)
        'T2w':   torch.FloatTensor (1, 128, 128)
        'label': torch.LongTensor (128, 128) with values in {0, 1, 2, 3}
        'case_slice': str ID, e.g. 'case0002_slice000' or 'Case2001_slice000'
  - MyoPSTestDataset (MyoPS380TestDataset alias):
      Input: split_file (test_vol.txt), data_root (Processed_data), dataset="myops380"
      Yields: dict with keys:
        'bSSFP': torch.FloatTensor (1, 128, 128, D)
        'LGE':   torch.FloatTensor (1, 128, 128, D)
        'T2w':   torch.FloatTensor (1, 128, 128, D)
        'label': torch.LongTensor (128, 128, D) with values in {0, 1, 2, 3}
        'case_id': str ID, e.g. 'case0001' or 'Case2011'
        'spacing': torch.FloatTensor (3,) [optional, e.g. (1.0, 1.0, s_z) in mm]
"""

from __future__ import annotations

import os
from pathlib import Path
import random
import re
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


# Dataset Constants
MYOPS380_EXPECTED_TRAIN_SLICES = 1972
MYOPS380_EXPECTED_TRAIN_PATIENTS = 304
MYOPS380_EXPECTED_TEST_VOLUMES = 76

MYOPSPP_EXPECTED_TRAIN_SLICES = 296
MYOPSPP_EXPECTED_TRAIN_PATIENTS = 64
MYOPSPP_EXPECTED_TEST_VOLUMES = 16

# Backward compatibility aliases
EXPECTED_TRAIN_SLICES = MYOPS380_EXPECTED_TRAIN_SLICES
EXPECTED_TRAIN_PATIENTS = MYOPS380_EXPECTED_TRAIN_PATIENTS
EXPECTED_TEST_VOLUMES = MYOPS380_EXPECTED_TEST_VOLUMES

CANONICAL_LUT = np.array([0, 1, 3, 2], dtype=np.int64)


def normalize_dataset_name(dataset: Optional[str]) -> str:
    """Normalize dataset identifier string."""
    if dataset is None:
        return "myops380"
    ds = str(dataset).strip().lower()
    if not ds:
        return "myops380"
    if ds in ("myopspp", "myopspp_unet2d_roi128_1mm", "myops_plus_plus", "myops++"):
        return "myopspp"
    if ds in ("myops380", "myops_380", "default"):
        return "myops380"
    raise ValueError(f"Unknown dataset identifier: '{dataset}', expected 'myops380' or 'myopspp'")


def seed_worker(worker_id: int) -> None:
    """Use SCAR's Python/NumPy worker seeding policy (also picklable on Windows)."""
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def resolve_split_file(split_file: Union[str, Path], default_filename: str) -> Path:
    """Resolves split file path whether a directory or file path is passed."""
    path = Path(split_file)
    if path.is_dir():
        path = path / default_filename
    if not path.is_file():
        raise FileNotFoundError(f"Split file not found: {path}")
    return path


def extract_patient_id(case_slice_or_vol: str) -> str:
    """Extracts normalized patient ID from slice name or volume filename (preserving case)."""
    clean = re.sub(r"(\.npy)?\.h5$", "", str(case_slice_or_vol).strip())
    return clean.split("_slice")[0]


patient_id = extract_patient_id


def read_manifest_lines(file_path: Union[str, Path]) -> List[str]:
    """Reads non-empty, stripped lines from a manifest file."""
    path = Path(file_path)
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines


def _to_image_tensor(img: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
    """Ensures 2D slice image is a torch.FloatTensor of shape (1, 128, 128)."""
    if isinstance(img, torch.Tensor):
        t = img.float()
    else:
        t = torch.from_numpy(np.asarray(img, dtype=np.float32))
    if t.dim() == 2:
        t = t.unsqueeze(0)
    elif t.dim() == 3 and t.shape[0] != 1 and t.shape[-1] == 1:
        t = t.permute(2, 0, 1)
    return t


def _to_label_tensor(lbl: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
    """Ensures 2D slice label is a torch.LongTensor of shape (128, 128)."""
    if isinstance(lbl, torch.Tensor):
        t = lbl.long()
    else:
        t = torch.from_numpy(np.asarray(lbl, dtype=np.int64))
    if t.dim() == 3 and t.shape[0] == 1:
        t = t.squeeze(0)
    elif t.dim() == 3 and t.shape[-1] == 1:
        t = t.squeeze(-1)
    return t


def _to_volume_image_tensor(img: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
    """Ensures 3D volume image is a torch.FloatTensor of shape (1, 128, 128, D)."""
    if isinstance(img, torch.Tensor):
        t = img.float()
    else:
        t = torch.from_numpy(np.asarray(img, dtype=np.float32))
    if t.dim() == 3:
        t = t.unsqueeze(0)
    return t


def _to_volume_label_tensor(lbl: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
    """Ensures 3D volume label is a torch.LongTensor of shape (128, 128, D)."""
    if isinstance(lbl, torch.Tensor):
        t = lbl.long()
    else:
        t = torch.from_numpy(np.asarray(lbl, dtype=np.int64))
    if t.dim() == 4 and t.shape[0] == 1:
        t = t.squeeze(0)
    return t


class MyoPSTrainDataset(Dataset):
    """Slice-level training dataset for MyoPS380 and MyoPS++ benchmarks.

    Loads aligned bSSFP, LGE, and T2w 2D image slices along with ground-truth
    segmentation masks across slices specified in train.txt.
    Supports both lowercase case IDs (case0001) and uppercase case IDs (Case2001).
    """

    def __init__(
        self,
        split_file: Union[str, Path],
        data_root: Union[str, Path],
        transform: Optional[Callable] = None,
        validate_files: bool = False,
        strict: bool = False,
        dataset: str = "myops380",
    ):
        self.dataset = normalize_dataset_name(dataset)
        self.split_file = resolve_split_file(split_file, "train.txt")
        self.data_root = Path(data_root)
        if not self.data_root.is_dir():
            raise FileNotFoundError(f"data_root directory not found: {self.data_root}")

        self.transform = transform
        self.slices: List[str] = read_manifest_lines(self.split_file)

        if not self.slices:
            raise ValueError(f"Train manifest is empty: {self.split_file}")

        self.patients = sorted(list({extract_patient_id(s) for s in self.slices}))

        if strict:
            if self.dataset == "myopspp":
                exp_slices = MYOPSPP_EXPECTED_TRAIN_SLICES
                exp_patients = MYOPSPP_EXPECTED_TRAIN_PATIENTS
            else:
                exp_slices = MYOPS380_EXPECTED_TRAIN_SLICES
                exp_patients = MYOPS380_EXPECTED_TRAIN_PATIENTS

            if len(self.slices) != exp_slices:
                raise ValueError(
                    f"Expected exactly {exp_slices} training slices for {self.dataset}, "
                    f"got {len(self.slices)} in {self.split_file}"
                )
            if len(self.patients) != exp_patients:
                raise ValueError(
                    f"Expected exactly {exp_patients} unique training patients for {self.dataset}, "
                    f"got {len(self.patients)} in {self.split_file}"
                )

        if validate_files:
            self._validate_all_files()

    def _validate_all_files(self) -> None:
        missing = []
        for modality in ("bSSFP", "LGE", "T2w"):
            mod_dir = self.data_root / modality / "train_npz"
            for slice_id in self.slices:
                npz_path = mod_dir / f"{slice_id}.npz"
                if not npz_path.is_file():
                    missing.append(str(npz_path))
        if missing:
            raise FileNotFoundError(
                f"Missing {len(missing)} train slice files; first 5: {missing[:5]}"
            )

    def __len__(self) -> int:
        return len(self.slices)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
        slice_id = self.slices[idx]

        bssfp_path = self.data_root / "bSSFP" / "train_npz" / f"{slice_id}.npz"
        lge_path = self.data_root / "LGE" / "train_npz" / f"{slice_id}.npz"
        t2w_path = self.data_root / "T2w" / "train_npz" / f"{slice_id}.npz"

        with np.load(bssfp_path) as d_bssfp:
            bssfp = np.asarray(d_bssfp["image"], dtype=np.float32)
            label = np.asarray(d_bssfp["label"])

        with np.load(lge_path) as d_lge:
            lge = np.asarray(d_lge["image"], dtype=np.float32)

        with np.load(t2w_path) as d_t2w:
            t2w = np.asarray(d_t2w["image"], dtype=np.float32)

        if self.transform is not None:
            sample_in = {
                "bSSFP": bssfp,
                "LGE": lge,
                "T2w": t2w,
                "image": bssfp,
                "image1": lge,
                "image2": t2w,
                "label": label,
                "case_slice": slice_id,
            }
            sample_out = self.transform(sample_in)

            if isinstance(sample_out, dict):
                if "bSSFP" in sample_out and (
                    isinstance(sample_out["bSSFP"], torch.Tensor)
                    or sample_out["bSSFP"] is not bssfp
                ):
                    bssfp = sample_out["bSSFP"]
                elif "image" in sample_out:
                    bssfp = sample_out["image"]
                else:
                    bssfp = sample_out.get("bSSFP", bssfp)

                if "LGE" in sample_out and (
                    isinstance(sample_out["LGE"], torch.Tensor)
                    or sample_out["LGE"] is not lge
                ):
                    lge = sample_out["LGE"]
                elif "image1" in sample_out:
                    lge = sample_out["image1"]
                else:
                    lge = sample_out.get("LGE", lge)

                if "T2w" in sample_out and (
                    isinstance(sample_out["T2w"], torch.Tensor)
                    or sample_out["T2w"] is not t2w
                ):
                    t2w = sample_out["T2w"]
                elif "image2" in sample_out:
                    t2w = sample_out["image2"]
                else:
                    t2w = sample_out.get("T2w", t2w)

                label = sample_out.get("label", label)
            elif isinstance(sample_out, (list, tuple)) and len(sample_out) >= 4:
                bssfp, lge, t2w, label = sample_out[0], sample_out[1], sample_out[2], sample_out[3]

        return {
            "bSSFP": _to_image_tensor(bssfp),
            "LGE": _to_image_tensor(lge),
            "T2w": _to_image_tensor(t2w),
            "label": _to_label_tensor(label),
            "case_slice": slice_id,
        }


# Backward compatibility alias
MyoPS380TrainDataset = MyoPSTrainDataset


class MyoPSTestDataset(Dataset):
    """Volume-level test evaluation dataset for MyoPS380 and MyoPS++ benchmarks.

    Loads aligned 3D volumes (128, 128, D) for bSSFP, LGE, and T2w modalities,
    along with 3D ground truth segmentation masks and dynamic physical spacing [1.0, 1.0, s_z].
    Supports both lowercase case IDs (case0001) and uppercase case IDs (Case2011).
    """

    def __init__(
        self,
        split_file: Union[str, Path],
        data_root: Union[str, Path],
        canonical_label: bool = False,
        validate_files: bool = False,
        strict: bool = False,
        dataset: str = "myops380",
    ):
        self.dataset = normalize_dataset_name(dataset)
        self.split_file = resolve_split_file(split_file, "test_vol.txt")
        self.data_root = Path(data_root)
        if not self.data_root.is_dir():
            raise FileNotFoundError(f"data_root directory not found: {self.data_root}")

        self.canonical_label = canonical_label
        raw_cases = read_manifest_lines(self.split_file)
        self.cases: List[str] = [re.sub(r"(\.npy)?\.h5$", "", c) for c in raw_cases]

        if not self.cases:
            raise ValueError(f"Test manifest is empty: {self.split_file}")

        if strict:
            exp_vols = MYOPSPP_EXPECTED_TEST_VOLUMES if self.dataset == "myopspp" else MYOPS380_EXPECTED_TEST_VOLUMES
            if len(self.cases) != exp_vols:
                raise ValueError(
                    f"Expected exactly {exp_vols} test volumes for {self.dataset}, "
                    f"got {len(self.cases)} in {self.split_file}"
                )

        if validate_files:
            self._validate_all_files()

    def _validate_all_files(self) -> None:
        missing = []
        for modality in ("bSSFP", "LGE", "T2w"):
            mod_dir = self.data_root / modality / "test_vol_h5"
            for case_id in self.cases:
                h5_path = mod_dir / f"{case_id}.npy.h5"
                if not h5_path.is_file():
                    h5_path_alt = mod_dir / f"{case_id}.h5"
                    if not h5_path_alt.is_file():
                        missing.append(str(h5_path))
        if missing:
            raise FileNotFoundError(
                f"Missing {len(missing)} test volume files; first 5: {missing[:5]}"
            )

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        case_id = self.cases[idx]

        bssfp_path = self.data_root / "bSSFP" / "test_vol_h5" / f"{case_id}.npy.h5"
        if not bssfp_path.is_file():
            bssfp_path = self.data_root / "bSSFP" / "test_vol_h5" / f"{case_id}.h5"

        lge_path = self.data_root / "LGE" / "test_vol_h5" / f"{case_id}.npy.h5"
        if not lge_path.is_file():
            lge_path = self.data_root / "LGE" / "test_vol_h5" / f"{case_id}.h5"

        t2w_path = self.data_root / "T2w" / "test_vol_h5" / f"{case_id}.npy.h5"
        if not t2w_path.is_file():
            t2w_path = self.data_root / "T2w" / "test_vol_h5" / f"{case_id}.h5"

        spacing = None
        with h5py.File(bssfp_path, "r") as f_bssfp:
            bssfp = f_bssfp["image"][:]
            label = f_bssfp["label"][:]
            if "spacing" in f_bssfp:
                spacing = np.asarray(f_bssfp["spacing"][:], dtype=np.float32)
            elif "spacing" in f_bssfp.attrs:
                spacing = np.asarray(f_bssfp.attrs["spacing"], dtype=np.float32)

        with h5py.File(lge_path, "r") as f_lge:
            lge = f_lge["image"][:]
            if spacing is None:
                if "spacing" in f_lge:
                    spacing = np.asarray(f_lge["spacing"][:], dtype=np.float32)
                elif "spacing" in f_lge.attrs:
                    spacing = np.asarray(f_lge.attrs["spacing"], dtype=np.float32)

        with h5py.File(t2w_path, "r") as f_t2w:
            t2w = f_t2w["image"][:]
            if spacing is None:
                if "spacing" in f_t2w:
                    spacing = np.asarray(f_t2w["spacing"][:], dtype=np.float32)
                elif "spacing" in f_t2w.attrs:
                    spacing = np.asarray(f_t2w.attrs["spacing"], dtype=np.float32)

        label_np = np.asarray(label, dtype=np.int64)
        # Double-swap prevention:
        # MyoPS380 GT is legacy order: convert to canonical only if requested.
        # MyoPS++ GT is ALREADY canonical: do NOT apply CANONICAL_LUT!
        if self.canonical_label and self.dataset == "myops380":
            label_np = CANONICAL_LUT[label_np]

        # For MyoPS++, if spacing was not stored in H5, default to [1.0, 1.0, 1.0] mm
        if spacing is None and self.dataset == "myopspp":
            spacing = np.array([1.0, 1.0, 1.0], dtype=np.float32)

        result: Dict[str, Any] = {
            "bSSFP": _to_volume_image_tensor(bssfp),
            "LGE": _to_volume_image_tensor(lge),
            "T2w": _to_volume_image_tensor(t2w),
            "label": _to_volume_label_tensor(label_np),
            "case_id": case_id,
        }
        if spacing is not None:
            result["spacing"] = torch.from_numpy(spacing.astype(np.float32))

        return result


# Backward compatibility alias
MyoPS380TestDataset = MyoPSTestDataset


def create_myops_train_loader(
    split_file: Union[str, Path],
    data_root: Union[str, Path],
    dataset: str = "myops380",
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    transform: Optional[Callable] = None,
    strict: bool = False,
    generator: Optional[torch.Generator] = None,
) -> DataLoader:
    """Factory helper to construct standardized train DataLoader for MyoPS380 or MyoPS++."""
    ds = MyoPSTrainDataset(
        split_file=split_file,
        data_root=data_root,
        transform=transform,
        strict=strict,
        dataset=dataset,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        generator=generator,
        worker_init_fn=seed_worker,
        persistent_workers=False,
    )


def create_myops380_train_loader(
    split_file: Union[str, Path],
    data_root: Union[str, Path],
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    transform: Optional[Callable] = None,
    strict: bool = False,
    generator: Optional[torch.Generator] = None,
) -> DataLoader:
    """Factory helper for MyoPS380 train DataLoader (100% backward compatible)."""
    return create_myops_train_loader(
        split_file=split_file,
        data_root=data_root,
        dataset="myops380",
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        transform=transform,
        strict=strict,
        generator=generator,
    )


def create_myops_test_loader(
    split_file: Union[str, Path],
    data_root: Union[str, Path],
    dataset: str = "myops380",
    canonical_label: bool = False,
    num_workers: int = 0,
    strict: bool = False,
) -> DataLoader:
    """Factory helper to construct standardized test DataLoader for MyoPS380 or MyoPS++."""
    ds = MyoPSTestDataset(
        split_file=split_file,
        data_root=data_root,
        canonical_label=canonical_label,
        strict=strict,
        dataset=dataset,
    )
    return DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
    )


def create_myops380_test_loader(
    split_file: Union[str, Path],
    data_root: Union[str, Path],
    canonical_label: bool = False,
    num_workers: int = 0,
    strict: bool = False,
) -> DataLoader:
    """Factory helper for MyoPS380 test DataLoader (100% backward compatible)."""
    return create_myops_test_loader(
        split_file=split_file,
        data_root=data_root,
        dataset="myops380",
        canonical_label=canonical_label,
        num_workers=num_workers,
        strict=strict,
    )
