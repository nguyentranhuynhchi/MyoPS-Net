"""
Contract Helpers & Reference Oracles for MyoPS-Net MyoPS380 Adaptation Tests
Author: E2E Test Writer
Protocol: myops380_train304_test76_last_v1
Metric Protocol: myops380_voxel_v3
"""

import os
import sys
import hashlib
import importlib.util
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Authoritative path constants
SCAR_ROOT = r"D:\nckh_model\SCAR"
if os.path.exists(SCAR_ROOT) and SCAR_ROOT not in sys.path:
    sys.path.insert(0, SCAR_ROOT)

SCAR_SPLIT_DIR = r"D:\nckh_model\SCAR\preprocessing\splits\myops380_no_val"
PROCESSED_DATA_DIR = r"D:\nckh_model\Dataset\MyoPS380\Processed_data"
SCAR_SURFACE_DISTANCE_PATH = r"D:\nckh_model\SCAR\training\metrics\surface_distance.py"
SCAR_MYOPS_DATASET_PATH = r"D:\nckh_model\SCAR\training\dataset\myops_dataset.py"

# Expected constants
EXPECTED_TRAIN_COUNT = 1972
EXPECTED_TEST_COUNT = 76
EXPECTED_TRAIN_PATIENT_COUNT = 304
EXPECTED_TRAIN_SHA256 = "16248ec3630bb0493fb421aab5ffb4dc9b1d0fd49c8614d69d3098a92305f919"
EXPECTED_TEST_SHA256 = "3237b31411833cedcb3ecd86d36edae356bd3f880d3fcfd915066f1c850d9cbc"


def compute_id_list_sha256(id_list: list[str]) -> str:
    """Computes SHA256 of sorted IDs as specified in SCAR split_metadata.json:
    lexically sorted, UTF-8 encoded, newline LF after every ID."""
    sorted_ids = sorted([s.strip() for s in id_list if s.strip()])
    blob = "".join(f"{item}\n" for item in sorted_ids).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def reference_canonical_to_myops_targets(label: torch.Tensor) -> dict[str, torch.Tensor]:
    """Reference implementation of Target Adapter:
    Converts canonical label (B, H, W) with values {0:BG, 1:Normal Myo, 2:Edema, 3:Scar}
    into 3 one-hot dictionaries: 'scar', 'edema', 'cardiac' each of shape (B, 3, H, W).
    """
    if not isinstance(label, torch.Tensor):
        label = torch.as_tensor(label)
    
    if label.dim() == 2:
        label = label.unsqueeze(0)
    
    B, H, W = label.shape
    device = label.device
    dtype = torch.float32

    # Canonical classes
    bg = (label == 0)
    myo = (label == 1)
    edema = (label == 2)
    scar = (label == 3)

    # 1. Target 'scar' (B, 3, H, W)
    # Channel 0: BG (0)
    # Channel 1: Non-scar Myo (1 + 2)
    # Channel 2: Scar (3)
    target_scar = torch.zeros((B, 3, H, W), dtype=dtype, device=device)
    target_scar[:, 0] = bg.to(dtype)
    target_scar[:, 1] = (myo | edema).to(dtype)
    target_scar[:, 2] = scar.to(dtype)

    # 2. Target 'edema' (B, 3, H, W)
    # Channel 0: BG (0)
    # Channel 1: Normal Myo (1)
    # Channel 2: Edema inclusive / AAR (2 + 3)
    target_edema = torch.zeros((B, 3, H, W), dtype=dtype, device=device)
    target_edema[:, 0] = bg.to(dtype)
    target_edema[:, 1] = myo.to(dtype)
    target_edema[:, 2] = (edema | scar).to(dtype)

    # 3. Target 'cardiac' (B, 3, H, W)
    # Channel 0: BG (0)
    # Channel 1: Total Myo (1 + 2 + 3)
    # Channel 2: Zeros (no LV label in MyoPS380)
    target_cardiac = torch.zeros((B, 3, H, W), dtype=dtype, device=device)
    target_cardiac[:, 0] = bg.to(dtype)
    target_cardiac[:, 1] = (myo | edema | scar).to(dtype)
    target_cardiac[:, 2] = torch.zeros((B, H, W), dtype=dtype, device=device)

    return {
        "scar": target_scar,
        "edema": target_edema,
        "cardiac": target_cardiac,
    }


def reference_myops_outputs_to_canonical(seg_outputs: dict[str, torch.Tensor]) -> torch.Tensor:
    """Reference reverse mapping from model output logits / predictions to canonical 4-class labels:
    0: BG, 1: Normal Myocardium, 2: Edema, 3: Scar.
    Preserves mutual exclusivity and anatomical priority.
    """
    # Extract prediction maps
    def get_argmax(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.dim() == 4 and tensor.shape[1] > 1:
            return torch.argmax(tensor, dim=1)
        return tensor.squeeze(1) if tensor.dim() == 4 else tensor

    pred_scar = get_argmax(seg_outputs["scar"])
    pred_edema = get_argmax(seg_outputs["edema"])
    pred_cardiac = get_argmax(seg_outputs["cardiac"])
    pred_mapping = get_argmax(seg_outputs["mapping"]) if "mapping" in seg_outputs and seg_outputs["mapping"] is not None else None

    # Priority 1: Scar (Class 3)
    # Detected by LGE scar branch (class 2) or mapping branch (class 2)
    is_scar = (pred_scar == 2)
    if pred_mapping is not None:
        is_scar = is_scar | (pred_mapping == 2)

    # Priority 2: Edema (Class 2)
    # Detected by T2 edema branch (class 2) and NOT scar
    is_edema = (pred_edema == 2) & (~is_scar)

    # Priority 3: Normal Myocardium (Class 1)
    # Detected by any myocardium branch (class 1 in scar, edema, or cardiac) and NOT pathology
    is_myo = ((pred_scar == 1) | (pred_edema == 1) | (pred_cardiac == 1)) & (~is_scar) & (~is_edema)

    canonical = torch.zeros_like(pred_scar, dtype=torch.int64)
    canonical[is_myo] = 1
    canonical[is_edema] = 2
    canonical[is_scar] = 3

    return canonical


def reference_hardened_weight_function(target: torch.Tensor) -> torch.Tensor:
    """Reference implementation of hardened WCELoss weight function:
    - Iterates over all target channels target.shape[1] (avoids shape mismatch)
    - Safely handles absent classes (voxels_i == 0) with 0.0 weight
    - Returns weights on the same device as target tensor
    """
    mask = torch.argmax(target, dim=1)
    voxels_sum = float(mask.numel())
    num_classes = target.shape[1]
    weights = torch.zeros(num_classes, dtype=torch.float32, device=target.device)

    for i in range(num_classes):
        voxels_i = float((mask == i).sum().item())
        if voxels_i > 0:
            weights[i] = float(np.log(voxels_sum / voxels_i))
        else:
            weights[i] = 0.0

    return weights


def load_module_from_file(module_name: str, file_path: str):
    """Dynamically loads a Python module from a file path."""
    if not os.path.exists(file_path):
        return None
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Dynamically load SCAR surface_distance module if available
scar_surface_distance = load_module_from_file("scar_surface_distance", SCAR_SURFACE_DISTANCE_PATH)

# Dynamically load SCAR myops_dataset module if available
scar_myops_dataset = load_module_from_file("scar_myops_dataset", SCAR_MYOPS_DATASET_PATH)


def try_import_target_adapter():
    """Attempts to import implementation canonical_to_myops_targets and myops_outputs_to_canonical."""
    try:
        from dataset.target_adapter import canonical_to_myops_targets, myops_outputs_to_canonical
        return canonical_to_myops_targets, myops_outputs_to_canonical
    except (ImportError, ModuleNotFoundError):
        try:
            from utils.tools import canonical_to_myops_targets, myops_outputs_to_canonical
            return canonical_to_myops_targets, myops_outputs_to_canonical
        except (ImportError, ModuleNotFoundError, AttributeError):
            return None, None


def try_import_myops380_dataset():
    """Attempts to import implementation MyoPS380 dataset classes."""
    try:
        from dataset.myops380_dataset import MyoPS380Dataset, MyoPS380VolumeDataset
        return MyoPS380Dataset, MyoPS380VolumeDataset
    except (ImportError, ModuleNotFoundError):
        try:
            from utils.dataloader import MyoPS380Dataset, MyoPS380VolumeDataset
            return MyoPS380Dataset, MyoPS380VolumeDataset
        except (ImportError, ModuleNotFoundError, AttributeError):
            return None, None


def try_import_data_augmentation():
    """Attempts to import implementation RandomGenerator."""
    try:
        from dataset.data_augmentation import RandomGenerator
        return RandomGenerator
    except (ImportError, ModuleNotFoundError):
        try:
            from utils.tools import RandomGenerator
            return RandomGenerator
        except (ImportError, ModuleNotFoundError, AttributeError):
            return None
