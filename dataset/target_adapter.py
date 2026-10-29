"""Target Adapter for MyoPS-Net MyoPS380 adaptation.

Converts between canonical 4-class labels {0: BG, 1: Normal Myo, 2: Edema, 3: Scar}
and MyoPS-Net multi-branch targets ('cardiac', 'scar', 'edema').
Conforms to protocol myops380_train304_test76_last_v1 and metric protocol myops380_voxel_v3.
Supports shape invariance, volumetric dimension handling, and rank restoration.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence
import torch
import torch.nn as nn


class TargetDict(dict):
    """Dictionary subclass holding adapted target tensors with batch metadata."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._unbatched_2d: bool = False
        self._unbatched_3d: bool = False

    @property
    def is_unbatched_2d(self) -> bool:
        return self._unbatched_2d

    @is_unbatched_2d.setter
    def is_unbatched_2d(self, value: bool) -> None:
        self._unbatched_2d = bool(value)

    @property
    def is_unbatched_3d(self) -> bool:
        return self._unbatched_3d

    @is_unbatched_3d.setter
    def is_unbatched_3d(self, value: bool) -> None:
        self._unbatched_3d = bool(value)

    @property
    def unbatched(self) -> bool:
        return self._unbatched_2d or self._unbatched_3d


def canonical_to_myops_targets(label: torch.Tensor | Any) -> TargetDict:
    """Converts canonical 4-class label tensor to MyoPS-Net 3-branch targets.

    Supports 2D single slices (H, W), batched 2D slices (B, H, W), unbatched 3D volumes
    (H, W, D), batched 3D volumes (B, H, W, D), and channel-1 inputs (B, 1, H, W)
    or (B, 1, H, W, D).

    Args:
        label: Tensor of shape (H, W), (B, H, W), (H, W, D), (B, H, W, D),
            (B, 1, H, W), or (B, 1, H, W, D) with canonical values in
            {0: BG, 1: Normal Myo, 2: Edema, 3: Scar}.

    Returns:
        TargetDict containing:
            - 'scar': shape (B, 3, *spatial_shape) float32 one-hot:
                ch 0 = BG (0)
                ch 1 = Non-scar Myocardium (1 | 2)
                ch 2 = Scar (3)
            - 'edema': shape (B, 3, *spatial_shape) float32 one-hot:
                ch 0 = BG (0)
                ch 1 = Normal Myocardium (1)
                ch 2 = Edema inclusive / Area at Risk (2 | 3)
            - 'cardiac': shape (B, 3, *spatial_shape) float32 one-hot:
                ch 0 = BG (0)
                ch 1 = Total Myocardium (1 | 2 | 3)
                ch 2 = All zeros (no LV blood pool label in MyoPS380)
    """
    if not isinstance(label, torch.Tensor):
        label = torch.as_tensor(label)

    is_unbatched_2d = False
    is_unbatched_3d = False

    # Squeeze singleton channel 1 if present
    if label.dim() == 4 and label.shape[1] == 1:
        label = label.squeeze(1)
    elif label.dim() == 5 and label.shape[1] == 1:
        label = label.squeeze(1)

    # Normalize dimensions to (B, *spatial_shape)
    if label.dim() == 2:
        is_unbatched_2d = True
        label = label.unsqueeze(0)  # (1, H, W)
    elif label.dim() == 3:
        # Check if unbatched 3D volume (H, W, D) vs batched 2D slices (B, H, W)
        if label.shape[0] == label.shape[1] and label.shape[2] != label.shape[0]:
            is_unbatched_3d = True
            label = label.unsqueeze(0)  # (1, H, W, D)
        # else batched 2D slices: (B, H, W)
    elif label.dim() == 4:
        # batched 3D volume: (B, H, W, D)
        pass
    else:
        raise ValueError(
            f"Expected label tensor with 2, 3, 4, or 5 dims (or singleton channel 1), got shape {label.shape}"
        )

    B = label.shape[0]
    spatial_shape = label.shape[1:]
    device = label.device
    dtype = torch.float32

    # Canonical class masks
    bg = (label == 0)
    myo = (label == 1)
    edema = (label == 2)
    scar = (label == 3)

    # 1. Target 'scar': (B, 3, *spatial_shape)
    target_scar = torch.zeros((B, 3, *spatial_shape), dtype=dtype, device=device)
    target_scar[:, 0] = bg.to(dtype)
    target_scar[:, 1] = (myo | edema).to(dtype)
    target_scar[:, 2] = scar.to(dtype)

    # 2. Target 'edema': (B, 3, *spatial_shape)
    target_edema = torch.zeros((B, 3, *spatial_shape), dtype=dtype, device=device)
    target_edema[:, 0] = bg.to(dtype)
    target_edema[:, 1] = myo.to(dtype)
    target_edema[:, 2] = (edema | scar).to(dtype)

    # 3. Target 'cardiac': (B, 3, *spatial_shape)
    target_cardiac = torch.zeros((B, 3, *spatial_shape), dtype=dtype, device=device)
    target_cardiac[:, 0] = bg.to(dtype)
    target_cardiac[:, 1] = (myo | edema | scar).to(dtype)
    # Channel 2 remains strictly 0.0 (no LV blood pool segmentation in MyoPS380)

    # Set metadata attributes on individual tensors for direct pass-through
    for t in (target_scar, target_edema, target_cardiac):
        t._unbatched_2d = is_unbatched_2d
        t._unbatched_3d = is_unbatched_3d

    targets = TargetDict({
        "scar": target_scar,
        "edema": target_edema,
        "cardiac": target_cardiac,
    })
    targets._unbatched_2d = is_unbatched_2d
    targets._unbatched_3d = is_unbatched_3d

    return targets


def myops_outputs_to_canonical(
    outputs: Mapping[str, torch.Tensor] | Sequence[torch.Tensor] | torch.Tensor,
) -> torch.Tensor:
    """Converts model branch predictions/logits into mutually exclusive canonical 4-class mask.

    Handles dicts (keys 'scar'/'LGE', 'edema'/'T2', 'cardiac'/'C0', and optional 'mapping')
    or sequences of tensors (res_c0, res_lge, res_t2, res_mapping) or direct tensors.
    Enforces clinical priority: Scar (3) > Edema (2) > Normal Myocardium (1) > BG (0).

    Args:
        outputs: Model outputs (logits, probabilities, or predictions).

    Returns:
        torch.Tensor of int64 with canonical values {0, 1, 2, 3}.
    """
    def _extract_argmax(tensor: torch.Tensor | None) -> torch.Tensor | None:
        if tensor is None:
            return None
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.as_tensor(tensor)

        dim = tensor.dim()
        shape = tensor.shape

        if dim == 2:
            return tensor.long()
        elif dim == 3:
            if shape[0] == 3 and tensor.is_floating_point():
                return torch.argmax(tensor, dim=0)
            return tensor.long()
        elif dim == 4:
            if shape[1] == 3:
                return torch.argmax(tensor, dim=1)
            elif shape[0] == 3 and shape[1] != 3:
                return torch.argmax(tensor, dim=0)
            elif shape[1] == 1:
                return tensor.squeeze(1).long()
            elif shape[0] == 1 and shape[1] != 1 and shape[1] != 3:
                return tensor.squeeze(0).long()
            else:
                return tensor.long()
        elif dim == 5:
            if shape[1] == 3:
                return torch.argmax(tensor, dim=1)
            elif shape[1] == 1:
                return tensor.squeeze(1).long()
            else:
                return torch.argmax(tensor, dim=1)
        else:
            raise ValueError(f"Unsupported prediction tensor dimensionality: {shape}")

    # Check unbatched metadata on container
    is_unbatched = (
        getattr(outputs, "_unbatched_2d", False)
        or getattr(outputs, "is_unbatched_2d", False)
        or getattr(outputs, "_unbatched_3d", False)
        or getattr(outputs, "is_unbatched_3d", False)
        or getattr(outputs, "unbatched", False)
    )

    # Unpack branch predictions
    if isinstance(outputs, Mapping):
        t_scar = outputs.get("scar", outputs.get("LGE"))
        t_edema = outputs.get("edema", outputs.get("T2"))
        t_cardiac = outputs.get("cardiac", outputs.get("C0"))
        t_mapping = outputs.get("mapping", None)
    elif isinstance(outputs, (list, tuple)):
        if len(outputs) == 4:
            t_cardiac, t_scar, t_edema, t_mapping = outputs
        elif len(outputs) == 3:
            t_cardiac, t_scar, t_edema = outputs
            t_mapping = None
        else:
            raise ValueError(f"Expected tuple/list of length 3 or 4, got {len(outputs)}")
    elif isinstance(outputs, torch.Tensor):
        res = _extract_argmax(outputs)
        if is_unbatched and res is not None and res.shape[0] == 1:
            res = res.squeeze(0)
        return res
    else:
        raise TypeError(f"Unsupported outputs type: {type(outputs)}")

    if t_scar is None or t_edema is None or t_cardiac is None:
        raise KeyError("outputs must provide 'scar' (or 'LGE'), 'edema' (or 'T2'), and 'cardiac' (or 'C0').")

    pred_scar = _extract_argmax(t_scar)
    pred_edema = _extract_argmax(t_edema)
    pred_cardiac = _extract_argmax(t_cardiac)
    pred_mapping = _extract_argmax(t_mapping)

    # Priority 1: Scar (Class 3)
    is_scar = (pred_scar == 2)
    if pred_mapping is not None:
        is_scar = is_scar | (pred_mapping == 2)

    # Priority 2: Edema (Class 2)
    is_edema = (pred_edema == 2) & (~is_scar)

    # Priority 3: Normal Myocardium (Class 1)
    is_myo = (
        ((pred_scar == 1) | (pred_edema == 1) | (pred_cardiac == 1))
        & (~is_scar)
        & (~is_edema)
    )

    canonical = torch.zeros_like(pred_scar, dtype=torch.int64)
    canonical[is_myo] = 1
    canonical[is_edema] = 2
    canonical[is_scar] = 3

    # Restore unbatched rank only when caller tagged input as unbatched and batch size is 1
    if is_unbatched and canonical.shape[0] == 1:
        canonical = canonical.squeeze(0)

    return canonical


class TargetAdapter(nn.Module):
    """Modular Target Adapter wrapper for training and inference pipelines."""

    def __init__(self) -> None:
        super().__init__()

    def to_targets(self, label: torch.Tensor) -> TargetDict:
        return canonical_to_myops_targets(label)

    def to_canonical(
        self, outputs: Mapping[str, torch.Tensor] | Sequence[torch.Tensor] | torch.Tensor
    ) -> torch.Tensor:
        return myops_outputs_to_canonical(outputs)

    def forward(self, label: torch.Tensor) -> TargetDict:
        return self.to_targets(label)
