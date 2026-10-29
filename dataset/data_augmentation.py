"""Data augmentation pipeline replicating SCAR's RandomGenerator and ResizeGenerator.

Provides synchronized spatial transforms across aligned modalities and independent
intensity variation on [0.0, 1.0] CMR images. Labels are strictly preserved in integer
space {0, 1, 2, 3}. Conforms to benchmark protocol myops380_train304_test76_last_v1.
"""
from __future__ import annotations

import random
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import ndimage
import torch
import torch.nn.functional as F


def random_rot_flip(*arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    """Synchronized random 90-degree rotation and reflection.

    Args:
        *arrays: 2D numpy arrays with identical spatial dimensions.

    Returns:
        Tuple of transformed arrays with contiguous C-order memory layout and positive strides.
    """
    k = int(np.random.randint(0, 4))
    axis = int(np.random.randint(0, 2))
    return tuple(np.flip(np.rot90(x, k), axis=axis).copy() for x in arrays)


def random_rotate(*arrays, angle: float | None = None) -> tuple[Any, ...]:
    """Synchronized random free-angle rotation in [-20.0, 20.0] degrees.

    Uses order=1 (bilinear) for image modalities and order=0 (nearest-neighbor)
    for segmentation mask to prevent fractional class interpolation.

    Can be called as:
        random_rotate(image, image1, image2, label, angle=None)
        random_rotate(images_list, label, angle=None)
    """
    if angle is None:
        angle = float(np.random.uniform(-20.0, 20.0))

    if len(arrays) == 2 and isinstance(arrays[0], (list, tuple)):
        images, label = arrays[0], arrays[1]
        rot_images = [
            ndimage.rotate(
                x,
                angle,
                order=1,
                reshape=False,
                mode="constant",
                cval=0.0,
                prefilter=False,
            )
            for x in images
        ]
        rot_label = ndimage.rotate(
            label,
            angle,
            order=0,
            reshape=False,
            mode="constant",
            cval=0,
            prefilter=False,
        )
        return rot_images, rot_label

    n = len(arrays)
    return tuple(
        ndimage.rotate(
            x,
            angle,
            order=0 if i == n - 1 else 1,
            reshape=False,
            mode="constant",
            cval=0.0 if i < n - 1 else 0,
            prefilter=False,
        )
        for i, x in enumerate(arrays)
    )


def random_shift(*arrays, shift: tuple[int, int] | None = None) -> tuple[Any, ...]:
    """Synchronized random 2D translation in [-5, 5] pixels along both axes.

    Uses order=1 (bilinear) for images and order=0 (nearest-neighbor) for labels.
    Boundary padding uses mode='constant', cval=0.0 (never wraps image edges).

    Can be called as:
        random_shift(image, image1, image2, label, shift=None)
        random_shift(images_list, label, shift=None)
    """
    if shift is None:
        shift = (random.randint(-5, 5), random.randint(-5, 5))

    if len(arrays) == 2 and isinstance(arrays[0], (list, tuple)):
        images, label = arrays[0], arrays[1]
        shift_images = [
            ndimage.shift(
                x,
                shift,
                order=1,
                mode="constant",
                cval=0.0,
                prefilter=False,
            )
            for x in images
        ]
        shift_label = ndimage.shift(
            label,
            shift,
            order=0,
            mode="constant",
            cval=0,
            prefilter=False,
        )
        return shift_images, shift_label

    n = len(arrays)
    return tuple(
        ndimage.shift(
            x,
            shift,
            order=0 if i == n - 1 else 1,
            mode="constant",
            cval=0.0 if i < n - 1 else 0,
            prefilter=False,
        )
        for i, x in enumerate(arrays)
    )


def add_gaussian_noise(*arrays: np.ndarray, mean: float = 0.0, std: float = 0.02) -> tuple[np.ndarray, ...]:
    """Adds independent Gaussian noise to image modalities, leaving label intact."""
    images = tuple(
        np.clip(x + np.random.normal(mean, std, x.shape), 0.0, 1.0)
        for x in arrays[:-1]
    )
    return (*images, arrays[-1])


def apply_intensity_transforms(
    image: np.ndarray,
    p_bright: float = 0.5,
    p_gamma: float = 0.5,
    p_noise: float = 0.25,
) -> np.ndarray:
    """Independent intensity variation on a single [0.0, 1.0] CMR image.

    Applies:
    - Random brightness scaling in [0.8, 1.2] (p=0.5)
    - Random gamma contrast in [0.8, 1.2] (p=0.5)
    - Additive Gaussian noise N(0, 0.02) (p=0.25)
    Always clips output to [0.0, 1.0].
    """
    out = image.copy()
    if random.random() < p_bright:
        factor = random.uniform(0.8, 1.2)
        out = np.clip(out * factor, 0.0, 1.0)
    if random.random() < p_gamma:
        gamma = random.uniform(0.8, 1.2)
        out = np.clip(out, 0.0, 1.0) ** gamma
    if random.random() < p_noise:
        noise = np.random.normal(0.0, 0.02, out.shape)
        out = np.clip(out + noise, 0.0, 1.0)
    return out.astype(np.float32)


class ResizeGenerator:
    """Deterministic, alignment-preserving image resize and PyTorch tensor conversion.

    Validates spatial shapes, resizes images via bilinear interpolation,
    resizes discrete labels via nearest-neighbor interpolation, and formats
    image tensors as (1, H, W) float32 and label as (H, W) int64.
    """

    def __init__(
        self,
        output_size: Sequence[int] = (128, 128),
        image_keys: Sequence[str] | None = None,
        label_key: str = "label",
    ) -> None:
        if len(output_size) != 2 or any(int(v) <= 0 for v in output_size):
            raise ValueError("output_size must contain two positive dimensions.")
        self.output_size = tuple(int(v) for v in output_size)
        self.image_keys = tuple(image_keys) if image_keys is not None else None
        self.label_key = label_key

    def _resolve_image_keys(self, sample: Mapping[str, Any]) -> tuple[str, ...]:
        if self.image_keys is not None:
            return self.image_keys
        if "C0" in sample and "LGE" in sample and "T2" in sample:
            return ("C0", "LGE", "T2")
        if "image" in sample and "image1" in sample and "image2" in sample:
            return ("image", "image1", "image2")
        return tuple(
            k
            for k, v in sample.items()
            if k != self.label_key and isinstance(v, (np.ndarray, torch.Tensor)) and v.ndim == 2
        )

    def __call__(self, sample: Mapping[str, Any]) -> dict[str, Any]:
        result = dict(sample)
        if self.label_key not in sample:
            raise KeyError(f"Sample missing required label key {self.label_key!r}")

        label_arr = np.asarray(sample[self.label_key])
        shape = label_arr.shape
        if len(shape) != 2:
            raise ValueError(f"Slice transforms require 2D samples, got shape {shape}.")

        image_keys = self._resolve_image_keys(sample)
        for key in (*image_keys, self.label_key):
            value = np.asarray(sample[key])
            if value.shape != shape:
                raise ValueError(
                    f"Modality {key!r} shape {value.shape} does not match label shape {shape}."
                )
            dtype = np.int64 if key == self.label_key else np.float32
            tensor = torch.from_numpy(np.array(value, dtype=dtype, copy=True, order="C"))
            if shape != self.output_size:
                if key == self.label_key:
                    tensor = F.interpolate(
                        tensor[None, None].float(),
                        size=self.output_size,
                        mode="nearest",
                    )[0, 0].long()
                else:
                    tensor = F.interpolate(
                        tensor[None, None],
                        size=self.output_size,
                        mode="bilinear",
                        align_corners=False,
                    )[0, 0]
            result[key] = tensor if key == self.label_key else tensor.unsqueeze(0)
        return result


class RandomGenerator(ResizeGenerator):
    """Shared spatial warps and independent intensity variation on [0.0, 1.0] CMR images.

    Replicates SCAR's exact RandomGenerator behavior:
    - Synchronized spatial transforms: rot/flip (p=0.5), free rotate [-20, 20] deg (p=0.5 if not flipped),
      shift [-5, 5] px (p=0.5).
    - Independent intensity transforms per modality: brightness (p=0.5), gamma (p=0.5), Gaussian noise (p=0.25).
    - Labels are untouched during intensity variations.
    - Deterministic resize and tensor formatting via ResizeGenerator.
    """

    def __call__(self, sample: Mapping[str, Any]) -> dict[str, Any]:
        result = dict(sample)
        image_keys = self._resolve_image_keys(sample)
        images = [np.asarray(sample[k], dtype=np.float32).copy() for k in image_keys]
        label = np.asarray(sample[self.label_key]).copy()

        # Step 1: Spatial Transforms (Synchronized)
        if random.random() < 0.5:
            *images, label = random_rot_flip(*images, label)
        elif random.random() < 0.5:
            *images, label = random_rotate(*images, label)

        if random.random() < 0.5:
            shift = (random.randint(-5, 5), random.randint(-5, 5))
            images = [
                ndimage.shift(x, shift, order=1, mode="constant", cval=0.0, prefilter=False)
                for x in images
            ]
            label = ndimage.shift(label, shift, order=0, mode="constant", cval=0, prefilter=False)

        # Step 2: Intensity Transforms (Independent per modality; label untouched)
        for i, key in enumerate(image_keys):
            images[i] = apply_intensity_transforms(images[i])
            result[key] = images[i]

        result[self.label_key] = label

        # Step 3: Resizing and Tensor Conversion via base class
        return super().__call__(result)
