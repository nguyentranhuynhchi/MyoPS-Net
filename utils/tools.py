"""
Target Adapters, Prediction Fusion, and Utility Tools for MyoPS-Net.
Standardized for MyoPS380 Benchmark Recipe.
"""

import random
from typing import Tuple, Union
import numpy as np
try:
    from skimage.transform import resize
except ImportError:
    try:
        from scipy.ndimage import zoom

        def resize(image, output_shape, order=1, mode='constant', preserve_range=True):
            factors = [o / max(i, 1) for o, i in zip(output_shape, image.shape)]
            return zoom(image, factors, order=order)
    except ImportError:
        resize = None
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from torchvision import transforms
except ImportError:
    transforms = None


def weights_init(m: nn.Module) -> None:
    """Kaiming normal initialization for conv layers, uniform for linear."""
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find("BatchNorm") != -1:
        nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(m.bias, 0.0)
    elif classname.find("Linear") != -1:
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


def to_one_hot(target: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Converts integer class index tensor to float one-hot tensor.
    
    Supports shapes:
      - (B, H, W) -> (B, num_classes, H, W)
      - (H, W) -> (num_classes, H, W)
      - (B, 1, H, W) -> (B, num_classes, H, W)
    """
    if target.dim() == 4 and target.shape[1] == 1:
        target = target.squeeze(1)

    target_long = target.long()
    one_hot = F.one_hot(target_long, num_classes=num_classes)
    if one_hot.dim() == 4:
        return one_hot.permute(0, 3, 1, 2).float()
    elif one_hot.dim() == 3:
        return one_hot.permute(2, 0, 1).float()
    return one_hot.float()


def create_multi_head_targets(
    label: Union[torch.Tensor, np.ndarray],
    mode: str = "legacy",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Adapts MyoPS380 labels into multi-head supervisory targets for MyoPS-Net.
    
    Args:
        label: Input tensor of shape (B, 128, 128), (128, 128), or (B, 1, 128, 128).
        mode: Label schema of input:
              - 'legacy' (default): 0: BG, 1: Normal, 2: Scar, 3: Edema-exclusive
              - 'canonical':        0: BG, 1: Normal, 2: Edema-exclusive, 3: Scar
              
    Returns:
        target_C0:  torch.long in {0, 1} (Whole Myo vs Background)
        target_LGE: torch.long in {0, 1, 2} (0: BG, 1: Non-scar Myo, 2: Scar)
        target_T2:  torch.long in {0, 1, 2} (0: BG, 1: Normal Myo, 2: Edema-inclusive)
    """
    if not isinstance(label, torch.Tensor):
        label = torch.as_tensor(label)

    # Squeeze channel dimension if shape is (B, 1, H, W)
    if label.dim() == 4 and label.shape[1] == 1:
        label = label.squeeze(1)

    device = label.device
    label_clamped = label.clamp(0, 3).long()

    if mode == "legacy":
        lut_c0 = torch.tensor([0, 1, 1, 1], dtype=torch.long, device=device)
        lut_lge = torch.tensor([0, 1, 2, 1], dtype=torch.long, device=device)
        lut_t2 = torch.tensor([0, 1, 2, 2], dtype=torch.long, device=device)
    elif mode == "canonical":
        lut_c0 = torch.tensor([0, 1, 1, 1], dtype=torch.long, device=device)
        lut_lge = torch.tensor([0, 1, 1, 2], dtype=torch.long, device=device)
        lut_t2 = torch.tensor([0, 1, 2, 2], dtype=torch.long, device=device)
    else:
        raise ValueError(f"Unknown mode '{mode}', expected 'legacy' or 'canonical'")

    target_C0 = lut_c0[label_clamped]
    target_LGE = lut_lge[label_clamped]
    target_T2 = lut_t2[label_clamped]

    return target_C0, target_LGE, target_T2


# Backward compatibility alias
canonical_to_myops_targets = lambda label: create_multi_head_targets(label, mode="canonical")


def fuse_predictions(
    seg_C0: Union[torch.Tensor, np.ndarray],
    seg_LGE: Union[torch.Tensor, np.ndarray],
    seg_T2: Union[torch.Tensor, np.ndarray],
) -> torch.Tensor:
    """Fuses multi-head predictions into canonical 4-class segmentation.
    
    Decision Hierarchy:
      1. Scar:               P_LGE == 2                         -> ID 3
      2. Edema-exclusive:    (P_T2 == 2) & ~Scar                -> ID 2
      3. Normal Myocardium:  (P_C0 == 1) & ~Scar & ~Edema       -> ID 1
      4. Background:         Remaining voxels                   -> ID 0
      
    Args:
        seg_C0:  Softmax probabilities (B, 2, H, W), (2, H, W), or argmax indices
        seg_LGE: Softmax probabilities (B, 3, H, W), (3, H, W), or argmax indices
        seg_T2:  Softmax probabilities (B, 3, H, W), (3, H, W), or argmax indices
        
    Returns:
        canonical: uint8 tensor with canonical IDs {0, 1, 2, 3}.
    """
    def _to_pred(x):
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x)
        if x.dim() == 4:
            return torch.argmax(x, dim=1)
        elif x.dim() == 3:
            if x.is_floating_point() and x.shape[0] in (2, 3, 4):
                return torch.argmax(x, dim=0)
            return x
        return x

    pred_C0 = _to_pred(seg_C0)
    pred_LGE = _to_pred(seg_LGE)
    pred_T2 = _to_pred(seg_T2)

    is_scar = (pred_LGE == 2)
    is_edema = (pred_T2 == 2) & (~is_scar)
    is_myo = (pred_C0 == 1) & (~is_scar) & (~is_edema)

    canonical = torch.zeros_like(pred_C0, dtype=torch.uint8)
    canonical[is_myo] = 1
    canonical[is_edema] = 2
    canonical[is_scar] = 3

    return canonical


# Backward compatibility alias
myops_outputs_to_canonical = lambda seg_dict: fuse_predictions(
    seg_dict.get("cardiac", seg_dict.get("C0")),
    seg_dict.get("scar", seg_dict.get("LGE")),
    seg_dict.get("edema", seg_dict.get("T2")),
)


class Normalization(object):
    def __call__(self, image, mode):
        if mode == 'Max_Min':
            eps = 1e-8
            mn = image.min()
            mx = image.max()
            image_normalized = (image - mn) / (mx - mn + eps)
        
        if mode == 'Zero_Mean_Unit_Std':
            eps = 1e-8
            mean = image.mean()
            std = image.std()
            image_normalized = (image - mean) / (std + eps)

        if mode == 'Truncate':
            Hist, _ = np.histogram(image, bins=int(image.max()))
            idexs = np.argwhere(Hist >= 20)
            idex_min = np.float32(0)
            idex_max = np.float32(idexs[-1, 0])

            image[np.where(image <= idex_min)] = idex_min
            image[np.where(image >= idex_max)] = idex_max

            sig = image[0, 0, 0]
            image = np.where(image != sig, image - np.mean(image[image != sig]), 0 * image)
            image_normalized = np.where(image != sig, image / np.std(image[image != sig] + 1e-8), 0 * image)
        
        return image_normalized


class RandomSizedCrop(object):
    def __init__(self, dim):
        self.crop_size = dim

    def __call__(self, image):
        scaler = np.random.uniform(0.9, 1.1)
        scale_size = int(self.crop_size * scaler)
        h_off = random.randint(0, image.shape[1] - 0 - scale_size)
        w_off = random.randint(0, image.shape[2] - 0 - scale_size)
        image = image[:, h_off:h_off+scale_size, w_off:w_off+scale_size]

        image = image.numpy() 

        C0_slice = image[:1,...]
        LGE_slice = image[1:2,...]
        T2_slice = image[2:3,...]
        T1m_slice = image[3:4,...]
        T2starm_slice = image[4:5,...]
        label_slice = image[5:,...]
        
        output_shape = (1, self.crop_size, self.crop_size)

        C0_resized = resize(C0_slice, output_shape, order=1, mode='constant', preserve_range=True)
        LGE_resized = resize(LGE_slice, output_shape, order=1, mode='constant', preserve_range=True)
        T2_resized = resize(T2_slice, output_shape, order=1, mode='constant', preserve_range=True)
        T1m_resized = resize(T1m_slice, output_shape, order=1, mode='constant', preserve_range=True)
        T2starm_resized = resize(T2starm_slice, output_shape, order=1, mode='constant', preserve_range=True)
        label_resized = resize(label_slice, output_shape, order=0, mode='edge', preserve_range=True)

        image = np.concatenate([C0_resized, LGE_resized, T2_resized, T1m_resized, T2starm_resized, label_resized], axis=0)
        image = torch.from_numpy(image).float()    
        return image


class ToTensor(object):
    def __call__(self, image):
        image = np.ascontiguousarray(image.transpose(2, 0, 1))
        image = torch.from_numpy(image).float()
        return image


class ImageTransform(object):
    def __init__(self, dim, stage):  
        self.dim = dim
        self.stage = stage

    def __call__(self, image):
        if self.stage == 'Train':
            transform = transforms.Compose([
                ToTensor(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(180),
                RandomSizedCrop(self.dim)
            ])     
        
        if self.stage == 'Valid' or self.stage == 'Test':
            transform = transforms.Compose([
                ToTensor(),
                transforms.CenterCrop(self.dim)
            ])

        return transform(image)


class LabelTransform(object):
    def __init__(self, stage):
        self.stage = stage

    def __call__(self, label):
        label = label.numpy()
        cardiac_gd = self.label_transform(label, 'cardiac')
        scar_gd = self.label_transform(label, 'scar')
        edema_gd = self.label_transform(label, 'edema')
        pathology_gd = self.label_transform(label, 'pathology')

        if self.stage == 'Train':
            cardiac_gd = self.convert_onehot(cardiac_gd, 3)
            scar_gd = self.convert_onehot(scar_gd, 3)
            edema_gd = self.convert_onehot(edema_gd, 3)
            pathology_gd = self.convert_onehot(pathology_gd, 3)

        return cardiac_gd, scar_gd, edema_gd, pathology_gd

    def convert_onehot(self, label, num_class):
        label = label.long()
        label_onehot = torch.zeros((num_class, label.shape[1], label.shape[2]))
        label_onehot.scatter_(0, label, 1).float()
        return label_onehot 

    def label_transform(self, label, mode):
        if mode == 'cardiac':
            label = np.where(label == 200, 1, label)
            label = np.where(label == 500, 2, label)
            label = np.where(label == 600, 0, label)
            label = np.where(label == 1220, 1, label)
            label = np.where(label == 2221, 1, label)
    
        if mode == 'scar':
            label = np.where(label == 200, 1, label)
            label = np.where(label == 500, 0, label)
            label = np.where(label == 600, 0, label)
            label = np.where(label == 1220, 1, label)
            label = np.where(label == 2221, 2, label)
    
        if mode == 'edema':
            label = np.where(label == 200, 1, label)
            label = np.where(label == 500, 0, label)
            label = np.where(label == 600, 0, label)
            label = np.where(label == 1220, 2, label)
            label = np.where(label == 2221, 2, label)
    
        if mode == 'pathology':
            label = np.where(label == 200, 0, label)
            label = np.where(label == 500, 0, label)
            label = np.where(label == 600, 0, label)
            label = np.where(label == 1220, 1, label)
            label = np.where(label == 2221, 2, label)    
        
        label = torch.from_numpy(label).float()
        return label


class ResultTransform(object):
    def __init__(self, ToOriginal = False):      
        self.flag = ToOriginal

    def __call__(self, seg_scar_LGE, seg_scar_mapping, seg_edema):
        seg_scar_LGE = seg_scar_LGE.numpy()
        seg_scar_mapping = seg_scar_mapping.numpy()
        seg_edema = seg_edema.numpy()

        seg_scar_LGE = np.where(seg_scar_LGE == 1, 0, seg_scar_LGE)
        seg_scar_mapping = np.where(seg_scar_mapping == 1, 0, seg_scar_mapping)
        seg_scar = seg_scar_LGE + seg_scar_mapping
        seg_scar = np.where(seg_scar == 4, 2, seg_scar)

        seg_edema = np.where(seg_edema == 1, 0, seg_edema)
        seg_edema = np.where(seg_edema == 2, 1, seg_edema)
        
        seg_pathology = seg_scar + seg_edema
        seg_pathology = np.where(seg_pathology == 3, 2, seg_pathology)

        if self.flag == True:
            seg_pathology = np.where(seg_pathology == 1, 1220, seg_pathology)
            seg_pathology = np.where(seg_pathology == 2, 2221, seg_pathology)
        
        seg_pathology = torch.from_numpy(seg_pathology)
        return seg_pathology
