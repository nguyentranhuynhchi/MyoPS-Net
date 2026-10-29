# MyoPS-Net
PyTorch implemetation for our MedIA paper "MyoPS-Net: Myocardial Pathology Segmentation with Flexible Combination of Multi-Sequence CMR Images" (https://doi.org/10.1016/j.media.2022.102694).

<img decoding="async" src="structure.png">

## Introduction
Myocardial pathology segmentation (MyoPS) can be a prerequisite for the accurate diagnosis and treatment planning of myocardial infarction. However, achieving this segmentation is challenging, mainly due to the inadequate and indistinct information from an image. In this work, we develop an end-to-end deep neural network, referred to as MyoPS-Net, to flexibly combine five-sequence cardiac magnetic resonance (CMR) images for MyoPS. 

The proposed MyoPS-Net were evaluated on two datasets, i.e., a private one consisting of 50 paired multi-sequence CMR images and a public one from MICCAI2020 MyoPS Challenge (https://zmiclab.github.io/zxh/0/myops20/). Experimental results showed that MyoPS-Net could achieve state-of-the-art performance in various scenarios. Note that in practical clinics, the subjects may not have full sequences, such as missing LGE CMR or mapping CMR scans. We therefore conducted extensive experiments to investigate the performance of the proposed method in dealing with such complex combinations of different CMR sequences. Results proved the superiority and generalizability of MyoPS-Net, and more importantly, indicated a practical clinical application.

## Usage
### Dataset
Please organize the dataset as the following structure:
```
data/
  -- train_set/
     -- train_image/
     -- train_gd/
  -- val_set/     
     -- val_image/
     -- val_gd/
  -- test_set/
     -- test_image/
  -- train.txt (with each line: image_path gd_path z_index)
  -- validation.txt (with each line: image_path gd_path z_index)
  -- test.csv (with each line: image_path stage dx dy dz)
```
### Train
```
python main.py --path "data_path" --batch_size 16 --dim 192 --lr 1e-4 --threshold 0.50 --end_epoch 200
```

### Predict
```
python predict.py --load_path checkpoints/xxx.pth --predict_mode single --threshold 0.50 --dim 192
```

## MyoPS380 Benchmark Adaptation (`myops380_train304_test76_last_v1`)

This repository has been adapted to the controlled benchmark protocol matching **SCAR** (`D:\nckh_model\SCAR`) on the **MyoPS380** dataset:
- **Protocol**: `myops380_train304_test76_last_v1` (304 train patients with 1,972 slices; 76 test patient volumes; no validation).
- **Data Augmentation**: SCAR's exact `RandomGenerator` on 3 aligned modalities (bSSFP, LGE, T2w) at 128x128.
- **5-Branch Input Support**: Zero-padding missing T1m and T2starm channels to preserve MyoPS-Net's multi-encoder architecture.
- **Objective**: Native `MyoPSLoss` (Dice + Weighted Cross-Entropy + Invariant + Inclusive) with hardened `WCELoss` and canonical 4-class `TargetAdapter`.
- **Training Budget**: 300 epochs, AdamW (`weight_decay=1e-4`), step-level Polynomial LR Decay (power=0.9), effective batch size 16 (sample-weighted gradient accumulation), `seed=1098`.
- **Checkpointing**: Atomic save of `last.pth` at final epoch 300 (no best checkpoint selection, no early stopping).
- **Evaluation**: Single-pass forward inference without morphological post-processing, export compressed `.npz` predictions, evaluated via SCAR's shared evaluator (`surface_distance.py`, protocol `myops380_voxel_v3`).

### 1. Training Command (300 Epochs, No-Val)
```bash
python main.py train \
    --data_root "D:\nckh_model\Dataset\MyoPS380\Processed_data" \
    --split_dir "dataset/splits/myops380_no_val" \
    --save_dir "checkpoints" \
    --epochs 300 \
    --batch_size 4 \
    --accum_steps 4 \
    --lr 1e-3 \
    --seed 1098
```

### 2. Resume Training
```bash
python main.py train \
    --data_root "D:\nckh_model\Dataset\MyoPS380\Processed_data" \
    --split_dir "D:\nckh_model\SCAR\preprocessing\splits\myops380_no_val" \
    --save_dir "checkpoints" \
    --resume
```

### 3. Evaluation on 76 Test Volumes
```bash
python main.py evaluate \
    --checkpoint "checkpoints/last.pth" \
    --data_root "D:\nckh_model\Dataset\MyoPS380\Processed_data" \
    --split_dir "D:\nckh_model\SCAR\preprocessing\splits\myops380_no_val" \
    --output_dir "test_predictions"
```

### 4. Running Test Suites
```bash
# Run unit & contract test suite (127 tests)
python -m tests.run_all_tests

# Run real pipeline end-to-end smoke test
python -m tests.test_smoke_real_pipeline
```

---

## Citation and Acknowledge
If you make use of the code, or if you found the code useful, please cite this paper in any resulting publications.
```
@article{MyoPS-Net,
title = {MyoPS-Net: Myocardial pathology segmentation with flexible combination of multi-sequence CMR images},
journal = {Medical Image Analysis},
volume = {84},
pages = {102694},
year = {2023},
issn = {1361-8415},
doi = {https://doi.org/10.1016/j.media.2022.102694},
author = {Junyi Qiu and Lei Li and Sihan Wang and Ke Zhang and Yinyin Chen and Shan Yang and Xiahai Zhuang}
}
```
J. Qiu, L. Li, S. Wang et al., MyoPS-Net: Myocardial pathology segmentation with flexible combination of multi-sequence CMR images. Medical Image Analysis (2022), doi: https://doi.org/10.1016/j.media.2022.102694.

