# Project: MyoPS-Net MyoPS380 Adaptation (`myops380_train304_test76_last_v1`)

## Architecture
MyoPS-Net is adapted to the MyoPS380 benchmark protocol (`myops380_train304_test76_last_v1`) matching the reference implementation in SCAR:
- **Input Data Pipeline**: Loads 3-modality (bSSFP, LGE, T2w) aligned slices/volumes from `D:\nckh_model\Dataset\MyoPS380\Processed_data`. Applies zero-padding for 2 missing modalities (T1m, T2starm) to satisfy MyoPS-Net's 5-branch architecture. Applies SCAR's exact `RandomGenerator` augmentation.
- **Target Adapter & Loss**: Maps canonical 4-class labels (0:BG, 1:Normal Myo, 2:Edema, 3:Scar) to 3 target dictionaries (`cardiac`, `scar`, `edema`) compatible with native `MyoPSLoss`. Hardens `WCELoss` against class-absence dimensional mismatches and zero division.
- **Training Engine**: Fixed 300 epochs, no validation (`no_val=True`), effective batch size 16 (microbatching with sample-weighted accumulation), AdamW (`weight_decay=1e-4`), step-level Polynomial LR Decay (power=0.9), seed=1234, saving atomic `last.pth`.
- **Inference & Evaluation**: Single-pass forward + argmax on 76 test volumes from `test_vol.txt`, morphological post-processing removed, exports 3D uint8 `(128, 128, D)` predictions to `test_predictions/{CaseID}_pred.npz` (key `'prediction'`), evaluated via SCAR's shared evaluator (`surface_distance.py`, protocol `myops380_voxel_v3`), producing `metrics.json` and `per_case.csv`.

---

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| F1 | Data Pipeline & Manifests | Load 304 train patients (1,972 slices) from SCAR `train.txt` and 76 test patients from `test_vol.txt`; zero patient overlap | M1 | Promt_train_test_no_val.txt §1, SCAR splits |
| F2 | Normalization & Zero-Padding | Ingest `[0.0, 1.0]` cache directly without redundant /255; zero-pad T1m and T2starm to form 5 branches | M1 | Promt_train_test_no_val.txt §1, MyoPS-Net architecture |
| F3 | SCAR RandomGenerator | Replicate synchronized spatial transforms (rot/flip, rotate [-20, 20]°, shift [-5, 5]px) and independent intensity transforms (brightness, gamma, Gaussian noise) resized to (128, 128) | M1 | Promt_train_test_no_val.txt §2, SCAR myops_dataset.py |
| F4 | Validation Elimination | Completely eliminate validation loader, validation dataset, and per-epoch validation loop | M1 | Promt_train_test_no_val.txt §1, SCAR no_val protocol |
| F5 | Target Adapter | Convert canonical 4 classes (0:BG, 1:Normal Myo, 2:Edema, 3:Scar) to 3 targets (`cardiac`, `scar`, `edema`) for `MyoPSLoss`, and reverse mapping for predictions | M2 | Promt_train_test_no_val.txt §3, MyoPS-Net loss.py |
| F6 | WCELoss Bug Fix | Fix zero-division and tensor size mismatch in `WCELoss.weight_function` for absent classes, respecting target device | M2 | Explorer & Spec Miner findings, network/criterion/loss.py |
| F7 | Optimizer & Scheduler | AdamW (weight_decay=1e-4, 0.0 for 1D/norm/bias), step-level Polynomial LR Decay (power=0.9), seed=1234, deterministic flags | M3 | Promt_train_test_no_val.txt §4, SCAR trainer.py |
| F8 | Batch Accumulation & Checkpoint | Target effective batch size 16 with sample-weighted accumulation, save atomic `last.pth` at epoch 300, no best checkpoint | M3 | Promt_train_test_no_val.txt §4-5, SCAR trainer.py |
| F9 | Test Inference Pipeline | Single-pass argmax without morphological post-processing, export `test_predictions/{CaseID}_pred.npz` (uint8, 128x128xD, key 'prediction') | M4 | Promt_train_test_no_val.txt §6, MyoPS-Net predict.py |
| F10 | Shared Evaluator | Shared evaluator calling SCAR `surface_distance.py` (`dataset_rows`, `summarize_dataset_rows`, `myops380_voxel_v3`), export `metrics.json`, `per_case.csv` | M4 | Promt_train_test_no_val.txt §6, SCAR surface_distance.py |
| F11 | End-to-End Verification | Smoke test training pass (multi-batch forward/backward/accum/checkpoint) and test inference + evaluation dry run on sample volume | M5 | Acceptance Criteria Verification Scope |

---

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | Data Pipeline & Augmentation | Implement MyoPS380 dataset loader with zero-padding, SCAR `RandomGenerator`, manifest integration, no validation | none | DONE |
| M2 | Target Adapter & Loss Hardening | Implement canonical 4-class to `cardiac`/`scar`/`edema` target adapter and fix `WCELoss` absent-class bug | none | DONE |
| M3 | Training Engine & Checkpointing | Configure AdamW, PolyLR, sample-weighted accumulation (eff bs=16), 300 epochs, atomic `last.pth` checkpoint | M1, M2 | DONE |
| M4 | Inference & Shared Evaluator | Implement postprocess-free test volume inference, `.npz` generation, and shared `surface_distance.py` evaluation | M1, M2 | DONE |
| M5 | E2E Integration & Verification | Execute end-to-end smoke test training run and inference/evaluation dry run; verify 100% test pass | M3, M4 | DONE |

---

## Interface Contracts

### M1 ↔ M3 (Dataset/DataLoader ↔ Training Engine)
- **DataLoader batch output**: Dictionary containing:
  - `'C0'`: Tensor `(B, 1, 128, 128)` float32 in $[0.0, 1.0]$ (bSSFP)
  - `'LGE'`: Tensor `(B, 1, 128, 128)` float32 in $[0.0, 1.0]$ (LGE)
  - `'T2'`: Tensor `(B, 1, 128, 128)` float32 in $[0.0, 1.0]$ (T2w)
  - `'T1m'`: Tensor `(B, 1, 128, 128)` float32 zeros (zero-padded)
  - `'T2starm'`: Tensor `(B, 1, 128, 128)` float32 zeros (zero-padded)
  - `'label'`: Tensor `(B, 128, 128)` int64 with canonical values $\{0, 1, 2, 3\}$
  - `'case_id'`: List of strings (e.g. `['case0002_slice000', ...]`)

### M2 ↔ M3 (Target Adapter ↔ Training Engine / Loss)
- **Function**: `canonical_to_myops_targets(label: torch.Tensor) -> dict[str, torch.Tensor]`
  - Input: `label` tensor `(B, 128, 128)` with values $0..3$.
  - Output dictionary:
    - `'scar'`: Tensor `(B, 3, 128, 128)` one-hot: 0=BG, 1=Non-scar Myo (1+2), 2=Scar (3)
    - `'edema'`: Tensor `(B, 3, 128, 128)` one-hot: 0=BG, 1=Normal Myo (1), 2=Edema inclusive / AAR (2+3)
    - `'cardiac'`: Tensor `(B, 3, 128, 128)` one-hot: 0=BG, 1=Total Myo (1+2+3), 2=zeros
- **Loss call**: `loss_val, loss_dict = criterion(model_outputs, targets_dict)`
  - Where `model_outputs = {'cardiac': seg_C0, 'scar': seg_LGE, 'edema': seg_T2, 'mapping': seg_mapping}`.
  - Hardened `WCELoss.weight_function` handles any absent class without division by zero or shape mismatch.

### M2 / M3 ↔ M4 (Model / Target Adapter ↔ Test Inference)
- **Function**: `myops_outputs_to_canonical(seg_outputs: dict[str, torch.Tensor]) -> torch.Tensor`
  - Converts model branch outputs (`cardiac`, `scar`, `edema`) into mutually exclusive canonical 4-class predictions $\{0, 1, 2, 3\}$.
  - Input: raw logits from model forward pass.
  - Output: `(B, 128, 128)` int64 tensor $\{0, 1, 2, 3\}$.

### M4 ↔ Evaluator
- **Test Output**: `test_predictions/{CaseID}_pred.npz`
  - Key `'prediction'`: numpy array `(128, 128, D)`, dtype `uint8`, values $\{0, 1, 2, 3\}$.
- **Evaluator Call**: `dataset_rows(prediction, target, case_id, dataset_id="myops380")` and `summarize_dataset_rows(rows, dataset_id="myops380")`
  - Output: `metrics.json` and `per_case.csv`.

---

## Code Layout
```
D:\nckh_model\MyoPS-Net\
├── dataset/
│   ├── myops380_dataset.py       # New MyoPS380 dataset and volume loader
│   ├── data_augmentation.py      # SCAR RandomGenerator augmentation
│   ├── target_adapter.py         # Canonical 4-class <-> 3-target adapter
│   └── ...
├── network/
│   ├── model.py                  # MyoPSNet architecture (kept intact)
│   └── criterion/
│       └── loss.py               # MyoPSLoss with hardened WCELoss
├── train.py                      # Training script with AdamW, PolyLR, accum, last.pth
├── evaluate.py                   # Test inference and shared surface_distance evaluation
├── tests/                        # E2E test suite (Tiers 1-4)
│   ├── test_data_pipeline.py
│   ├── test_target_adapter.py
│   ├── test_training_dynamics.py
│   └── test_inference_eval.py
└── test_predictions/             # Output directory for test .npz predictions
```
