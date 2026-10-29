# Testing Infrastructure Documentation (`TEST_INFRA.md`)
## MyoPS-Net Adaptation for MyoPS380 (`myops380_train304_test76_last_v1`)

### 1. Test Architecture & File Layout
The test suite resides exclusively in `tests/` and is designed as an opaque-box, multi-tier automated test harness.

```
tests/
├── __init__.py
├── contract_helpers.py         # Authoritative reference oracles, manifest hashes, dynamic module loaders
├── test_data_pipeline.py       # F1 (Manifests), F2 (Normalization & Zero-Padding), F3 (SCAR RandomGenerator), F4 (No Validation)
├── test_target_adapter.py      # F5 (Target Adapter canonical 4-class <-> 3-target dictionary & reverse mapping)
├── test_loss_hardening.py      # F6 (Hardened WCELoss, DiceLoss, InvariantLoss, InclusiveLoss, MyoPSLoss)
├── test_training_dynamics.py   # F7 (AdamW & PolyLR scheduler, determinism), F8 (Batch accumulation, last.pth checkpoint)
├── test_inference_eval.py      # F9 (Inference pipeline, no postprocessing), F10 (Shared evaluator), F11 (End-to-end verification)
└── run_all_tests.py            # Master test runner with tier and feature summary reporting
```

---

### 2. Four-Tier Testing Methodology

| Tier | Category | Description | Test Count | Status |
|:---|:---|:---|:---:|:---:|
| **Tier 1** | **Feature Coverage** | >=5 test cases per feature covering primary behavior (happy path) for F1 through F11 | **55** | PASSED |
| **Tier 2** | **Boundary & Corner Cases** | >=5 test cases per feature covering edge cases, absent classes in masks, zero/max intensities, NaN/Inf checks, single-slice batches, accumulation boundaries, and empty predictions vs empty targets | **64** | PASSED |
| **Tier 3** | **Cross-Feature Combinations** | Multi-module interaction tests: data pipeline + target adapter, target adapter + MyoPSLoss, optimizer + PolyLR + gradient accumulation, model forward + target adapter reverse + inference + evaluator | **6** | PASSED |
| **Tier 4** | **Real-World Application Scenarios** | Full end-to-end training smoke pass (multi-batch forward/backward/accum/checkpoint) and test volume inference + shared evaluation dry run on sample volume | **2** | PASSED |
| **Total** | | **Comprehensive opaque-box verification** | **127** | **ALL PASSED** |

---

### 3. Feature Coverage Matrix (F1 – F11)

| # | Feature | Scope & Specification | Associated Test File | Tests |
|:---|:---|:---|:---|:---:|
| **F1** | Data Pipeline & Manifests | 304 train patients (1,972 slices) from SCAR `train.txt`, 76 test volumes from `test_vol.txt`, zero patient overlap, SHA256 integrity, configurable data path | `tests/test_data_pipeline.py` | 10 |
| **F2** | Normalization & Zero-Padding | Ingest `[0.0, 1.0]` cache directly without redundant /255 division; zero-pad T1m and T2starm to form 5-branch input | `tests/test_data_pipeline.py` | 10 |
| **F3** | SCAR RandomGenerator | Synchronized spatial transforms (flip/rot90, rotate $\pm 20^\circ$, shift $\pm 5$ px) and independent intensity transforms (brightness, gamma, noise) resized to (128, 128) | `tests/test_data_pipeline.py` | 10 |
| **F4** | Validation Elimination | Eliminate validation loader, validation dataset, and per-epoch validation loop (`no_val=True`, 300 epochs) | `tests/test_data_pipeline.py` | 10 |
| **F5** | Target Adapter | Convert canonical 4 classes (0:BG, 1:Normal Myo, 2:Edema, 3:Scar) to 3 targets (`cardiac`, `scar`, `edema`) for `MyoPSLoss`, and reverse mapping for predictions | `tests/test_target_adapter.py` | 15 |
| **F6** | WCELoss Bug Fix & Hardening | Harden `WCELoss.weight_function` against absent classes (prevent shape mismatch), zero-division, and device inflexibility | `tests/test_loss_hardening.py` | 14 |
| **F7** | Optimizer & Scheduler | AdamW (wd=1e-4 for weights, 0.0 for 1D/norm/bias), step-level Polynomial LR Decay (power=0.9), seed=1234, determinism | `tests/test_training_dynamics.py` | 10 |
| **F8** | Batch Accumulation & Checkpoint | Effective batch size 16 with sample-weighted accumulation, save atomic `last.pth` at epoch 300, no best checkpoint | `tests/test_training_dynamics.py` | 10 |
| **F9** | Test Inference Pipeline | Single-pass argmax without morphological post-processing, export `test_predictions/{CaseID}_pred.npz` (uint8, 128x128xD, key 'prediction') | `tests/test_inference_eval.py` | 10 |
| **F10** | Shared Evaluator | Shared evaluator calling SCAR `surface_distance.py` (`dataset_rows`, `summarize_dataset_rows`, `myops380_voxel_v3`), export `metrics.json`, `per_case.csv` | `tests/test_inference_eval.py` | 10 |
| **F11** | End-to-End Verification | Multi-batch forward/backward/accum/checkpoint smoke pass and inference + evaluation dry run on real/synthetic volumes | `tests/test_inference_eval.py` | 10 |
| **System** | Cross-Module / System Integration | Multi-stage pipeline consistency and end-to-end dataflow round-trips | All test modules | 8 |

---

### 4. Authoritative Output Derivation
All expected outputs and assertions in the test suite are derived from:
1. **Reference Program (Oracle)**: `D:\nckh_model\SCAR\training\metrics\surface_distance.py` (`dataset_rows` and `summarize_dataset_rows`) for all metric calculations and empty-mask policy validations (`status="ok"`, `"both_empty"`, `"prediction_empty"`).
2. **Reference Split Manifests**: `D:\nckh_model\SCAR\preprocessing\splits\myops380_no_val\` (`train.txt`, `test_vol.txt`, `split_metadata.json`) for slice and patient count verifications and SHA256 integrity.
3. **Reference Augmentation**: `D:\nckh_model\SCAR\training\dataset\myops_dataset.py` (`RandomGenerator`) for spatial and intensity transform ranges.
4. **Mathematical Contracts**: `PROJECT.md` § Interface Contracts for Target Adapter mappings and sample-weighted gradient accumulation.

---

### 5. How to Run the Tests

#### Running the Master Test Suite
```powershell
python tests/run_all_tests.py
```
*Expected observable behavior*: Runs all 127 tests, outputs progress dots, prints full execution summary table categorized by Tier and Feature, and exits with code 0.

#### Running Individual Test Modules
```powershell
python -m unittest tests/test_data_pipeline.py
python -m unittest tests/test_target_adapter.py
python -m unittest tests/test_loss_hardening.py
python -m unittest tests/test_training_dynamics.py
python -m unittest tests/test_inference_eval.py
```

---

### 6. Implementation Defect Log (For Implementing Agents)

| Defect ID | Component | Location | Issue Description | Required Fix |
|:---|:---|:---|:---|:---|
| **BUG-01** | Loss Function | `network/criterion/loss.py:64-68` | `WCELoss.weight_function` loops over `range(mask.max() + 1)`. When class 2 (scar/edema) is absent from batch, `weights` has length 2 while `ce_loss` has length 3, crashing with `RuntimeError: The size of tensor a (2) must match the size of tensor b (3)`. | Iterate over `range(target.shape[1])`. Assign weight 0.0 when class voxel count is 0. |
| **BUG-02** | Loss Function | `network/criterion/loss.py:66` | When a class has 0 voxels, `np.log(voxels_sum / voxels_i)` causes `ZeroDivisionError` or Inf/NaN. | Handle `voxels_i == 0` safely with weight 0.0 or epsilon. |
| **BUG-03** | Loss Function | `network/criterion/loss.py:68` | Hardcoded `.cuda()` call crashes when running on CPU or custom devices. | Use `weights = weights.to(target.device)`. |
| **BUG-04** | Entrypoint | `main.py:9` | `from train import CrossModalSegNetTrain` raises `ImportError: cannot import name 'CrossModalSegNetTrain' from 'train'` because `train.py` defines `MyoPSNetTrain`. | Harmonize function import in `main.py` with `train.py`. |
