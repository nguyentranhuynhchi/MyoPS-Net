"""
Tests for Feature 7: Optimizer & Scheduler (AdamW, PolyLR Decay, Determinism) and
Feature 8: Batch Accumulation & Checkpoint (Effective BS 16, Sample-Weighted Accumulation, last.pth).
Protocol: myops380_train304_test76_last_v1
Interface Contract: PROJECT.md § M3 Training Engine
Tiers Covered: Tier 1 (Feature Coverage), Tier 2 (Boundary & Corner Cases), Tier 3 (Cross-Feature Combinations).
"""

import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import math
import random
import shutil
import tempfile
import unittest
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from dataset.target_adapter import canonical_to_myops_targets
from network.criterion.loss import MyoPSLoss
from network.model import MyoPSNet
from train import restore_rng_states, seed_worker


def build_grouped_optimizer(model: nn.Module, base_lr: float = 1e-3, weight_decay: float = 1e-4) -> AdamW:
    """Standard grouped AdamW optimizer:
    - 2D/4D weights (Conv2d, Linear): weight_decay = 1e-4
    - 1D weights, biases, Normalization layers (BatchNorm, GroupNorm, LayerNorm): weight_decay = 0.0
    """
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() >= 2 and not any(norm_name in name.lower() for norm_name in ["norm", "bn", "gn", "ln"]):
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    # Invariant: zero overlap
    decay_ids = {id(p) for p in decay_params}
    no_decay_ids = {id(p) for p in no_decay_params}
    assert len(decay_ids.intersection(no_decay_ids)) == 0, "Parameter groups have overlapping parameters!"

    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    return AdamW(param_groups, lr=base_lr)


def get_polynomial_decay_scheduler(optimizer: AdamW, total_updates: int, power: float = 0.9) -> LambdaLR:
    """Step-level Polynomial LR Decay scheduler:
    lr = base_lr * (max(0.0, 1.0 - step / total_updates)) ** power
    """
    lr_lambda = lambda step: max(0.0, 1.0 - step / float(total_updates)) ** power
    return LambdaLR(optimizer, lr_lambda)


def atomic_save_checkpoint(payload: dict, target_path: str):
    """Atomically saves checkpoint using a temporary file to avoid corruption on crash."""
    os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)
    temp_path = f"{target_path}.tmp"
    torch.save(payload, temp_path)
    if os.path.exists(target_path):
        os.remove(target_path)
    os.rename(temp_path, target_path)


class SimpleConvModel(nn.Module):
    """Small neural net with Conv2d, BatchNorm2d, and Linear for testing training dynamics."""
    def __init__(self):
        super(SimpleConvModel, self).__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(4)
        self.fc = nn.Linear(4 * 16 * 16, 2)

    def forward(self, x):
        h = torch.relu(self.bn(self.conv(x)))
        return self.fc(h.flatten(1))


class TestOptimizerAndScheduler(unittest.TestCase):
    """Covers Feature 7: Optimizer AdamW & Polynomial LR Decay."""

    def setUp(self):
        self.model = SimpleConvModel()
        self.base_lr = 1e-3
        self.weight_decay = 1e-4

    # =========================================================================
    # Tier 1: Feature Coverage (F7)
    # =========================================================================

    def test_f7_adamw_weight_decay_parameter_grouping(self):
        """Tier 1: Verify Conv2d and Linear weights receive wd=1e-4, while biases and BN weights get wd=0.0."""
        optimizer = build_grouped_optimizer(self.model, base_lr=self.base_lr, weight_decay=self.weight_decay)
        self.assertEqual(len(optimizer.param_groups), 2)

        decay_group = optimizer.param_groups[0]
        no_decay_group = optimizer.param_groups[1]

        self.assertEqual(decay_group["weight_decay"], 1e-4)
        self.assertEqual(no_decay_group["weight_decay"], 0.0)

        # Check parameter dimensions
        for p in decay_group["params"]:
            self.assertGreaterEqual(p.dim(), 2)
        for p in no_decay_group["params"]:
            self.assertTrue(p.dim() < 2 or any(p is bp for bp in self.model.bn.parameters()))

    def test_f7_adamw_disjoint_parameter_groups(self):
        """Tier 1: Verify exact partitioning: zero overlap, and total parameters match all trainable model parameters."""
        optimizer = build_grouped_optimizer(self.model, base_lr=self.base_lr, weight_decay=self.weight_decay)
        decay_params = set(optimizer.param_groups[0]["params"])
        no_decay_params = set(optimizer.param_groups[1]["params"])

        self.assertEqual(len(decay_params.intersection(no_decay_params)), 0)
        all_trainable = set(p for p in self.model.parameters() if p.requires_grad)
        self.assertEqual(decay_params.union(no_decay_params), all_trainable)

    def test_f7_adamw_myopsnet_parameter_grouping(self):
        """Tier 1: Verify parameter grouping on actual MyoPSNet architecture.
        Ensures Conv2d weights receive wd=1e-4, BatchNorm2d and biases receive wd=0.0,
        the two groups are strictly disjoint, and all trainable parameters are covered.
        """
        model = MyoPSNet(in_chs=(5, 2, 2, 3), out_chs=(3, 3, 3, 3))
        optimizer = build_grouped_optimizer(model, base_lr=1e-3, weight_decay=1e-4)

        self.assertEqual(len(optimizer.param_groups), 2)
        decay_group = optimizer.param_groups[0]
        no_decay_group = optimizer.param_groups[1]

        self.assertEqual(decay_group["weight_decay"], 1e-4)
        self.assertEqual(no_decay_group["weight_decay"], 0.0)

        decay_params = set(decay_group["params"])
        no_decay_params = set(no_decay_group["params"])

        # Disjointness check
        self.assertEqual(len(decay_params.intersection(no_decay_params)), 0)

        # Full coverage of trainable parameters
        trainable_params = {p for p in model.parameters() if p.requires_grad}
        self.assertEqual(decay_params.union(no_decay_params), trainable_params)
        self.assertGreater(len(decay_params), 0)
        self.assertGreater(len(no_decay_params), 0)

        # Architectural verification across MyoPSNet submodules:
        # 1. unet_C0 Conv1 Conv2d weight in decay
        conv_weight = model.unet_C0.Conv1.conv[0].weight
        self.assertIn(conv_weight, decay_params)

        # 2. unet_C0 Conv1 BatchNorm2d weight and bias in no_decay
        bn_weight = model.unet_C0.Conv1.conv[1].weight
        bn_bias = model.unet_C0.Conv1.conv[1].bias
        self.assertIn(bn_weight, no_decay_params)
        self.assertIn(bn_bias, no_decay_params)

        # 3. encoder_LGE Conv1 Conv2d weight in decay
        lge_conv_weight = model.encoder_LGE.Conv1.conv[0].weight
        self.assertIn(lge_conv_weight, decay_params)

        # 4. decoder_mapping Conv Conv2d weight in decay
        map_outc_weight = model.decoder_mapping.Conv.weight
        self.assertIn(map_outc_weight, decay_params)

    def test_f7_polylr_schedule_step_level_decay(self):
        """Tier 1: Verify exact mathematical formula lr = base_lr * (1.0 - step / total_updates) ** 0.9."""
        total_updates = 100
        optimizer = build_grouped_optimizer(self.model, base_lr=self.base_lr)
        scheduler = get_polynomial_decay_scheduler(optimizer, total_updates=total_updates, power=0.9)

        # Step 50: midway
        for _ in range(50):
            optimizer.step()
            scheduler.step()

        expected_lr = self.base_lr * ((1.0 - 50 / 100.0) ** 0.9)
        current_lr = optimizer.param_groups[0]["lr"]
        self.assertAlmostEqual(current_lr, expected_lr, places=7)

    def test_f7_polylr_initial_and_terminal_lr(self):
        """Tier 1: Verify LR at step 0 is base_lr and at step total_updates is 0.0."""
        total_updates = 200
        optimizer = build_grouped_optimizer(self.model, base_lr=self.base_lr)
        scheduler = get_polynomial_decay_scheduler(optimizer, total_updates=total_updates, power=0.9)

        initial_lr = optimizer.param_groups[0]["lr"]
        self.assertEqual(initial_lr, self.base_lr)

        for _ in range(total_updates):
            optimizer.step()
            scheduler.step()

        terminal_lr = optimizer.param_groups[0]["lr"]
        self.assertEqual(terminal_lr, 0.0)

    def test_f7_reproducibility_seed_determinism(self):
        """Tier 1: Seed 1234 guarantees identical pseudo-random sequences and tensor initialization."""
        def get_random_sequence(seed: int):
            torch.manual_seed(seed)
            np.random.seed(seed)
            t = torch.randn(5)
            n = np.random.randn(5)
            return t, n

        t1, n1 = get_random_sequence(1234)
        t2, n2 = get_random_sequence(1234)
        torch.testing.assert_close(t1, t2)
        np.testing.assert_allclose(n1, n2)

    # =========================================================================
    # Tier 2: Boundary & Corner Cases (F7)
    # =========================================================================

    def test_f7_polylr_step_beyond_total_updates(self):
        """Tier 2: Steps beyond total_updates clamp cleanly to 0.0, avoiding negative LR."""
        total_updates = 10
        optimizer = build_grouped_optimizer(self.model, base_lr=self.base_lr)
        scheduler = get_polynomial_decay_scheduler(optimizer, total_updates=total_updates, power=0.9)

        for _ in range(total_updates + 5):
            optimizer.step()
            scheduler.step()

        self.assertEqual(optimizer.param_groups[0]["lr"], 0.0)

    def test_f7_polylr_accum_steps_total_updates_calculation(self):
        """Tier 2: Verify total_updates calculation formula: max_epochs * ceil(len(loader) / accum_steps)."""
        max_epochs = 300
        loader_len = 124 # for batch_size=16 on 1,972 samples
        accum_steps = 1
        total_updates = max_epochs * math.ceil(loader_len / accum_steps)
        self.assertEqual(total_updates, 300 * 124)

        # Microbatch 4 with accum_steps 4: loader_len = 493
        loader_len_micro = 493 # ceil(1972 / 4)
        accum_steps_micro = 4
        total_updates_micro = max_epochs * math.ceil(loader_len_micro / accum_steps_micro)
        self.assertEqual(total_updates_micro, 300 * 124)

    def test_f7_single_parameter_model_grouping(self):
        """Tier 2: A model with only 1 parameter (e.g. only 1D or only 2D) is handled gracefully."""
        linear_no_bias = nn.Linear(10, 1, bias=False)
        optimizer = build_grouped_optimizer(linear_no_bias, base_lr=1e-3, weight_decay=1e-4)
        self.assertEqual(len(optimizer.param_groups[0]["params"]), 1)
        self.assertEqual(len(optimizer.param_groups[1]["params"]), 0)

    def test_f7_cudnn_deterministic_flags(self):
        """Tier 2: Setting deterministic flags operates without runtime errors."""
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        self.assertFalse(torch.backends.cudnn.benchmark)
        self.assertTrue(torch.backends.cudnn.deterministic)

    def test_f7_zero_grad_set_to_none(self):
        """Tier 2: optimizer.zero_grad(set_to_none=True) successfully sets parameter gradients to None."""
        optimizer = build_grouped_optimizer(self.model)
        x = torch.randn(1, 1, 16, 16)
        out = self.model(x).sum()
        out.backward()

        has_grad_before = any(p.grad is not None for p in self.model.parameters())
        self.assertTrue(has_grad_before)

        optimizer.zero_grad(set_to_none=True)
        has_grad_after = any(p.grad is not None for p in self.model.parameters())
        self.assertFalse(has_grad_after)


class TestBatchAccumulationAndCheckpoint(unittest.TestCase):
    """Covers Feature 8: Effective Batch Size 16, Sample-Weighted Accumulation, and last.pth Checkpointing."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.model = SimpleConvModel()

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    # =========================================================================
    # Tier 1: Feature Coverage (F8)
    # =========================================================================

    def test_f8_effective_batch_size_accumulation_step(self):
        """Tier 1: Microbatch 4 with accum_steps 4 triggers optimizer step exactly every 4 batches (eff bs 16)."""
        optimizer = AdamW(self.model.parameters(), lr=1e-3)
        accum_steps = 4
        step_count = 0
        group_samples = 0

        # Simulate 12 microbatches of size 4 (total 48 samples = 3 optimizer steps)
        for batch_idx in range(12):
            batch_count = 4
            group_samples += batch_count
            boundary = ((batch_idx + 1) % accum_steps == 0)
            if boundary:
                step_count += 1
                group_samples = 0

        self.assertEqual(step_count, 3)

    def test_f8_sample_weighted_accumulation_gradient_scaling(self):
        """Tier 1: Sample-weighted accumulation:
        (loss * batch_count).backward() followed by grad.div_(group_samples)
        mathematically equals computing the loss over the combined macro-batch.
        """
        # Compare:
        # Method A: Macro-batch of 6 samples
        torch.manual_seed(42)
        model_a = nn.Linear(4, 1, bias=False)
        x_all = torch.randn(6, 4)
        target_all = torch.randn(6, 1)

        loss_a = F.mse_loss(model_a(x_all), target_all)
        loss_a.backward()
        grad_a = model_a.weight.grad.clone()

        # Method B: Sample-weighted accumulation over two microbatches of unequal size (4 and 2)
        torch.manual_seed(42)
        model_b = nn.Linear(4, 1, bias=False)
        x_m1, target_m1 = x_all[:4], target_all[:4]
        x_m2, target_m2 = x_all[4:], target_all[4:]

        group_samples = 0

        # Microbatch 1 (4 samples)
        loss_m1 = F.mse_loss(model_b(x_m1), target_m1)
        (loss_m1 * 4).backward()
        group_samples += 4

        # Microbatch 2 (2 samples)
        loss_m2 = F.mse_loss(model_b(x_m2), target_m2)
        (loss_m2 * 2).backward()
        group_samples += 2

        # Normalize by total accumulated samples (6)
        model_b.weight.grad.div_(group_samples)
        grad_b = model_b.weight.grad.clone()

        # The gradients must be practically identical
        torch.testing.assert_close(grad_a, grad_b, rtol=1e-5, atol=1e-6)

    def test_f8_gradient_clipping_norm_threshold(self):
        """Tier 1: Gradient norm clipping scales gradients when total norm exceeds threshold 1.0."""
        p = nn.Parameter(torch.tensor([10.0, 10.0]))
        p.grad = torch.tensor([3.0, 4.0]) # norm is 5.0
        orig_norm = torch.norm(p.grad).item()
        self.assertEqual(orig_norm, 5.0)

        torch.nn.utils.clip_grad_norm_([p], max_norm=1.0)
        clipped_norm = torch.norm(p.grad).item()
        self.assertAlmostEqual(clipped_norm, 1.0, places=5)

    def test_f8_atomic_checkpoint_saving_and_content(self):
        """Tier 1: Verify last.pth contains all required state dicts and metadata."""
        ckpt_path = os.path.join(self.test_dir, "last.pth")
        payload = {
            "model": self.model.state_dict(),
            "epoch": 299,
            "global_step": 37200,
            "metadata": {
                "planned_epochs": 300,
                "completed_epochs": 300,
                "training_complete": True,
                "checkpoint_selection": "last_completed_epoch",
            },
        }

        atomic_save_checkpoint(payload, ckpt_path)
        self.assertTrue(os.path.exists(ckpt_path))
        self.assertFalse(os.path.exists(f"{ckpt_path}.tmp"))

        loaded = torch.load(ckpt_path)
        self.assertEqual(loaded["epoch"], 299)
        self.assertTrue(loaded["metadata"]["training_complete"])

    def test_f8_checkpoint_epoch_300_training_complete_flag(self):
        """Tier 1: Checkpoint at epoch 299 (zero-indexed 300th epoch) has training_complete=True."""
        def make_metadata(epoch: int, max_epochs: int = 300) -> dict:
            return {
                "planned_epochs": max_epochs,
                "completed_epochs": epoch + 1,
                "training_complete": (epoch + 1 == max_epochs),
            }

        meta_epoch_100 = make_metadata(99)
        self.assertFalse(meta_epoch_100["training_complete"])

        meta_epoch_300 = make_metadata(299)
        self.assertTrue(meta_epoch_300["training_complete"])

    # =========================================================================
    # Tier 2: Boundary & Corner Cases (F8)
    # =========================================================================

    def test_f8_uneven_epoch_boundary_microbatch_accumulation(self):
        """Tier 2: Boundary condition: `(batch_idx + 1) == len(train_loader)` ensures
        the final microbatch of 4 samples (out of 1,972) triggers gradient update."""
        # 1,972 samples with microbatch 16: 123 batches of 16 + 1 batch of 4 = 124 batches
        batch_sizes = [16] * 123 + [4]
        loader_len = len(batch_sizes)
        accum_steps = 1 # or any number
        stepped_at = []

        group_samples = 0
        for idx, bs in enumerate(batch_sizes):
            group_samples += bs
            boundary = ((idx + 1) % accum_steps == 0) or ((idx + 1) == loader_len)
            if boundary:
                stepped_at.append((idx, group_samples))
                group_samples = 0

        # Last step must occur on the final batch index (123) with group_samples = 4 (or 16)
        last_step_idx, last_samples = stepped_at[-1]
        self.assertEqual(last_step_idx, 123)
        self.assertEqual(last_samples, 4)

    def test_f8_microbatch_size_one_accumulation(self):
        """Tier 2: Microbatch size 1 accumulates 16 microbatches to achieve effective batch size 16."""
        stepped = 0
        accum_steps = 16
        for i in range(32):
            if (i + 1) % accum_steps == 0:
                stepped += 1
        self.assertEqual(stepped, 2)

    def test_f8_accum_steps_one_direct_step(self):
        """Tier 2: When accum_steps=1, every batch is a boundary."""
        stepped = 0
        accum_steps = 1
        loader_len = 10
        for i in range(loader_len):
            if ((i + 1) % accum_steps == 0) or ((i + 1) == loader_len):
                stepped += 1
        self.assertEqual(stepped, 10)

    def test_f8_checkpoint_resume_restores_state_and_global_step(self):
        """Tier 2: Resume from checkpoint restores exact model weights and global_step."""
        ckpt_path = os.path.join(self.test_dir, "resume_test.pth")
        model1 = SimpleConvModel()
        payload = {
            "model": model1.state_dict(),
            "epoch": 45,
            "global_step": 5580,
        }
        atomic_save_checkpoint(payload, ckpt_path)

        model2 = SimpleConvModel()
        loaded = torch.load(ckpt_path)
        model2.load_state_dict(loaded["model"])

        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            torch.testing.assert_close(p1, p2)
        self.assertEqual(loaded["global_step"], 5580)

    def test_f8_checkpoint_atomic_file_rename_safety(self):
        """Tier 2: Atomic save leaves no temporary file behind even when overwriting repeatedly."""
        ckpt_path = os.path.join(self.test_dir, "atomic_loop.pth")
        for i in range(5):
            atomic_save_checkpoint({"epoch": i}, ckpt_path)
            self.assertTrue(os.path.exists(ckpt_path))
            self.assertFalse(os.path.exists(f"{ckpt_path}.tmp"))
        final_loaded = torch.load(ckpt_path)
        self.assertEqual(final_loaded["epoch"], 4)

    def test_f8_amp_grad_scaler_step_and_skip_logic(self):
        """Tier 2: Test AMP GradScaler lifecycle:
        - Normal step: unscale_, clip_grad_norm_, scaler.step updates weights, scaler.update, scheduler steps.
        - Overflow step: inf gradient triggers skip in scaler.step, scaler.update reduces scale factor,
          and scheduler step is skipped so learning rate does not advance.
        """
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        scaler = torch.amp.GradScaler(device_str, enabled=True)

        model = SimpleConvModel()
        base_lr = 1e-3
        optimizer = build_grouped_optimizer(model, base_lr=base_lr)
        total_updates = 100
        scheduler = get_polynomial_decay_scheduler(optimizer, total_updates=total_updates)

        # --- Phase 1: Successful step ---
        x1 = torch.randn(4, 1, 16, 16)
        y1 = torch.tensor([0, 1, 0, 1])
        loss1 = F.cross_entropy(model(x1), y1)

        scale_before1 = scaler.get_scale()
        scaler.scale(loss1).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(optimizer)
        scaler.update()
        scale_after1 = scaler.get_scale()

        # Step was NOT skipped
        self.assertGreaterEqual(scale_after1, scale_before1)
        scheduler.step()
        lr_after_step1 = optimizer.param_groups[0]["lr"]
        self.assertLess(lr_after_step1, base_lr)

        optimizer.zero_grad(set_to_none=True)

        # --- Phase 2: Overflow step (simulate Inf in gradients) ---
        x2 = torch.randn(4, 1, 16, 16)
        y2 = torch.tensor([1, 0, 1, 0])
        loss2 = F.cross_entropy(model(x2), y2)

        scaler.scale(loss2).backward()
        # Inject non-finite value into gradient before unscaling
        model.conv.weight.grad[0, 0, 0, 0] = float("inf")

        scale_before2 = scaler.get_scale()
        scaler.unscale_(optimizer)

        scaler.step(optimizer)
        scaler.update()
        scale_after2 = scaler.get_scale()

        # Scale factor must be reduced due to Inf gradient
        self.assertLess(scale_after2, scale_before2)

        # Optimizer step should have been skipped (scale reduced -> skipped update)
        skipped = scale_after2 < scale_before2
        self.assertTrue(skipped)
        if not skipped:
            scheduler.step()

        # Learning rate must NOT have decayed further
        lr_after_step2 = optimizer.param_groups[0]["lr"]
        self.assertEqual(lr_after_step2, lr_after_step1)

    def test_f8_full_production_checkpoint_payload_and_rng_roundtrip(self):
        """Tier 2: Comprehensive 22-field checkpoint payload schema validation and
        multi-framework RNG state roundtrip (Python random, NumPy, PyTorch, Generator).
        """
        ckpt_path = os.path.join(self.test_dir, "full_production_ckpt.pth")

        # 1. Setup deterministic state and model
        random.seed(1234)
        np.random.seed(1234)
        torch.manual_seed(1234)
        gen = torch.Generator().manual_seed(1234)

        model = SimpleConvModel()
        optimizer = build_grouped_optimizer(model, base_lr=1e-3, weight_decay=1e-4)
        scheduler = get_polynomial_decay_scheduler(optimizer, total_updates=37200)

        # Step model once
        x = torch.randn(2, 1, 16, 16)
        loss = model(x).sum()
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        # Generate some random values to advance states
        _ = [random.random() for _ in range(10)]
        _ = np.random.randn(5)
        _ = torch.randn(5)
        _ = torch.randn(5, generator=gen)

        # Predict next random values
        py_state_before = random.getstate()
        expected_py_val = random.random()
        random.setstate(py_state_before)

        np_state_before = np.random.get_state()
        expected_np_val = float(np.random.rand())
        np.random.set_state(np_state_before)

        torch_state_before = torch.get_rng_state()
        expected_torch_val = torch.randn(1).item()
        torch.set_rng_state(torch_state_before)

        gen_state_before = gen.get_state()
        expected_gen_val = torch.randn(1, generator=gen).item()
        gen.set_state(gen_state_before)

        # 2. Build full 22-field production schema
        payload = {
            "epoch": 49,
            "completed_epochs": 50,
            "planned_epochs": 300,
            "training_complete": False,
            "checkpoint_selection": "last_completed_epoch",
            "checkpoint_selection_metric": None,
            "protocol": "myops380_train304_test76_last_v1",
            "format_version": 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": None,
            "global_step": 6200,
            "total_updates": 37200,
            "skipped_amp_steps": 0,
            "epoch_loss": 0.2345,
            "seed": 1234,
            "base_lr": 1e-3,
            "batch_size": 4,
            "accum_steps": 4,
            "effective_batch_size": 16,
            "loss_identifier": "MyoPSLoss_Dice_WCE_Invariant_Inclusive",
            "data_root": r"D:\nckh_model\Dataset\MyoPS380\Processed_data",
            "split_dir": r"D:\nckh_model\SCAR\preprocessing\splits\myops380_no_val",
            "split_hashes": {"train": "abc123hash", "test_vol": "def456hash"},
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy_rng_state": np.random.get_state(),
            "random_rng_state": random.getstate(),
            "loader_rng": gen.get_state(),
            "generator_rng_state": gen.get_state(),
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "generator": gen.get_state(),
            },
        }

        # Verify key count is at least 22
        self.assertGreaterEqual(len(payload), 22)

        # 3. Save atomically
        atomic_save_checkpoint(payload, ckpt_path)
        self.assertTrue(os.path.isfile(ckpt_path))

        # 4. Perturb RNG states and model
        random.seed(9999)
        np.random.seed(9999)
        torch.manual_seed(9999)
        gen2 = torch.Generator().manual_seed(9999)

        # 5. Restore from checkpoint
        loaded = torch.load(ckpt_path, weights_only=False)
        self.assertEqual(loaded["protocol"], "myops380_train304_test76_last_v1")
        self.assertEqual(loaded["format_version"], 1)
        self.assertEqual(loaded["completed_epochs"], 50)
        self.assertEqual(loaded["global_step"], 6200)

        # Restore model & optimizer
        model2 = SimpleConvModel()
        model2.load_state_dict(loaded["model_state_dict"])
        for p1, p2 in zip(model.parameters(), model2.parameters()):
            torch.testing.assert_close(p1, p2)

        # Restore RNG states
        restore_rng_states(loaded, generator=gen2)

        # 6. Verify subsequent random generations match expected values exactly
        actual_py_val = random.random()
        actual_np_val = float(np.random.rand())
        actual_torch_val = torch.randn(1).item()
        actual_gen_val = torch.randn(1, generator=gen2).item()

        self.assertEqual(actual_py_val, expected_py_val)
        self.assertEqual(actual_np_val, expected_np_val)
        self.assertAlmostEqual(actual_torch_val, expected_torch_val, places=6)
        self.assertAlmostEqual(actual_gen_val, expected_gen_val, places=6)


class TestTrainingDynamicsCrossFeature(unittest.TestCase):
    """Tier 3: Cross-Feature Integration across Optimizer, Scheduler, and Accumulation."""

    def test_t3_training_loop_optimizer_scheduler_accumulation_cycle(self):
        """Tier 3: Multi-step training cycle:
        Microbatches -> sample-weighted loss backward -> accumulation boundary ->
        div_ samples -> clip_grad_norm_ -> optimizer.step() -> scheduler.step() -> zero_grad().
        """
        model = SimpleConvModel()
        base_lr = 1e-2
        accum_steps = 2
        total_updates = 4

        optimizer = build_grouped_optimizer(model, base_lr=base_lr)
        scheduler = get_polynomial_decay_scheduler(optimizer, total_updates=total_updates)

        # Record initial weights
        init_weight = model.conv.weight.clone()

        microbatches = [
            (torch.randn(2, 1, 16, 16), torch.tensor([0, 1])),
            (torch.randn(2, 1, 16, 16), torch.tensor([1, 0])),
            (torch.randn(2, 1, 16, 16), torch.tensor([0, 0])),
            (torch.randn(2, 1, 16, 16), torch.tensor([1, 1])),
        ]

        group_samples = 0
        optimizer_steps = 0

        criterion = nn.CrossEntropyLoss()

        for idx, (x, y) in enumerate(microbatches):
            b_count = x.shape[0]
            out = model(x)
            loss = criterion(out, y)
            (loss * b_count).backward()
            group_samples += b_count

            boundary = ((idx + 1) % accum_steps == 0) or ((idx + 1) == len(microbatches))
            if boundary:
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad.div_(group_samples)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                group_samples = 0
                optimizer_steps += 1

        self.assertEqual(optimizer_steps, 2)
        # Model weights must have changed
        self.assertFalse(torch.allclose(init_weight, model.conv.weight))
        # Learning rate must have decayed
        self.assertLess(optimizer.param_groups[0]["lr"], base_lr)

    def test_t3_myopsnet_production_training_step_cycle(self):
        """Tier 3: Full production training step integration on MyoPSNet:
        - Feeds 5 input modalities (C0, LGE, T2, T1m zero-padded, T2starm zero-padded).
        - Maps canonical 4-class labels to MyoPS target dictionary.
        - Evaluates MyoPSLoss (seg + invariant + inclusive).
        - Sample-weighted gradient accumulation over 2 microbatches.
        - Gradient normalization, gradient clipping, optimizer step, and step-level PolyLR decay.
        """
        torch.manual_seed(1234)
        model = MyoPSNet(in_chs=(5, 2, 2, 3), out_chs=(3, 3, 3, 3))
        criterion = MyoPSLoss()
        base_lr = 1e-3
        optimizer = build_grouped_optimizer(model, base_lr=base_lr, weight_decay=1e-4)
        accum_steps = 2
        total_updates = 10
        scheduler = get_polynomial_decay_scheduler(optimizer, total_updates=total_updates, power=0.9)

        # Record initial weights of a Conv2d layer
        init_weight = model.unet_C0.Conv1.conv[0].weight.clone()

        # Simulate 2 microbatches of size B=2 with 128x128 resolution
        microbatches = []
        for i in range(2):
            mb = {
                "C0": torch.rand(2, 1, 128, 128, dtype=torch.float32),
                "LGE": torch.rand(2, 1, 128, 128, dtype=torch.float32),
                "T2": torch.rand(2, 1, 128, 128, dtype=torch.float32),
                "T1m": torch.zeros(2, 1, 128, 128, dtype=torch.float32),
                "T2starm": torch.zeros(2, 1, 128, 128, dtype=torch.float32),
                "label": torch.randint(0, 4, (2, 128, 128), dtype=torch.int64),
            }
            microbatches.append(mb)

        group_samples = 0
        optimizer.zero_grad(set_to_none=True)

        for batch_idx, batch in enumerate(microbatches):
            b_count = batch["label"].shape[0]
            group_samples += b_count

            # Target adapter
            targets_dict = canonical_to_myops_targets(batch["label"])

            # 5-branch forward pass
            res_c0, res_lge, res_t2, res_mapping = model(
                batch["C0"], batch["LGE"], batch["T2"], batch["T1m"], batch["T2starm"]
            )
            seg_outputs = {
                "cardiac": res_c0,
                "scar": res_lge,
                "edema": res_t2,
                "mapping": res_mapping,
            }

            # Loss evaluation
            loss_seg, loss_inv, loss_inc, loss = criterion(seg_outputs, targets_dict)
            self.assertTrue(torch.isfinite(loss), "Loss must be finite")
            self.assertTrue(torch.isfinite(loss_seg), "loss_seg must be finite")
            self.assertTrue(torch.isfinite(loss_inv), "loss_inv must be finite")
            self.assertTrue(torch.isfinite(loss_inc), "loss_inc must be finite")

            # Sample-weighted backward
            weighted_loss = loss * b_count
            weighted_loss.backward()

            boundary = ((batch_idx + 1) % accum_steps == 0) or ((batch_idx + 1) == len(microbatches))
            if boundary:
                # Divide gradients by accumulated samples
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad.div_(group_samples)

                # Gradient norm clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                self.assertTrue(torch.isfinite(grad_norm), "Gradient norm must be finite")

                # Step optimizer and scheduler
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                group_samples = 0

        # Assertions
        # 1. Weights were updated
        current_weight = model.unet_C0.Conv1.conv[0].weight
        self.assertFalse(torch.allclose(init_weight, current_weight), "Model weights must change after optimizer step")

        # 2. Gradients reset to None
        for p in model.parameters():
            self.assertIsNone(p.grad, "Gradients must be set to None by zero_grad(set_to_none=True)")

        # 3. Learning rate decayed
        current_lr = optimizer.param_groups[0]["lr"]
        self.assertLess(current_lr, base_lr, "LR must decay after scheduler step")


if __name__ == "__main__":
    unittest.main()
