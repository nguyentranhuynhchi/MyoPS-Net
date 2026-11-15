"""
training/checkpoint.py
======================
Atomic Checkpoint Serialization and Lossless Resumption for MyoPS-Net.
Protocol: myops380_train304_test76_last_v1
Benchmark: MyoPS380 (304 Train / 76 Test / 300 Epochs / Strict NO Validation)
"""

from __future__ import annotations
import math
import os
from pathlib import Path
import random
import time
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn

from training.optimizer import OPTIMIZATION_PROTOCOL
from training.provenance import validate_resume_contract


def capture_rng_state() -> Dict[str, Any]:
    """Capture complete RNG state across Python, NumPy, and PyTorch (CPU and CUDA)."""
    np_state = np.random.get_state()
    # Store numpy state array as uint32 list for broad compatibility
    np_state_list = [
        np_state[0],
        np_state[1].tolist() if hasattr(np_state[1], "tolist") else list(np_state[1]),
        np_state[2],
        np_state[3],
        np_state[4],
    ]
    state = {
        "python_random": random.getstate(),
        "numpy_random": np_state_list,
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        # Dual-key aliases for backward / cross-tool interoperability
        "python": random.getstate(),
        "numpy": np_state_list,
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }
    return state


def restore_rng_state(rng_dict: Dict[str, Any], *, strict: bool = False) -> None:
    """Restore complete RNG state across Python, NumPy, and PyTorch (CPU and CUDA)."""
    if not isinstance(rng_dict, dict):
        if strict:
            raise ValueError("Missing RNG state for exact resumption")
        return

    if strict:
        for key in ("python_random", "numpy_random", "torch_cpu", "torch_cuda"):
            if key not in rng_dict or rng_dict[key] is None:
                raise ValueError(f"Missing RNG state: {key}")

    # 1. Python random
    py_state = rng_dict.get("python_random", rng_dict.get("python"))
    if py_state is not None:
        try:
            random.setstate(py_state)
        except Exception as exc:
            if strict:
                raise ValueError("Cannot restore Python RNG state") from exc

    # 2. NumPy random
    np_state = rng_dict.get("numpy_random", rng_dict.get("numpy"))
    if np_state is not None:
        try:
            s0 = np_state[0]
            s1 = np.asarray(np_state[1], dtype=np.uint32)
            s2 = int(np_state[2])
            s3 = int(np_state[3])
            s4 = float(np_state[4])
            np.random.set_state((s0, s1, s2, s3, s4))
        except Exception as exc:
            if strict:
                raise ValueError("Cannot restore NumPy RNG state") from exc

    # 3. PyTorch CPU
    cpu_state = rng_dict.get("torch_cpu", rng_dict.get("torch"))
    if cpu_state is not None:
        try:
            if isinstance(cpu_state, torch.Tensor):
                torch.set_rng_state(cpu_state.cpu())
            elif isinstance(cpu_state, (bytes, bytearray)):
                torch.set_rng_state(torch.frombuffer(cpu_state, dtype=torch.uint8))
            elif strict:
                raise ValueError("Unsupported PyTorch RNG state")
        except Exception as exc:
            if strict:
                raise ValueError("Cannot restore PyTorch RNG state") from exc

    # 4. PyTorch CUDA
    cuda_state = rng_dict.get("torch_cuda", rng_dict.get("cuda"))
    if torch.cuda.is_available() and cuda_state is not None and len(cuda_state) > 0:
        try:
            torch.cuda.set_rng_state_all([state.cpu() for state in cuda_state])
        except Exception as exc:
            if strict:
                raise ValueError("Cannot restore CUDA RNG state") from exc


def safe_atomic_save(payload: Dict[str, Any], filepath: Union[str, Path]) -> None:
    """Atomically write checkpoint to disk via PID-tagged temporary file with Windows retry backoff."""
    target_path = Path(filepath).resolve()
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_name(f"{target_path.name}.tmp_{os.getpid()}")

    try:
        with open(tmp_path, "wb") as f:
            torch.save(payload, f)
            f.flush()
            os.fsync(f.fileno())

        max_retries = 5
        for attempt in range(max_retries):
            try:
                os.replace(tmp_path, target_path)
                break
            except (PermissionError, OSError) as exc:
                if attempt == max_retries - 1:
                    raise exc
                time.sleep(0.05 * (2 ** attempt))
    except Exception:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        raise


def atomic_save_checkpoint(path: Union[str, Path], payload: Dict[str, Any]) -> None:
    """Convenience alias for safe_atomic_save."""
    safe_atomic_save(payload, path)


def validate_checkpoint_payload(payload: Dict[str, Any], require_complete: bool = False) -> None:
    """Enforce benchmark schema validity and protocol consistency on checkpoint dictionary."""
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint payload must be a dict, got {type(payload)}")

    required_keys = [
        "epoch",
        "completed_epochs",
        "planned_epochs",
        "training_complete",
        "global_step",
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "scaler_state_dict",
        "seed",
        "protocol_id",
        "opt_protocol",
        "rng_state",
    ]
    missing = [k for k in required_keys if k not in payload]
    if missing:
        raise ValueError(f"Checkpoint payload missing mandatory keys: {missing}")

    # Check protocol identifiers
    valid_protocols = {
        "myops380_train304_test76_last_v1",
        "myopspp_train64_test16_last_v1",
        "myopspp_unet2d_roi128_1mm_train64_test16_last_v1",
        "myopspp_bc80_train64_test16_last_v1",
    }
    if payload["protocol_id"] not in valid_protocols:
        raise ValueError(f"Invalid protocol_id: {payload['protocol_id']}")
    opt_proto = payload["opt_protocol"]
    opt_id = opt_proto.get("id") if isinstance(opt_proto, dict) else str(opt_proto)
    if opt_id != "im_mseg_adamw_poly_v1":
        raise ValueError(f"Invalid opt_protocol: {payload['opt_protocol']}")

    # Check budget and counter consistency
    epoch = payload["epoch"]
    completed = payload["completed_epochs"]
    planned = payload["planned_epochs"]
    complete = payload["training_complete"]

    if not isinstance(epoch, int) or epoch < 1:
        raise ValueError(f"Invalid epoch: {epoch}")
    if completed != epoch:
        raise ValueError(f"Mismatched epoch counters: epoch={epoch}, completed_epochs={completed}")
    if not isinstance(planned, int) or planned < 1 or completed > planned:
        raise ValueError(f"Inconsistent planned epochs budget: completed={completed}, planned={planned}")
    if complete != (completed >= planned):
        raise ValueError(
            f"Inconsistent training_complete flag: completed={completed}/{planned}, "
            f"training_complete={complete}"
        )

    if require_complete and not complete:
        raise ValueError(f"Incomplete training checkpoint: completed {completed}/{planned} epochs.")

    # Check RNG dictionary
    rng = payload["rng_state"]
    if not isinstance(rng, dict):
        raise ValueError("rng_state must be a dictionary.")
    for rng_key in ("python_random", "numpy_random", "torch_cpu", "torch_cuda"):
        if rng_key not in rng:
            raise ValueError(f"rng_state missing key: {rng_key}")


def save_checkpoint(
    state_or_path: Optional[Union[Dict[str, Any], str, Path]] = None,
    checkpoint_path: Optional[Union[str, Path]] = None,
    *,
    model: Optional[nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
    epoch: Optional[int] = None,
    completed_epochs: Optional[int] = None,
    planned_epochs: int = 300,
    global_step: Optional[int] = None,
    total_updates: int = 37200,
    seed: int = 1234,
    args: Optional[Any] = None,
    amp_skipped_updates: int = 0,
    loader_generator: Optional[torch.Generator] = None,
    **extra: Any,
) -> None:
    """Save standardized checkpoint with atomic replacement.
    
    Supports flexible calling conventions:
    1. Direct dictionary payload:
       `save_checkpoint(state_dict, checkpoint_path)`
    2. Explicit component arguments with keyword checkpoint_path:
       `save_checkpoint(checkpoint_path=..., model=..., optimizer=..., scheduler=..., ...)`
    3. Explicit component arguments with positional path:
       `save_checkpoint(checkpoint_path, model=..., optimizer=..., scheduler=..., ...)`
    """
    # Resolve target_path and payload mode
    if state_or_path is None:
        if checkpoint_path is None:
            raise ValueError("checkpoint_path or state_or_path must be provided.")
        target_path = checkpoint_path
    elif isinstance(state_or_path, (str, Path)) and checkpoint_path is None:
        target_path = state_or_path
    elif isinstance(state_or_path, dict):
        if checkpoint_path is None:
            raise ValueError("checkpoint_path must be specified when saving a dictionary payload.")
        target_path = checkpoint_path
        payload = dict(state_or_path)

        # Ensure mandatory aliases and standard keys are present
        if "model_state_dict" in payload and "model" not in payload:
            payload["model"] = payload["model_state_dict"]
        elif "model" in payload and "model_state_dict" not in payload:
            payload["model_state_dict"] = payload["model"]

        if "optimizer_state_dict" in payload and "optimizer" not in payload:
            payload["optimizer"] = payload["optimizer_state_dict"]
        elif "optimizer" in payload and "optimizer_state_dict" not in payload:
            payload["optimizer_state_dict"] = payload["optimizer"]

        if "scheduler_state_dict" in payload and "scheduler" not in payload:
            payload["scheduler"] = payload["scheduler_state_dict"]
        elif "scheduler" in payload and "scheduler_state_dict" not in payload:
            payload["scheduler_state_dict"] = payload["scheduler"]

        if "scaler_state_dict" in payload and "scaler" not in payload:
            payload["scaler"] = payload["scaler_state_dict"]
        elif "scaler" in payload and "scaler_state_dict" not in payload:
            payload["scaler_state_dict"] = payload["scaler"]

        if "rng_state" not in payload:
            payload["rng_state"] = capture_rng_state()
        if "rng" not in payload:
            payload["rng"] = payload["rng_state"]

        if "protocol_id" not in payload:
            ds = str(payload.get("dataset", "myops380")).lower()
            if ds in ("myopspp", "myopspp_unet2d_roi128_1mm"):
                payload["protocol_id"] = "myopspp_train64_test16_last_v1"
            else:
                payload["protocol_id"] = "myops380_train304_test76_last_v1"
        if "opt_protocol" not in payload:
            payload["opt_protocol"] = "im_mseg_adamw_poly_v1"
        if "optimization_protocol" not in payload:
            payload["optimization_protocol"] = OPTIMIZATION_PROTOCOL

        if "epoch" in payload and "completed_epochs" not in payload:
            payload["completed_epochs"] = payload["epoch"]
        elif "completed_epochs" in payload and "epoch" not in payload:
            payload["epoch"] = payload["completed_epochs"]

        comp = payload.get("completed_epochs", 1)
        plan = payload.get("planned_epochs", 300)
        if "training_complete" not in payload:
            payload["training_complete"] = (comp >= plan)

        validate_checkpoint_payload(payload)
        safe_atomic_save(payload, target_path)
        return
    else:
        target_path = checkpoint_path or state_or_path

    if model is None:
        raise ValueError("model must be provided when constructing checkpoint payload.")
    if optimizer is None:
        raise ValueError("optimizer must be provided when constructing checkpoint payload.")
    if scheduler is None:
        raise ValueError("scheduler must be provided when constructing checkpoint payload.")
    if scaler is None:
        raise ValueError("scaler must be provided when constructing checkpoint payload.")

    cur_epoch = epoch if epoch is not None else (completed_epochs if completed_epochs is not None else 1)
    cur_completed = completed_epochs if completed_epochs is not None else cur_epoch
    cur_step = global_step if global_step is not None else 0
    rng_dict = capture_rng_state()

    args_dict = vars(args) if hasattr(args, "__dict__") else (dict(args) if isinstance(args, dict) else None)

    model_state = model.state_dict()
    optimizer_state = optimizer.state_dict()
    scheduler_state = scheduler.state_dict()
    scaler_state = scaler.state_dict()

    protocol_id = extra.pop("protocol_id", None)
    if protocol_id is None:
        ds = extra.get("dataset", (args_dict.get("dataset", "myops380") if args_dict else "myops380"))
        if str(ds).lower() in ("myopspp", "myopspp_unet2d_roi128_1mm"):
            protocol_id = "myopspp_train64_test16_last_v1"
        else:
            protocol_id = "myops380_train304_test76_last_v1"

    payload = {
        "protocol_id": protocol_id,
        "opt_protocol": "im_mseg_adamw_poly_v1",
        "optimization_protocol": OPTIMIZATION_PROTOCOL,
        "format_version": 1,
        "model_name": "myopsnet",
        "seed": seed,
        "epoch": cur_epoch,
        "completed_epochs": cur_completed,
        "planned_epochs": planned_epochs,
        "global_step": cur_step,
        "total_updates": total_updates,
        "training_complete": (cur_completed >= planned_epochs),
        "amp_skipped_updates": amp_skipped_updates,
        # Primary state dicts
        "model_state_dict": model_state,
        "optimizer_state_dict": optimizer_state,
        "scheduler_state_dict": scheduler_state,
        "scaler_state_dict": scaler_state,
        # Aliases (shared object references, no duplicate calls to state_dict)
        "model": model_state,
        "optimizer": optimizer_state,
        "scheduler": scheduler_state,
        "scaler": scaler_state,
        # RNG states
        "rng_state": rng_dict,
        "rng": rng_dict,
        "args": args_dict,
    }

    if loader_generator is not None:
        payload["loader_rng"] = loader_generator.get_state()

    for k, v in extra.items():
        if k not in payload:
            payload[k] = v

    validate_checkpoint_payload(payload)
    safe_atomic_save(payload, target_path)


def load_checkpoint(
    checkpoint_path: Union[str, Path],
    model: Optional[nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
    map_location: Union[str, torch.device] = "cpu",
    expected_resume_contract: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Load checkpoint file and optionally restore model, optimizer, scheduler, scaler, RNG."""
    target_path = Path(checkpoint_path).resolve()
    if not target_path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {target_path}")

    payload = torch.load(target_path, map_location=map_location, weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid checkpoint format in {target_path}: expected dict, got {type(payload)}")

    validate_checkpoint_payload(payload)
    if expected_resume_contract is not None:
        # Validate before mutating any model, optimizer or RNG state.
        validate_resume_contract(payload, expected_resume_contract)

    # 1. Restore Model
    if model is not None:
        m_sd = payload.get("model_state_dict", payload.get("model"))
        if m_sd is None:
            raise KeyError(f"No model state dictionary found in {target_path}")
        model.load_state_dict(m_sd, strict=True)

    # 2. Restore Optimizer
    if optimizer is not None:
        o_sd = payload.get("optimizer_state_dict", payload.get("optimizer"))
        if o_sd is not None:
            optimizer.load_state_dict(o_sd)

    # 3. Restore Scheduler
    if scheduler is not None:
        s_sd = payload.get("scheduler_state_dict", payload.get("scheduler"))
        if s_sd is not None:
            scheduler.load_state_dict(s_sd)
            if optimizer is not None and hasattr(scheduler, "get_last_lr"):
                for param_group, lr in zip(optimizer.param_groups, scheduler.get_last_lr()):
                    param_group["lr"] = lr

    # 4. Restore Scaler
    if scaler is not None:
        sc_sd = payload.get("scaler_state_dict", payload.get("scaler"))
        if sc_sd is not None:
            scaler.load_state_dict(sc_sd)

    # 5. Restore RNG
    rng_dict = payload.get("rng_state", payload.get("rng"))
    if rng_dict is not None:
        restore_rng_state(rng_dict, strict=expected_resume_contract is not None)

    return payload


def resume_checkpoint(
    checkpoint_path: Union[str, Path],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    device: Union[str, torch.device] = "cpu",
) -> Tuple[int, int, int, bool]:
    """Restore training state from checkpoint file.
    
    Returns:
        start_epoch (int): 0-indexed loop start epoch (equal to completed_epochs).
        completed_epochs (int): Number of epochs already completed (1-indexed).
        global_step (int): Total successful updates completed.
        training_complete (bool): True if completed_epochs >= planned_epochs.
    """
    ckpt = load_checkpoint(
        checkpoint_path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        map_location=device,
    )
    completed = ckpt.get("completed_epochs", ckpt.get("epoch", 0))
    planned = ckpt.get("planned_epochs", 300)
    is_complete = ckpt.get("training_complete", completed >= planned)
    start_epoch = completed
    global_step = ckpt.get("global_step", 0)

    return start_epoch, completed, global_step, is_complete


def resume_training_state(
    checkpoint_path: Union[str, Path],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    loader_generator: Optional[torch.Generator] = None,
    device: Union[str, torch.device] = "cpu",
    expected_resume_contract: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Comprehensive resumption routine with clean exit detection and loader RNG restoration."""
    checkpoint_path = Path(checkpoint_path).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint file does not exist: {checkpoint_path}")

    checkpoint = load_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        map_location=device,
        expected_resume_contract=expected_resume_contract,
    )

    completed = checkpoint["completed_epochs"]
    planned = checkpoint["planned_epochs"]
    complete = checkpoint["training_complete"]

    # Restore DataLoader generator state if provided
    if loader_generator is not None and "loader_rng" in checkpoint:
        loader_generator.set_state(checkpoint["loader_rng"].cpu())

    if complete or completed >= planned:
        return {
            "status": "already_completed",
            "start_epoch": completed,
            "next_epoch": completed + 1,
            "completed_epochs": completed,
            "planned_epochs": planned,
            "global_step": checkpoint["global_step"],
            "amp_skipped_updates": checkpoint.get("amp_skipped_updates", 0),
            "training_complete": True,
            "checkpoint": checkpoint,
        }

    start_epoch = completed
    next_epoch = completed + 1
    global_step = checkpoint["global_step"]

    return {
        "status": "resumed",
        "start_epoch": start_epoch,
        "next_epoch": next_epoch,
        "completed_epochs": completed,
        "planned_epochs": planned,
        "global_step": global_step,
        "amp_skipped_updates": checkpoint.get("amp_skipped_updates", 0),
        "training_complete": False,
        "checkpoint": checkpoint,
    }
