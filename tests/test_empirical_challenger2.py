"""
Adversarial Empirical Test Suite for Milestone 2 Challenger 2:
Pipeline Integration, Shape Invariance, and Round-Trip Invariants.
"""
import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np

from dataset.myops380_dataset import build_myops380_dataloader
from dataset.target_adapter import canonical_to_myops_targets, myops_outputs_to_canonical, TargetAdapter
from network.model import MyoPSNet
from network.criterion.loss import MyoPSLoss

def run_empirical_tests():
    print("=" * 70)
    print("CHALLENGE 2.A: Real Data Integration & Backward Across 10+ Batches")
    print("=" * 70)
    
    loader = build_myops380_dataloader(batch_size=4, num_workers=0)
    model = MyoPSNet(in_chs=(5, 2, 2, 3), out_chs=(3, 3, 3, 3))
    criterion = MyoPSLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    batch_count = 0
    max_batches = 12  # Test 12 batches (> 10)
    
    for i, batch in enumerate(loader):
        if batch_count >= max_batches:
            break
            
        C0 = batch["C0"]
        LGE = batch["LGE"]
        T2 = batch["T2"]
        T1m = batch["T1m"]
        T2starm = batch["T2starm"]
        label = batch["label"]
        case_id = batch["case_id"]
        
        # Verify shapes
        assert C0.shape == (4, 1, 128, 128), f"Batch {i}: C0 shape unexpected {C0.shape}"
        assert LGE.shape == (4, 1, 128, 128), f"Batch {i}: LGE shape unexpected {LGE.shape}"
        assert T2.shape == (4, 1, 128, 128), f"Batch {i}: T2 shape unexpected {T2.shape}"
        assert T1m.shape == (4, 1, 128, 128), f"Batch {i}: T1m shape unexpected {T1m.shape}"
        assert T2starm.shape == (4, 1, 128, 128), f"Batch {i}: T2starm shape unexpected {T2starm.shape}"
        assert label.shape == (4, 128, 128), f"Batch {i}: label shape unexpected {label.shape}"
        
        # Adapt labels
        targets = canonical_to_myops_targets(label)
        assert "cardiac" in targets and "scar" in targets and "edema" in targets
        assert targets["cardiac"].shape == (4, 3, 128, 128)
        assert targets["scar"].shape == (4, 3, 128, 128)
        assert targets["edema"].shape == (4, 3, 128, 128)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(C0, LGE, T2, T1m, T2starm)
        # outputs is a tuple: (seg_C0, seg_LGE, seg_T2, seg_mapping)
        assert len(outputs) == 4
        for branch_idx, branch_out in enumerate(outputs):
            assert branch_out.shape == (4, 3, 128, 128), f"Branch {branch_idx} shape: {branch_out.shape}"
            assert torch.isfinite(branch_out).all(), f"Batch {i} branch {branch_idx} contains non-finite values"
            
        # Compute loss
        loss_seg, loss_inv, loss_inc, total_loss = criterion(outputs, targets)
        
        assert torch.isfinite(loss_seg), f"Batch {i}: loss_seg not finite: {loss_seg.item()}"
        assert torch.isfinite(loss_inv), f"Batch {i}: loss_inv not finite: {loss_inv.item()}"
        assert torch.isfinite(loss_inc), f"Batch {i}: loss_inc not finite: {loss_inc.item()}"
        assert torch.isfinite(total_loss), f"Batch {i}: total_loss not finite: {total_loss.item()}"
        
        # Backward pass
        total_loss.backward()
        
        # Check gradients
        grad_norms = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Batch {i}: param {name} has None grad"
                assert torch.isfinite(param.grad).all(), f"Batch {i}: param {name} has non-finite grad"
                grad_norms.append(param.grad.norm().item())
                
        optimizer.step()
        batch_count += 1
        print(f"Batch {batch_count:02d} [{case_id[0]}]: total_loss={total_loss.item():.4f} "
              f"(seg={loss_seg.item():.4f}, inv={loss_inv.item():.4f}, inc={loss_inc.item():.4f}), "
              f"max_grad={max(grad_norms):.4e}")
              
    print(f"--> Challenge 2.A SUCCESS: {batch_count} batches passed zero-error forward/backward.")
    
    print("\n" + "=" * 70)
    print("CHALLENGE 2.B: Shape Invariance (2D, Batched 2D, 3D volumes)")
    print("=" * 70)
    
    shapes_to_test = [
        ("2D slice (H, W)", (128, 128)),
        ("Batched 2D (B, H, W)", (3, 128, 128)),
        ("4D slice (B, 1, H, W)", (2, 1, 128, 128)),
        ("3D unbatched volume (H, W, D)", (128, 128, 8)),
        ("3D batched volume (B, H, W, D)", (2, 128, 128, 8)),
    ]
    
    for desc, shape in shapes_to_test:
        print(f"Testing input shape {shape} ({desc})...")
        label = torch.randint(0, 4, shape, dtype=torch.int64)
        try:
            targets = canonical_to_myops_targets(label)
            print(f"  Target shapes: scar={targets['scar'].shape}, edema={targets['edema'].shape}, cardiac={targets['cardiac'].shape}")
            
            # Now test reverse mapping
            pred_canonical = myops_outputs_to_canonical(targets)
            print(f"  Reverse mapping shape: {pred_canonical.shape}, dtype: {pred_canonical.dtype}")
            print(f"  Match input shape? {pred_canonical.shape == label.shape} (Input: {label.shape}, Output: {pred_canonical.shape})")
            if pred_canonical.shape == label.shape:
                match = torch.equal(pred_canonical, label)
                print(f"  Exact content match: {match}")
            else:
                print(f"  SHAPE MISMATCH: Input {label.shape} vs Output {pred_canonical.shape}")
        except Exception as e:
            print(f"  EXCEPTION RAISED: {type(e).__name__}: {e}")

if __name__ == "__main__":
    run_empirical_tests()
