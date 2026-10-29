"""Unified Entry Point for MyoPS-Net on MyoPS380 Benchmark.

Usage:
  # Run training (300 epochs, no_val protocol, effective batch 16, AdamW, PolyLR)
  python main.py train --data_root "D:\\nckh_model\\Dataset\\MyoPS380\\Processed_data"
  python main.py train --data_dir "D:\\nckh_model\\Dataset\\MyoPS380\\Processed_data"

  # Run evaluation on test cohort (76 cases)
  python main.py evaluate --checkpoint "checkpoints/last.pth"
"""
import sys
from train import (
    build_arg_parser as build_train_parser,
    train,
    MyoPSNetTrain,
    CrossModalSegNetTrain,
)
from evaluate import build_arg_parser as build_eval_parser, evaluate

__all__ = [
    "train",
    "evaluate",
    "MyoPSNetTrain",
    "CrossModalSegNetTrain",
]

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("eval", "evaluate"):
        parser = build_eval_parser()
        args = parser.parse_args(sys.argv[2:])
        evaluate(args)
    elif len(sys.argv) > 1 and sys.argv[1] in ("train", "fit"):
        parser = build_train_parser()
        args = parser.parse_args(sys.argv[2:])
        train(args)
    else:
        parser = build_train_parser()
        args = parser.parse_args()
        train(args)
