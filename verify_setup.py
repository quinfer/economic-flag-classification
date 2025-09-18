#!/usr/bin/env python3
"""
Setup verification for hierarchical flag classification task
Checks dataset structure, available splits, and model checkpoint.
"""

from pathlib import Path

def verify_dataset():
    """Check dataset structure and splits (generic reporting)"""
    data_root = Path("datasets/NIFlagsV2")

    print("Dataset verification:")

    # Report any available split files and their line counts
    split_names = ["train", "val", "test"]
    total = 0
    found_any = False
    for split in split_names:
        split_file = data_root / f"{split}.txt"
        if split_file.exists():
            found_any = True
            with open(split_file) as f:
                actual = len(f.readlines())
            total += actual
            print(f"  {split}: {actual} samples")
        else:
            print(f"  {split}: not present")

    if found_any:
        print(f"  Total (across present splits): {total}")
    else:
        print("  No split files found under datasets/NIFlagsV2")

    # Check class names
    classnames_file = data_root / "classnames.txt"
    if classnames_file.exists():
        with open(classnames_file) as f:
            classes = [c for c in f.read().strip().split('\n') if c]
        print(f"  Classes file present: {len(classes)} categories declared")
    else:
        print("  Classes file missing: datasets/NIFlagsV2/classnames.txt")

def verify_model_checkpoint():
    """Check if RS5M checkpoint is available"""
    checkpoint_path = Path("checkpoints/RS5M_ViT-H-14.pt")
    if checkpoint_path.exists():
        size_gb = checkpoint_path.stat().st_size / (1024**3)
        print(f"Model checkpoint: Available ({size_gb:.1f}GB)")
    else:
        print("Model checkpoint: Not found - run download command first")

if __name__ == "__main__":
    print("Economic Consolidation Setup Check")
    print("-" * 40)
    verify_dataset()
    verify_model_checkpoint()
    print("\nReady for training if all components are available.")
