#!/usr/bin/env python3
"""
Setup verification for Economic Consolidation reproduction
"""

from pathlib import Path

def verify_dataset():
    """Check dataset structure and splits"""
    data_root = Path("datasets/NIFlagsV2")
    
    print("Dataset verification:")
    
    # Check split files exist and have expected counts
    expected_counts = {'train': 3823, 'val': 841, 'test': 826}
    
    for split, expected in expected_counts.items():
        split_file = data_root / f"{split}.txt"
        if split_file.exists():
            with open(split_file) as f:
                actual = len(f.readlines())
            print(f"  {split}: {actual} samples (expected: {expected})")
        else:
            print(f"  {split}: MISSING")
    
    # Check class names
    classnames_file = data_root / "classnames.txt"
    if classnames_file.exists():
        with open(classnames_file) as f:
            classes = f.read().strip().split('\n')
        print(f"  Classes: {len(classes)} categories")
    
    print(f"  Total expected: 5,490 samples across 7 economic categories")

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
