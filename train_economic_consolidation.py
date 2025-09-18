#!/usr/bin/env python3
"""
Hierarchical Taxonomy Training Script (Economic 7-category task)
Reference: hierarchical flag classification guided by economic domain knowledge.

Reference results (same data source, different taxonomies):
- 70-class (fine-grained): ~40.8% accuracy
- 16-class (intermediate): ~72.6% accuracy
- 7-class (economic): ~94.78% accuracy; Macro-F1 ~67.5%
- Attention focus on flag regions ~87% under hierarchical prompting

Usage:
    python train_economic_consolidation.py --seed 42
    python train_economic_consolidation.py --seed 123  
    python train_economic_consolidation.py --seed 456
"""

import argparse
import json
import random
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from torch.utils.data import DataLoader
import numpy as np

# Paper-specified hyperparameters
HYPERPARAMETERS = {
    'batch_size': 8,
    'learning_rate': 1e-4,
    'epochs': 30,
    'optimizer': 'AdamW',
    'seeds': [42, 123, 456]
}

# Economic consolidation categories (from paper)
ECONOMIC_CATEGORIES = {
    'Major_Unionist': 2047,
    'Cultural_Fraternal': 892, 
    'International': 485,
    'Nationalist': 354,
    'Paramilitary': 312,
    'Commemorative': 233,
    'Sport_Community': 178
}

def set_seed(seed: int):
    """Set random seed for reproducibility (paper requirement)"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def load_dataset_splits(data_root: Path):
    """Load available splits and report counts (no hard-coded expectations)."""
    split_names = ['train', 'val', 'test']
    splits = {}
    for split_name in split_names:
        split_file = data_root / f"{split_name}.txt"
        if split_file.exists():
            with open(split_file) as f:
                lines = f.readlines()
            splits[split_name] = len(lines)
        else:
            # Mark missing splits explicitly
            splits[split_name] = 0

    total = sum(splits.values())
    print("📊 Dataset splits detected:")
    for k in split_names:
        present = "present" if splits[k] > 0 else "missing"
        print(f"   {k:>5}: {splits[k]} samples ({present})")
    print(f"   Total (across present splits): {total}")

    # If none found, raise a clear error
    if all(splits[k] == 0 for k in split_names):
        raise FileNotFoundError(f"No split files found under {data_root}")

    return splits


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Economic Consolidation Training')
    parser.add_argument('--seed', type=int, default=42, choices=[42, 123, 456],
                       help='Random seed (paper uses 42, 123, 456)')
    parser.add_argument('--data-root', type=str, default='datasets/NIFlagsV2',
                       help='Dataset root directory')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/RS5M_ViT-H-14.pt',
                       help='RS5M ViT-H-14 checkpoint path')
    parser.add_argument('--output-dir', type=str, default='experiments/economic_consolidation',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    print("🚀 Economic Consolidation Training")
    print("=" * 50)
    print("Paper: Hierarchical Flag Classification through Economic Domain Knowledge")
    print(f"Seed: {args.seed}")
    print(f"Expected Results: 94.78% accuracy, 67.5% macro-F1")
    print()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Load and verify dataset splits
    data_root = Path(args.data_root)
    splits = load_dataset_splits(data_root)
    
    # Verify economic consolidation categories
    print(f"\n📋 Economic Categories (from paper):")
    total_samples = sum(ECONOMIC_CATEGORIES.values())
    for category, count in ECONOMIC_CATEGORIES.items():
        percentage = (count / total_samples) * 100
        print(f"   {category}: {count} ({percentage:.1f}%)")
    print(f"   Total: {total_samples}")
    
    # Training configuration
    print(f"\n⚙️ Training Configuration:")
    for key, value in HYPERPARAMETERS.items():
        if key != 'seeds':
            print(f"   {key}: {value}")
    
    print(f"\n✅ Setup complete. Ready for training with seed {args.seed}")
    print(f"💾 Results will be saved to: {args.output_dir}")
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save configuration for reproducibility
    config = {
        'seed': args.seed,
        'hyperparameters': HYPERPARAMETERS,
        'economic_categories': ECONOMIC_CATEGORIES,
        'dataset_splits': splits,
        'expected_results': {
            'accuracy': 94.78,
            'macro_f1': 67.45,
            'attention_improvement': '23% → 87%'
        }
    }
    
    with open(Path(args.output_dir) / f'config_seed_{args.seed}.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"📝 Configuration saved: {args.output_dir}/config_seed_{args.seed}.json")


if __name__ == "__main__":
    main()
