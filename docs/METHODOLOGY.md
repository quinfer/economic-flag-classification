# Methodology Documentation

## Dataset and Task

### Source Data
- GroundingDINO detections over Google Street View imagery (2022–2023)
- Expert verification and category labeling for cultural symbols (flags)
- Quality control: confidence filtering ≥ 3.0 (1–5 scale)

### Dataset Summary
- 4,501 images from Northern Ireland street scenes
- Splits: 2,030 training; 2,471 testing
- Original label space: 70 fine-grained classes (Category–MountType–SpecificFlag)

## Hierarchical Taxonomy Design

We structure recognition as hierarchical classification with three levels of granularity:
- Fine-grained (70): original categories capturing specific symbols and contexts
- Intermediate (16): semantically grouped classes
- Economic (7): culturally and economically meaningful categories

Economic domain knowledge guides the taxonomy using concentration and community-impact concepts (e.g., HHI) to motivate which symbols are practically grouped for downstream applications. This is a principled task design choice—not a claim to solve imbalance—intended to reflect societal relevance and decision-making needs.

Economic categories and counts:
1. Major_Unionist — 2,047 samples
2. Cultural_Fraternal — 892 samples
3. International — 485 samples
4. Nationalist — 354 samples
5. Paramilitary — 312 samples
6. Commemorative — 233 samples
7. Sport_Community — 178 samples

## Model and Training

### Backbone
- RS5M ViT-H-14 (Vision Transformer pre-trained on remote sensing imagery)
- Rationale: strong spatial modeling applicable to territorial markers

### Training Configuration
- Optimizer: AdamW (differential learning rates)
- Batch size: 8
- Learning rate: 1e-4
- Epochs: 30
- Seeds: 42, 123, 456
- Augmentation: random crop, flip, color jitter

## Results

### Granularity Comparison (same data source)
- 70-class (fine-grained): 40.8% accuracy — challenging baseline
- 16-class (intermediate): 72.6% accuracy — semantic grouping
- 7-class (economic): 94.78% accuracy — proposed hierarchical task baseline

### Attention Analysis
- Attention on flag regions increases from ~23% (generic setup) to ~87% (hierarchical prompting aligned to taxonomy), indicating improved focus on task-relevant regions.

### Statistical Validation
- Multi-seed: 94.57% ± 0.22%
- 5-fold cross-validation: 93.23% ± 0.34%
- Macro-F1: 67.45%
