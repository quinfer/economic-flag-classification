# Economic Consolidation for Extreme Class Imbalance: Flag Classification

**MSc Artificial Intelligence - Themed Research Project**  
**Researcher**: Barry Quinn, Professor of Finance  
**Supervisor**: Dr. Shuyan Wang  
**Institution**: Queen's University Belfast  
**Year**: 2024-2025

## Project Overview

This research investigates the application of economic concentration theory to address extreme class imbalance in computer vision tasks. The study focuses on flag classification within Northern Ireland street imagery, a domain characterized by severe class imbalance (169:1 ratio) that renders traditional machine learning approaches ineffective.

**Key Contribution**: Economic consolidation of 70 fine-grained classes into 7 theoretically-motivated categories achieves 94.78% classification accuracy, representing a 169-fold improvement over baseline performance (0.56%).

**Data Availability Notice**: This repository contains code, methodology, and results only. The underlying Google Street View imagery cannot be publicly shared due to licensing restrictions and terms of service limitations.

## Research Approach

This work demonstrates that domain-specific theoretical frameworks can provide superior solutions to extreme class imbalance compared to traditional data engineering approaches. The methodology employs the Herfindahl-Hirschman Index (HHI) from industrial economics to consolidate flag classes based on their economic and social impact within Northern Ireland communities.

The approach challenges conventional wisdom in machine learning by prioritizing domain knowledge over algorithmic complexity, showing that theoretically-grounded class consolidation substantially outperforms standard techniques such as oversampling, undersampling, and focal loss implementations.

## Empirical Results

- **Classification Accuracy**: 94.78% (baseline: 0.56%)
- **Performance Improvement**: 169-fold increase over conventional approaches
- **Architecture**: RS5M ViT-H-14 (Vision Transformer pre-trained on remote sensing data)
- **Dataset**: 5,490 expert-annotated flag instances from Northern Ireland street imagery
- **Statistical Validation**: Multi-seed testing and 5-fold cross-validation confirm robustness

## Implementation

**System Requirements**: GPU with minimum 8GB VRAM recommended for full model training. Development and testing conducted on Mac Studio M4 Max configuration.

### Environment Setup
```bash
# Install required dependencies
pip install -r requirements.txt

# Download RS5M pre-trained model (3.8GB)
wget https://github.com/om-ai-lab/RS5M/releases/download/v1.0/RS5M_ViT-H-14.pt -P checkpoints/

# Verify setup
python verify_setup.py
```

### Training Protocol
```bash
# Execute primary training pipeline with economic consolidation
python train_economic_consolidation.py

# Alternative: use configuration-based training
python train.py --config configs/rs5m_economic_consolidation.yaml
```

### Results Reproduction

**Multi-seed validation** (reproduces 94.78% ± 0.22%):
```bash
python train_economic_consolidation.py --seed 42 --output-dir results/seed_42
python train_economic_consolidation.py --seed 123 --output-dir results/seed_123  
python train_economic_consolidation.py --seed 456 --output-dir results/seed_456
```

**Analysis and figures**:
```bash
# Compute aggregated metrics
python scripts/compute_metrics.py --results-dir results/

# Generate figures for publication
python scripts/simple_real_figures.py
```

## Repository Structure

```
├── train.py                 # Main training script (RS5M ViT-H-14)
├── requirements.txt         # Dependencies
├── datasets/
│   └── NIFlagsV2/          # Flag dataset (5,490 samples)
├── configs/                # Training configurations  
├── trainers/               # Model implementations
├── scripts/                # Evaluation and figure generation
├── checkpoints/            # Model checkpoints (download separately)
└── docs/                   # Methodology documentation
```

## Dataset Description

**Data Availability**: Google Street View terms of service explicitly prohibit redistribution of raw imagery. Consequently, this repository provides dataset structure, annotations, class mappings, and metadata only. Researchers seeking to reproduce this work must obtain their own street-level imagery through appropriate licensing channels.

The dataset comprises 5,490 flag instances from Northern Ireland, derived from a comprehensive street-level imagery analysis. The original taxonomy contained 70 fine-grained classes exhibiting extreme imbalance (169:1 ratio). The core methodological innovation involves consolidating these into 7 economically-motivated categories based on community impact theory.

**Economic Class Taxonomy**:
1. **Major_Unionist** (2,047 instances) - Ulster Banner, Union Jack territorial displays
2. **Cultural_Fraternal** (892 instances) - Orange Order, GAA organizational symbols  
3. **International** (485 instances) - EU, foreign national representations
4. **Nationalist** (354 instances) - Irish tricolor territorial markers
5. **Paramilitary** (312 instances) - UVF, UDA, associated symbols
6. **Commemorative** (233 instances) - Historical, military memorial displays
7. **Sport_Community** (178 instances) - Local sporting club identifiers

## Methodology

1. **Economic Consolidation**: Groups classes by community impact using HHI concentration theory
2. **RS5M ViT-H-14**: Vision transformer pre-trained on remote sensing imagery  
3. **Hierarchical Prompting**: Multi-level attention steering for economic categories
4. **Quality Control**: Confidence filtering and inter-annotator reliability

## Reproducibility

All experiments use fixed seeds (42, 123, 456) and documented hyperparameters:
- Batch size: 8
- Learning rate: 1e-4  
- Epochs: 30
- Optimizer: AdamW with differential learning rates

## Citation

```bibtex
@mastersthesis{quinn2025economic,
  title={Economic Concentration as Domain Knowledge for Extreme Class Imbalance: A Case Study in Flag Classification},
  author={Quinn, Barry},
  school={Queen's University Belfast},
  year={2025},
  type={MSc Thesis}
}
```

## Contact

For inquiries regarding methodology, reproduction, or collaboration opportunities:  
Barry Quinn - b.quinn@ulster.ac.uk

## Acknowledgments

The author acknowledges Dr. Shuyan Wang for supervision and methodological guidance, particularly in identifying and resolving critical implementation issues that initially produced artifactually high performance metrics. Appreciation is extended to the RS5M development team for providing pre-trained model weights under open license.

---
*This work forms part of an MSc Artificial Intelligence thesis at Queen's University Belfast. The experimental phase is complete; manuscript preparation is in progress.*
