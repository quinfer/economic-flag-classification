# Hierarchical Flag Classification through Economic Domain Knowledge: A Vision Transformer Approach for Cultural Symbol Recognition

**MSc Artificial Intelligence - Themed Research Project**  
**Researcher**: Barry Quinn, Professor of Finance  
**Supervisor**: Dr. Shuyan Li  
**Institution**: Queen's University Belfast  
**Year**: 2024-2025

## Project Overview

This work introduces a novel flag classification task addressing real-world challenges in cultural symbol recognition. We collected a new dataset of Northern Ireland street imagery and developed a hierarchical classification framework guided by economic domain knowledge. Rather than claiming to “solve” a 70-class fine-grained problem by collapsing labels, we explicitly formulate a task with a principled taxonomy designed for practical applications and measurable evaluation.

Key contributions:
- New dataset of 4,501 images from Northern Ireland for cultural symbol recognition
- Hierarchical taxonomy design (70 → 16 → 7) informed by economic domain knowledge
- Strong baseline models based on a ViT-H-14 backbone, achieving 94.78% accuracy on the 7-category task

Data availability: This repository contains code, methodology, and summary results. The underlying Google Street View imagery cannot be publicly shared due to licensing restrictions and terms of service limitations.

## Task Formulation and Approach

We frame flag recognition as hierarchical classification with three granularity levels: (1) 70 fine-grained classes, (2) 16 semantically grouped classes, and (3) 7 economically meaningful categories. Economic domain knowledge guides the taxonomy design using concentration concepts (e.g., HHI) to articulate community impact and practical relevance. This framing clarifies the target task and evaluation protocol without positioning taxonomy design as a remedy for class imbalance.

## Empirical Results

- Granularity comparison (same data source, different taxonomies):
  - 70-class fine-grained baseline: 40.8% accuracy (challenging reference)
  - 16-class intermediate grouping: 72.6% accuracy (semantic consolidation)
  - 7-class economic taxonomy: 94.78% accuracy (task-focused baseline)
- Architecture: RS5M ViT-H-14 (Vision Transformer pre-trained on remote sensing data)
- Dataset: 4,501 images (2,030 training; 2,471 testing)
- Validation: multi-seed testing and 5-fold cross-validation conducted for robustness

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
# Execute primary training pipeline for the 7-category economic taxonomy
python train_economic_consolidation.py

# Alternative: use configuration-based training
python train.py --config configs/rs5m_economic_consolidation.yaml
```

### Results Reproduction

Multi-seed validation (reference: 94.78% ± 0.22% on 7-category task):
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
│   └── NIFlagsV2/          # Flag dataset structure and splits metadata
├── configs/                # Training configurations  
├── trainers/               # Model implementations
├── scripts/                # Evaluation and figure generation
├── figures/                # Publication-quality result figures
├── docs/
│   ├── thesis_paper.pdf    # Complete academic paper
│   └── METHODOLOGY.md      # Technical methodology (hierarchical taxonomy)
└── checkpoints/            # Model checkpoints (download separately)
```

## Dataset Description

Google Street View terms prohibit redistribution of raw imagery. This repository provides taxonomy, splits metadata, scripts, and summary results. Researchers should obtain street-level imagery via appropriate licensing channels.

Dataset overview:
- 4,501 images from Northern Ireland street scenes
- Splits: 2,030 training, 2,471 testing
- Original taxonomy: 70 fine-grained classes
- Proposed hierarchy: 70 → 16 → 7 (economically meaningful categories)

Economic categories (largest to smallest by count):
1. Major_Unionist — 2,047 samples
2. Cultural_Fraternal — 892 samples  
3. International — 485 samples
4. Nationalist — 354 samples
5. Paramilitary — 312 samples
6. Commemorative — 233 samples
7. Sport_Community — 178 samples

## Methodology

1. Hierarchical Taxonomy Design: economically grounded grouping from 70 → 16 → 7
2. RS5M ViT-H-14: Vision transformer backbone and training setup  
3. Hierarchical Prompting: multi-level attention steering aligned to taxonomy
4. Quality Control: confidence filtering and inter-annotator reliability

## Reproducibility

Experiments use fixed seeds (42, 123, 456) and documented hyperparameters:
- Batch size: 8
- Learning rate: 1e-4  
- Epochs: 30
- Optimizer: AdamW with differential learning rates

## Academic Paper

The complete manuscript is available at `docs/thesis_paper.pdf`. It details:
- Task definition and dataset contribution
- Hierarchical taxonomy design using economic domain knowledge
- Baseline models and evaluation protocol
- Statistical validation (multi-seed, cross-validation)
- Figures available in `figures/`

## Citation

```bibtex
@mastersthesis{quinn2025hierarchical,
  title={Hierarchical Flag Classification through Economic Domain Knowledge: A Vision Transformer Approach for Cultural Symbol Recognition},
  author={Quinn, Barry},
  school={Queen's University Belfast},
  year={2025},
  type={MSc Thesis}
}
```

## Contact

For inquiries regarding methodology, reproduction, or collaboration:  
Barry Quinn — b.quinn@ulster.ac.uk

## Acknowledgments

The author acknowledges Dr. Shuyan Li for supervision and methodological guidance. Appreciation is extended to the RS5M development team for providing pre-trained model weights under open license.

---
*This work forms part of an MSc Artificial Intelligence thesis at Queen's University Belfast. The experimental phase is complete; manuscript preparation follows the reframed task specification.*
