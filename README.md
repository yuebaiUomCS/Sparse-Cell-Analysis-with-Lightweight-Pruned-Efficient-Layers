# Sparse Cell Analysis with Lightweight Pruned Efficient Layers

[![python >3.8](https://img.shields.io/badge/python-3.8+-brightgreen)](https://www.python.org/) 

### Bayesian Sparse Transformer for Single-cell RNA Sequencing Data Analysis

This project implements a Bayesian Sparse Transformer architecture for single-cell RNA sequencing data analysis, with structured pruning capabilities and uncertainty quantification. The model combines:

- **Bayesian Neural Networks**: For uncertainty quantification in predictions
- **Structured Pruning**: To achieve model compression while maintaining performance  
- **Two-Stage Training**: Stage A (deterministic pretraining) + Stage B (Bayesian fine-tuning with pruning)
- **Performer Architecture**: Efficient transformer implementation for genomic data

The approach enables robust cell type annotation with calibrated uncertainty estimates and efficient model deployment through structured sparsity.

## Installation

[![scipy-1.5.4](https://img.shields.io/badge/scipy-1.5.4-yellowgreen)](https://github.com/scipy/scipy) [![torch-1.8.1](https://img.shields.io/badge/torch-1.8.1-orange)](https://github.com/pytorch/pytorch) [![numpy-1.19.2](https://img.shields.io/badge/numpy-1.19.2-red)](https://github.com/numpy/numpy) [![pandas-1.1.5](https://img.shields.io/badge/pandas-1.1.5-lightgrey)](https://github.com/pandas-dev/pandas) [![scanpy-1.7.2](https://img.shields.io/badge/scanpy-1.7.2-blue)](https://github.com/theislab/scanpy)

## Key Features

- **Bayesian Uncertainty Quantification**: Monte Carlo sampling for prediction uncertainty
- **Structured Pruning**: Gradual pruning with fine-tuning for model compression
- **Two-Stage Training Pipeline**: 
  - Stage A: Deterministic pretraining
  - Stage B: Bayesian fine-tuning with optional pruning

## Model Architecture

The model uses a Performer-based transformer with:
- Bayesian linear layers for uncertainty quantification
- Structured pruning capabilities in fully connected layers

## Data Preparation

The input data should be in `.h5ad` format (AnnData). Preprocessing steps:

1. Gene symbol revision according to NCBI Gene database
2. Remove unmatched and duplicated genes  
3. Normalize with `sc.pp.normalize_total` and `sc.pp.log1p`

*Note: Data preprocessing utilities are not included in this core repository.*

## Usage

### Two-Stage Training Pipeline

**Stage A: Deterministic Pretraining**
```bash
python finetune_sparse.py \
    --data_path ./data/STARmap_merged_fixed.h5ad \
    --gene_num 166 \
    --bin_num 5 \
    --epoch 100 \
    --batch_size 4 \
    --learning_rate 1e-4 \
    --pos_embed
```

**Stage B: Bayesian Fine-tuning with Pruning**
```bash
python finetune_sparse.py \
    --data_path ./data/STARmap_merged_fixed.h5ad \
    --model_path ./stage_A_best.pth \
    --bayes_head \
    --kl_weight 1e-6 \
    --prune_target 0.3 \
    --prune_step 0.1
```

### Automated Pipeline Scripts

**Complete A+B Pipeline:**
```bash
./train_sparse.sh
```

**Stage B Only (Round 1 - Bayesian training):**
```bash
./stageb_round1.sh
```

**Stage B Only (Round 2 - Secondary pruning):**
```bash
START_CKPT=/path/to/round1_best.pth ./stageb_round2.sh
```

### Prediction and Evaluation

**Standard Prediction:**
```bash
python predict_updated.py \
    --data_path ./data/test_data.h5ad \
    --model_path ./final_model_best.pth \
    --bayes_mc_samples 8
```

*Note: Evaluation and analysis scripts are not included in this core repository.* 

## Key Parameters

### Model Architecture
| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `gene_num` | Number of genes | 166 | Dataset specific |
| `bin_num` | Expression bins | 5 | [3, 5, 7] |
| `dim` | Embedding dimension | 200 | [100, 200, 300] |
| `heads` | Attention heads | 10 | [8, 10, 16] |
| `depth` | Transformer layers | 6 | [4, 6, 8] |

### Bayesian Settings
| Parameter | Description | Default |
|-----------|-------------|---------|
| `kl_weight` | KL divergence weight | 1e-6 |
| `kl_steps` | KL annealing steps | 10000 |
| `mc_samples` | Monte Carlo samples (train) | 1 |
| `mc_samples_eval` | Monte Carlo samples (eval) | 8 |

### Pruning Settings
| Parameter | Description | Default |
|-----------|-------------|---------|
| `prune_target` | Target sparsity | 0.3 |
| `prune_step` | Pruning increment | 0.1 |
| `prune_ft_epochs` | Fine-tune epochs per step | 4 |
| `prune_layers` | Layers to prune | "fc1,fc2" |

## Project Structure

```
├── Core Training Scripts
│   ├── finetune_sparse.py           # Unified sparse training (Stage A & B)
│   └── finetune_updated.py          # Baseline scBERT training
│
├── Model Architecture
│   ├── bayesian_sparse_transformer.py  # Core Bayesian model
│   ├── utils.py                        # Utility functions
│   └── performer_pytorch/              # Custom Performer implementation
│
├── Automation Scripts
│   ├── train_sparse.sh             # Complete A+B training pipeline
│   ├── stageb_round1.sh           # Bayesian training only
│   └── stageb_round2.sh           # Secondary pruning only
│
└── Configuration
    ├── README.md                  # This file
    └── LICENSE                   
```

## Expected Output

- **Cell Type Predictions**: Classification results for each cell
- **Uncertainty Estimates**: Bayesian uncertainty quantification via Monte Carlo sampling
- **Model Checkpoints**: Saved models at each training stage
- **Pruning Statistics**: Sparsity levels and performance metrics
- **Cross-validation Results**: Performance across different data splits

## Requirements

- Python >= 3.8
- PyTorch >= 1.8.1
- CUDA-capable GPU (recommended)
- 16GB+ RAM for large datasets

## Core Files for GitHub Repository

### Essential Core Files Included:

**Training Scripts:**
- `finetune_sparse.py` - Unified sparse training (handles both Stage A & B)
- `finetune_updated.py` - Baseline scBERT training implementation

**Model Architecture:**
- `bayesian_sparse_transformer.py` - Core Bayesian model implementation
- `utils.py` - Essential utility functions
- `performer_pytorch/` - Custom Performer implementation directory

**Automation Scripts:**
- `train_sparse.sh` - Complete A+B training pipeline
- `stageb_round1.sh` - Bayesian training only
- `stageb_round2.sh` - Secondary pruning only

**Configuration & Documentation:**
- `requirements.txt` - Python dependencies
- `README.md` - This documentation
- `LICENSE` - GPL v3 License file

### Files Excluded from Core Repository:
- Evaluation and analysis scripts
- Data processing and visualization tools
- Model checkpoints (`.pth` files)
- Dataset-specific files (`label/`, `*.h5ad`)
- Transfer learning utilities

## References

This implementation builds upon and extends the following works:

1. **Original scBERT**: [TencentAILabHealthcare/scBERT](https://github.com/TencentAILabHealthcare/scBERT.git)
   - Base transformer architecture for single-cell analysis
   - Pre-training methodology for genomic data

2. **scBERT Reusability Study**: [TranslationalBioinformaticsUnit/scbert-reusability](https://github.com/TranslationalBioinformaticsUnit/scbert-reusability)
   - Reproducibility improvements and evaluation frameworks
   - Cross-dataset validation approaches

### Key Innovations in This Implementation:
- **Bayesian uncertainty quantification** via variational inference
- **Structured neural network pruning** for model compression
- **Two-stage training pipeline** with deterministic pretraining
- **Progressive pruning** with performance monitoring
- **Robust cross-validation** and evaluation metrics

## Contributing

This project implements advanced techniques in:
- Bayesian deep learning for genomics
- Structured neural network pruning  
- Uncertainty quantification in biological data
- Efficient transformer architectures

## Citation

If you use this code in your research, please cite the original scBERT paper and acknowledge the referenced repositories:

```bibtex
@article{yang2022scbert,
  title={scBERT as a large-scale pretrained deep language model for cell type annotation of single-cell RNA-seq data},
  author={Yang, Fan and Wang, Wenchuan and Wang, Fang and Fang, Yuan and Tang, Duyu and Huang, Junzhou and Lu, Hui and Yao, Jianhua},
  journal={Nature Machine Intelligence},
  volume={4},
  number={10},
  pages={852--866},
  year={2022},
  publisher={Nature Publishing Group}
}
```

## License

This project is for research purposes. Please cite appropriately if used in publications.

## Contact

For questions regarding this implementation, please create an issue in this repository.
