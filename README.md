# GraPPI: a unified graph-based deep learning framework designed specifically for protein complex modeling
## Overview

**GraPPI** is a self-supervised graph neural network framework for protein-protein interaction (PPI) modeling. It includes: (1) self-supervised pre-training via masked edge prediction on protein complex graphs; (2) frozen-encoder fine-tuning on three downstream tasks — binding mode classification, and binding affinity ($\Delta G$) regression, mutational binding affinity upon mutation (mut $\Delta G$) regression.

The core encoder of GraPPI is an **Edge-Enhanced Heterogeneous Graph Transformer (EdgeEnhancedHGT)** that processes protein complexes as heterogeneous graphs with receptor and ligand node types (binding parts 1 and 2) and 3 edge types (intra-receptor, intra-ligand, and bidirectional inter-molecular edges). Node features encode amino acid identity with ESM-2 and physicochemical properties (optionally residue one-hot encoded for sequence identity), while edge features capture inter-residue distance histograms and direction vectors.

## Abstract

Protein-protein interactions (PPIs) are fundamental to cellular processes and prime targets for therapeutic development. While recent AI advances, including protein language models, excel at capturing sequence and monomeric structure representations, they often inadequately model the intricate structural, geometrical, and physicochemical contexts of protein complexes—especially at the binding interface. Consequently, existing AI approaches for PPIs often rely on fragmented, task-specific models with limited model generalizability.
To bridge this gap, we developed GraPPI, a unified graph-based deep learning framework designed specifically for protein complex modeling. GraPPI represents complexes as heterogeneous graphs, using residues as nodes and spatial relationships as edges. At its core is PPIencoder, a self-supervised model that acts as a “language” encoder for protein complexes, learning transferable representations by explicitly capturing the precise geometric and physicochemical environment of protein interfaces. GraPPI then leverages these pretrained embeddings for critical downstream evaluations: classifying binders with plausible interface against decoys and predicting binding affinity.

## Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Phase 1: SSL Pre-training                        │
│                                                                     │
│   PDB complexes → Heterogeneous Graphs → EdgeEnhancedHGT Encoder    │
│                         ↓                                           │
│         Masked Edge Prediction (inter + intra molecular)            │
│                         ↓                                           │
│             Pre-trained Encoder Weights                             │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────────┐
│               Phase 2: Downstream Fine-tuning                       │
│                                                                     │
│  Load frozen encoder → Precompute node embeddings (+ optional JK)   │
│  Free encoder from memory → Train Pool+Head models only             │
│                                                                     │
│  ┌─────────────┐  ┌────────────────┐  ┌───────────────────┐         │
│  │ ΔG Regressor│  │mut ΔG Regressor│  │ Binder Classifier │         │
│  │ (5-fold CV) │  │   (5-fold CV)  │  │    (5-fold CV)    │         │
│  │    (test)   │  │     (test)     │  │      (test)       │         │
│  └─────────────┘  └────────────────┘  └───────────────────┘         │
└─────────────────────────────────────────────────────────────────────┘
```

## Key Features

- **Unified training pipeline** via YAML configs (`train_unified.py`)
- **Heterogeneous graph representation** with gated edge message passing + HGT layers
- **Multiple SSL strategies**: dynamic, stratified, curriculum, hard negative sampling
- **Frozen-encoder paradigm**: precompute embeddings once, train lightweight heads efficiently
- **Jumping Knowledge (JK-Net)**: aggregate intermediate HGT layer outputs for richer representations
- **Cross-attention aggregation**: learnable queries with cross-chain interaction for graph-level pooling
- **Flexible node features**: base (25-dim), ESM-2 (1285-dim) or concatenated variants
- **Baseline comparison**: MLP and ML models (RandomForest, GradientBoosting, SVM) for benchmarking

## Data

Original datasets, curated datasets, Dunbrack library, and training outputs are not included in this repo. They are stored at [OneDrive Link](https://liveutmb-my.sharepoint.com/:f:/r/personal/hazhao_utmb_edu/Documents/ZhaoLab_Files/ZS?csf=1&web=1&e=IBx1YC).

## Installation

**Verified versions:** Python 3.12, PyTorch 2.7.1+cu128, PyTorch Geometric 2.7.0

```bash
conda create -n GraPPI python=3.12
conda activate GraPPI
conda install -c bioconda usalign
pip install pandas biopython tqdm openpyxl matplotlib seaborn numpy networkx PyYAML
pip install -U scikit-learn
pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 torchaudio==2.7.1+cu128 --index-url https://download.pytorch.org/whl/cu128
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster -f https://data.pyg.org/whl/torch-2.7.1+cu128.html
```

> **Note:** Use the full Python path with `nohup` — `conda activate` does not propagate to background processes:
> ```bash
> nohup /path/to/miniconda3/envs/GraPPI/bin/python train_unified.py -config model_configs/config_ssl.yaml > train_unified.out 2>&1 &
> ```

## Quick Start

```bash
# Full pipeline (SSL + all downstream tasks)
python train_unified.py -config model_configs/config_full_pipeline.yaml

# SSL pre-training only
python train_unified.py -config model_configs/config_ssl.yaml

# Individual downstream tasks (requires pretrained encoder)
python train_unified.py -config model_configs/config_dG.yaml
python train_unified.py -config model_configs/config_ddG.yaml
python train_unified.py -config model_configs/config_disc_binder.yaml

# Baseline models (no pretrained encoder)
python train_unified.py -config model_configs/config_baseline.yaml
```

## Repository Structure

```
train_unified.py              # Entry point — orchestrates all training phases
SSL_train_module.py           # SSL pre-training orchestration
finetune_dG_module.py         # ΔG regression fine-tuning (5-fold CV + test)
finetune_ddG_module.py        # ΔΔG mutation fine-tuning (5-fold CV + test)
finetune_disc_binder_module.py# Binder discrimination fine-tuning
baseline_test_modules.py      # Baseline NN and ML model training

model_configs/                # YAML configuration files
  config_full_pipeline.yaml   #   Full SSL + finetune pipeline
  config_ssl.yaml             #   SSL pre-training only
  config_dG.yaml              #   ΔG regression only
  config_ddG.yaml             #   ΔΔG mutation only
  config_disc_binder.yaml     #   Binder discrimination only
  config_baseline.yaml        #   Baseline models
  for_multirun/               #   Auto-generated configs for grid search

model_collection/             # Model definitions
  EMP_HGT.py                 #   Core encoder: EdgeEnhancedHGT + CrossAttentionAggregator
  SSLModels.py                #   SSL models: SSLEdgePredictor, SSLInterIntraEdgePredictor
  FineTuneModels.py           #   Pool+Head models: Classifier, Regressor, Gated, DDG
  FineTune_baselineModels.py  #   Baseline MLP models
  HGTConv_fixed.py            #   Fixed HGTConv implementation
  HGTConv_with_attention.py   #   Attention extraction utilities

utils/
  gen_graphs_unified.py       #   Graph generation from PDB structures
  gen_heterograph_ba.py       #   Heterogeneous graph builder with binding affinity
  SSL_modules/                #   SSL data loading, masking strategies, transforms
  Training_modules/           #   Training loops, early stopping, data loaders
  Quantity_compute/           #   Loss functions, metrics, dimensionality reduction
  database_process/           #   Database curation utilities

run_multiple_ssl.sh           # Batch SSL training over architecture grid
run_multiple_finetune.sh      # Batch fine-tuning over architecture grid
```

## Documentation

- **[README_UNIFIED_TRAINING.md](README_UNIFIED_TRAINING.md)** — Unified training pipeline details and configuration reference
- **[README_SSL_EDGE_PREDICTOR.md](README_SSL_EDGE_PREDICTOR.md)** — SSL edge prediction model and training strategies
- **[MODEL_ARCHITECTURE.md](MODEL_ARCHITECTURE.md)** — Detailed model architecture with mathematical formulations

