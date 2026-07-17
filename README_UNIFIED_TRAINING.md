# Unified Training Pipeline

This document describes the `train_unified.py` entry point, the YAML configuration system, and each downstream training task.

## Overview

`train_unified.py` is the single entry point for all training in EntroPPI. It loads a YAML config, sets the seed and device, then dispatches to one or more task modules in sequence:

```
train_unified.py  →  YAML config
                      ↓
              ┌───────────────────────────────────────────────────┐
              │  training_task list selects which phases to run:  │
              │                                                   │
              │  ssl         → SSL_train_module.py                │
              │  dg_reg      → finetune_dG_module.py              │
              │  mutation    → finetune_ddG_module.py             │
              │  disc_binder → finetune_disc_binder_module.py     │
              │  baseline    → baseline_test_modules.py           │
              └───────────────────────────────────────────────────┘
```

All downstream tasks (dg_reg, mutation, disc_binder) follow a **precomputed-embedding** paradigm:

1. Load pretrained SSL encoder (frozen, eval mode)
2. Forward-pass every graph through the encoder to get node-level embeddings
3. Optionally concatenate ESM features or use JK-Net aggregation
4. Store enriched embeddings in `graph.x`; delete the encoder from memory
5. Train a lightweight **Pool + Head** model on the precomputed embeddings

## Files

| File | Purpose |
|------|---------|
| `train_unified.py` | Entry point — config loader and task dispatcher |
| `SSL_train_module.py` | SSL masked edge prediction training loop |
| `finetune_dG_module.py` | ΔG regression with 5-fold CV + held-out test |
| `finetune_ddG_module.py` | ΔΔG mutation regression (paired mut/wt graphs) |
| `finetune_disc_binder_module.py` | Binary binder classification |
| `baseline_test_modules.py` | Baseline NN (MLP) and sklearn ML models |
| `model_configs/config_full_pipeline.yaml` | Full pipeline config template |
| `model_configs/config_ssl.yaml` | SSL-only config |
| `run_multiple_ssl.sh` | Batch SSL training across architecture grid |
| `run_multiple_finetune.sh` | Batch finetune across pretrained checkpoints |

## Quick Start

### Run SSL pre-training only

```bash
python train_unified.py -config model_configs/config_ssl.yaml
```

### Run downstream fine-tuning only

Set `pretrained_path` in the task section of your config to an existing SSL checkpoint, then run:

```bash
python train_unified.py -config model_configs/config_full_pipeline.yaml
```

### Run full pipeline (SSL → downstream)

Include both `ssl` and the desired downstream tasks in `training_task`:

```yaml
training_task:
  - ssl
  - dg_reg
  - mutation
  - disc_binder
```

The SSL checkpoint path is passed automatically to each downstream task.

## Configuration Reference

All settings live in a single YAML file. The top-level keys are:

### `training_task`

A list of phases to run in order. Valid values: `ssl`, `dg_reg`, `mutation`, `disc_binder`, `baseline`.

```yaml
training_task:
  - ssl
  - dg_reg
  - mutation
```

### `system`

```yaml
system:
  cuda_id: 0     # GPU device ID
  seed: 42       # Random seed
```

### `data`

```yaml
data:
  pdb_root: /path/to/curated_db
  dist: "8"                    # Distance threshold for graph edges (Angstroms)
  embedding_type: esm          # base | esm | esm480 | +esm | +esm480 | only_esm
  min_nodes: 5                 # Minimum nodes per chain
```

`embedding_type` controls node feature dimensionality (base=25, esm=1285, esm480=485). When loading a pretrained checkpoint, the saved `embedding_type` overrides this value.

### `ssl`

Controls the SSL masked edge prediction phase. See [README_SSL_EDGE_PREDICTOR.md](README_SSL_EDGE_PREDICTOR.md) for architecture details.

```yaml
ssl:
  # Edge masking
  edge_mask_ratio: 0.25
  use_intra_edges: True
  intra_edge_mask_ratio: 0.15
  intra_edge_beta: 0.65         # Loss = beta * L_inter + (1-beta) * L_intra
  negative_ratio: 1.0
  min_interface_edges: 5
  strategy_type: dynamic         # dynamic | stratified | curriculum | hard_negative

  # Encoder architecture
  num_layers: 2
  hidden_dim_power: 9            # Hidden dim = 2^power
  hgt_heads: 4
  dropout: 0.2
  message_style: gated_src       # gated_src | additive
  use_checkpoint: True

  # Training
  batch_size: 40
  n_epochs: 1000
  lr: 0.0001
  weight_decay: 0.001
  clip_max_norm: 5.0
  patience: 50
  scheduler_t0: 10

  save_dir: /path/to/output/
```

### `dg_reg`

ΔG binding affinity regression. Trains a `PoolHeadRegressor` with stratified 5-fold CV and evaluates on a held-out test set per fold.

```yaml
dg_reg:
  pretrained_path: /path/to/ssl_edge_best.pt  # Auto-set if ssl phase runs first
  regressor_layers: 3
  ft_hdim: 512                    # Hidden dim for regression head

  # Encoder (frozen by default)
  freeze_encoder: True
  freeze_epochs: 1000

  # Architecture (null → auto-loaded from SSL config)
  num_layers: null
  hidden_dim_power: null
  hgt_heads: 4
  dropout: 0.3
  use_checkpoint: True
  message_style: null

  # Pooling
  pool_mode: cross_attn           # cross_attn | attn | mean | max | attn+mean | attn+max | attn+mean+max
  use_jk: True
  jk_mode: mean                   # sum | mean | max | concat
  cross_attn_queries: 2
  cross_attn_heads: 4

  # Loss
  loss_fun: logcosh               # mse | logcosh | l1 | huber
  gamma: 1.0
  delta: 1.0

  # Training
  batch_size: 30
  n_epochs: 1000
  lr: 0.0005
  weight_decay: 0.005
  clip_max_norm: 5.0
  patience: 50
  scheduler_t0: 10
  scheduler_t_mult: 2
  scheduler_eta_min: 1e-6

  save_dir: /path/to/output/
```

### `mutation`

ΔΔG mutation effect prediction. Trains a `PoolHeadDDGRegressor` that takes paired mutant and wild-type graphs and predicts ΔΔG.

```yaml
mutation:
  pretrained_path: /path/to/ssl_edge_best.pt
  regressor_layers: 3
  ft_hdim: 512

  freeze_encoder: True
  freeze_epochs: 1000

  num_layers: null
  hidden_dim_power: null
  hgt_heads: 4
  dropout: 0.3
  use_checkpoint: True
  message_style: null

  pool_mode: cross_attn
  use_jk: True
  jk_mode: mean
  cross_attn_queries: 4
  cross_attn_heads: 2

  ddg_input_mode: concat_diff     # concat_diff | diff

  batch_size: 40
  n_epochs: 1000
  lr: 0.0001
  weight_decay: 0.005
  clip_max_norm: 5.0
  patience: 50
  scheduler_t0: 10
  scheduler_t_mult: 2
  scheduler_eta_min: 1e-6

  save_dir: /path/to/output/
```

`ddg_input_mode` determines how mutant and wild-type complex-level representations are combined before the regression head:
- `concat_diff`: `[h_mut ‖ h_wt ‖ h_mut − h_wt]`
- `diff`: `h_mut − h_wt`

### `disc_binder`

Binary classification (binder vs non-binder). Trains a `PoolHeadClassifier`.

```yaml
disc_binder:
  pretrained_path: /path/to/ssl_edge_best.pt
  classifier_layers: 2
  ft_hdim: 128

  freeze_encoder: True
  freeze_epochs: 1000

  num_layers: null
  hidden_dim_power: null
  hgt_heads: 4
  dropout: 0.6
  use_checkpoint: True
  message_style: null

  pool_mode: cross_attn
  use_jk: True
  jk_mode: mean
  cross_attn_queries: 2
  cross_attn_heads: 4

  label_smoothing: 0.1

  batch_size: 90
  n_epochs: 1000
  lr: 0.0002
  weight_decay: 0.02
  clip_max_norm: 5.0
  patience: 20
  scheduler_t0: 10
  scheduler_t_mult: 2
  scheduler_eta_min: 1e-6

  save_dir: /path/to/output/
```

## Pooling & Aggregation Options

All downstream tasks share the same pooling configuration keys:

| Key | Description |
|-----|-------------|
| `pool_mode` | How node embeddings are reduced to a graph-level vector. `cross_attn` uses a `CrossAttentionAggregator` with learnable queries and cross-chain interaction; other options are standard attention/mean/max or combinations. |
| `use_jk` / `jk_mode` | **Jumping Knowledge (JK-Net)**: when enabled, intermediate HGT layer outputs are aggregated (sum, mean, max, or concat) before pooling, giving each layer a voice in the final representation. |
| `cross_attn_queries` | Number of learnable query vectors per chain (only used when `pool_mode: cross_attn`). |
| `cross_attn_heads` | Number of attention heads in the cross-attention aggregator. |

When `use_jk: True`, the output directory gets a suffix like `_jkmean`, `_jksum`, etc. When `pool_mode: cross_attn`, the suffix `_cattn` is appended.

## Output Structure

### SSL output

```
<save_dir>/
└── 2layers_9hdim_esm/
    ├── ssl_edge_best.pt            # Best encoder checkpoint
    ├── ssl_edge_config.json        # Saved model architecture for downstream loading
    └── ssl_edge_training_log.json  # Per-epoch training metrics
```

### Downstream output (example: dg_reg with JK-mean + cross-attn)

```
<save_dir>/
└── 2layers_9hdim_esm_jkmean_cattn/
    ├── dg_reg_jkmean_fold1_best.pt   # Best model for fold 1
    ├── dg_reg_jkmean_fold2_best.pt
    ├── ...
    ├── dg_reg_jkmean_fold5_best.pt
    ├── test_results_fold1.json       # Held-out test metrics per fold
    └── dg_reg_cv_log.json            # CV training logs
```

Mutation and disc_binder follow the same pattern with prefixes `mut_ddg_` and `disc_binder_` respectively.

## Batch Training

### SSL grid search

`run_multiple_ssl.sh` iterates over `(num_layers, hidden_dim_power)` pairs, generates per-combo YAML configs, and launches `train_unified.py` for each:

```bash
bash run_multiple_ssl.sh
```

### Downstream grid search

`run_multiple_finetune.sh` finds existing SSL checkpoints and runs downstream tasks for each:

```bash
bash run_multiple_finetune.sh
```

Both scripts use `sed` to generate per-config YAML files under `model_configs/for_multirun/`.

## Key Design Decisions

### Precomputed embeddings (frozen encoder)

Rather than fine-tuning the encoder end-to-end, all downstream tasks:
1. Load the SSL encoder once
2. Precompute all node embeddings into `graph.x`
3. Delete the encoder from GPU memory
4. Train only the Pool + Head (pooling layer + task-specific MLP)

This makes downstream training significantly faster and more memory-efficient. The encoder architecture is auto-loaded from the `ssl_edge_config.json` saved alongside the SSL checkpoint — `num_layers`, `hidden_dim_power`, `hgt_heads`, `message_style`, and `embedding_type` from the task config can be left as `null`.

### Architecture auto-inheritance

When `pretrained_path` is set, each finetune module calls `load_ssl_config()` to read the encoder's saved config. Fields like `num_layers`, `hidden_dim_power`, and `embedding_type` are automatically filled in, ensuring the encoder instantiation matches the checkpoint.

### CosineAnnealingWarmRestarts scheduler

All tasks use `CosineAnnealingWarmRestarts` with configurable `T_0`, `T_mult`, and `eta_min`. Combined with AdamW and dynamic gradient clipping (`clip_max_norm`), this provides stable training across different model sizes.

## Troubleshooting

### "No samples loaded"

Check that `pdb_root` is correct and contains the expected data directories (e.g., `pos_sthg_8A` for SSL, `all_sthg_8A` for finetune).

### CUDA out of memory

1. Reduce `batch_size`
2. Enable `use_checkpoint: True` (gradient checkpointing in the encoder)
3. Reduce `hidden_dim_power`

### SSL config not found

If running downstream tasks only, ensure:
1. `pretrained_path` points to a valid checkpoint file
2. The same directory contains `ssl_edge_config.json`

### Architecture mismatch

Leave `num_layers`, `hidden_dim_power`, and `message_style` as `null` in the downstream config — they will be auto-loaded from the SSL config.
