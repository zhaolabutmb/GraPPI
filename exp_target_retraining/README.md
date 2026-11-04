# Target-aware retraining of the GraPPI binder classifier on the experimental set

**Status: implemented.** The results notebook reports whether completed training
exports are available. This document records the agreed strategy, the facts
verified against the current code/data, and the commands to run it.

Scope: retrain **only the classifier head** (cross-attention pool + MLP) on the Zenodo
experimental binder set, using weighted BCE and a target-aware split. Compare the frozen
GraPPI encoder with a cached-ESM baseline; published checkpoints are not modified. The
Anthropic set is **on hold**.

---

## 1. Confirmed data facts (verified)

Source: `../GraPPI_data/exp_binder_data/final_dataset.csv` (3,676 rows).

- **Labeled rows: 3,669** — label column `binder` (`True`/`False`). The other 7 rows have
  empty `binder` **and** empty `target_id` (the unlabeled exclusions); they are ignored.
- **Targets: exactly 15** distinct `target_id` values among labeled rows — confirms your
  count. (A 16th apparent value is just the empty id on the 7 unlabeled rows.)
- **Class balance: 394 positive / 3,275 negative.**
- **Labels for training: `1` = binder, `0` = non-binder** (consistent with your previous
  classifier setup; hard labels, no label smoothing).

### Per-target counts (labeled subset)

| target_id | pos | neg | total | % of labeled |
|---|---:|---:|---:|---:|
| FGFR2 | 193 | 1930 | **2123** | **57.9%** |
| EGFR | 28 | 406 | 434 | 11.8% |
| IL7Ra | 38 | 133 | 171 | 4.7% |
| TrkA | 9 | 119 | 128 | 3.5% |
| InsulinR | 20 | 97 | 117 | 3.2% |
| VirB8 | 9 | 90 | 99 | 2.7% |
| SARS_CoV2_RBD | 9 | 90 | 99 | 2.7% |
| pMHC_SILSY1 | 2 | 94 | 96 | 2.6% |
| Mdm2 | 55 | 41 | 96 | 2.6% |
| Pdl1 | 12 | 83 | 95 | 2.6% |
| IL2Ra | 6 | 60 | 66 | 1.8% |
| sntx | 7 | 42 | 49 | 1.3% |
| pMHC_NY1 | 1 | 40 | 41 | 1.1% |
| LTK | 3 | 30 | 33 | 0.9% |
| IL10Ra | 2 | 20 | 22 | 0.6% |
| **Total** | **394** | **3275** | **3669** | 100% |

---

## 2. Models and checkpoints (verified config)

**Encoder (frozen)** — `../GraPPI_data/trained_data_ssl_finetune/2layers_10hdim_esm/ssl_edge_best.pt`
(`ssl_edge_config.json`): embedding_type `esm`, num_layers 2, hidden_dim_power 10
(hidden_dim 1024), hgt_heads 4, message_style `gated_src`, dropout 0.2.

**Classifier head** — mirror
`../GraPPI_data/trained_data_ssl_finetune/2layers_10hdim_esm_jkmean/`
(`disc_binder_jkmean_5fold_summary.json` + `model_configs/config_disc_binder.yaml`):
pool_mode `cross_attn`, cross_attn_queries 2, cross_attn_heads 4, use_jk `True`,
jk_mode `mean`, classifier_layers 2, ft_hdim 256, dropout 0.4, embedding_type `esm`.

Graph construction matches inference: `dist = 8`, chain order `B,A` (target B, binder A),
`min_nodes = 5`. **Encoder stays fully frozen; embeddings are precomputed once and reused.**

---

## 3. Target-size similarity analysis (the decision gate)

Per decision #1, the CV scheme is chosen by whether per-target sample counts are similar.
Reproduce with `python exp_target_retraining/analyze_target_sizes.py`
(reads `final_dataset.csv`, no GPU, no training).

Decision rule (targets are "similar in size" only if **all** hold):
`max/min ratio ≤ 3`, `coefficient of variation ≤ 0.5`, and χ²-vs-uniform `p > 0.05`.

**Result on the 15 labeled targets:**

| Statistic | Value | Similar threshold | Pass? |
|---|---:|---|:--:|
| max / min ratio | 96.5 (2123 vs 22) | ≤ 3 | ❌ |
| coefficient of variation | 2.09 | ≤ 0.5 | ❌ |
| Gini | 0.663 | → 0 | ❌ |
| χ² vs uniform (dof 14) | 16003.4, p ≈ 0 | p > 0.05 | ❌ |

The imbalance persists after holding out any single target as the test target
(remaining-14 max/min ratio ≥ 19.7; ≥ 64 for all holdouts except FGFR2).

**➡ Superseded by the LOTO decision (§4).** The size analysis originally chose the
sample-level 5-fold branch, but the protocol was later changed to **leave-one-target-out
CV** (per-target evaluation, matching the meta-analysis paper's per-target panels). The
analysis is kept here only as context on how unequal the targets are; it no longer drives
the split. The target-grouped 7-fold / sample-level 5-fold schemes are **not** used.

---

## 4. Final split design (active): leave-one-target-out (LOTO)

**One fold per target — 15 folds.** In fold *k*, target *k* is the **validation set** and
all samples from the other 14 targets are the **training set**. Every target is validated
exactly once; there is **no separate pre-held-out test target** (the old design was
replaced). Targets are iterated in sorted order for determinism.

**Per fold:**
- Train the pool+head with weighted BCE (§5) on the 14 training targets.
- Early-stop and select the best checkpoint on the **held-out target's AUPRC** (§6). *Note:*
  because the held-out target is the only validation signal, its reported AUPRC/AUROC is a
  best-epoch-on-that-target number (chosen by you); it is a per-target CV score, not a
  blind-test estimate.
- Record the held-out target's **AUROC and AUPRC**, plus the **raw per-sample logit and
  probability** for every design in that target.

**Aggregate across the 15 targets:**
- **Median + IQR** of AUROC and AUPRC (equal weight per target — matches the paper's panels
  D/E, and prevents FGFR2's 58% mass from dominating).
- **Pooled** AUROC/AUPRC computed by concatenating all 15 held-out score sets into one
  curve (robust for the 1–3 positive targets such as pMHC_NY1, IL10Ra, pMHC_SILSY1, LTK).
- Mean ± std are also reported.

**pos_weight** stays the fixed global `3275/394 = 8.3122` across all folds (decision #5),
even though each fold trains on only 14 targets.

---

## 5. Loss: weighted BCE (what changes, exactly)

Your formula

```
L_WBCE = -(1/N) Σ [ w_+ · y_i · log(p_i) + (1 - y_i) · log(1 - p_i) ]
```

is **exactly** `torch.nn.BCEWithLogitsLoss(pos_weight=w_+)` (mean reduction). So:

- **No training-loop edits.** `train_epoch_classifier` and `validate_classifier`
  (`utils/Training_modules/finetune_training.py`) already take `criterion` and call
  `criterion(logits, y_cls)`; we just pass a WBCE criterion.
- **`w_+` is the fixed global value** `N_neg / N_pos = 3275 / 394 = 8.3122`, constant across
  all folds (decision #5):
  `criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([8.3122], device=device))`.
- **Labels:** `y = 1.0` for binder, `0.0` for non-binder. The existing loop derives the
  class as `(y > 0)`, so `1→positive`, `0→negative` — matches your prior setup.
- **`label_smoothing = 0`** (hard labels) so the loss equals your formula exactly.
- **Do NOT edit** the two `nn.BCEWithLogitsLoss()` lines at
  `utils/finetune_modules/finetune_disc_binder_module.py:337` and `:917` — those belong to
  the published curated-DB pipeline and stay untouched.

---

## 6. Metrics (verified gap to close)

> **"AUPRO" → AUROC.** Your request said "collect the AUPRO and AUPRC". In binary
> classification there is no standard metric called AUPRO, and the natural pair with
> AUPRC is **AUROC** (area under the ROC curve), so AUPRO is read as **AUROC** here.
> This is non-blocking either way: the driver saves **raw per-sample scores**
> (`label, logit, prob`), so if you meant a different metric we can compute it later
> from the CSV without retraining.

- Base `compute_classification_metrics` (`utils/Training_modules/common_utils.py:152`)
  returns `accuracy, precision, recall, f1, auc` — **no AUPRC**. AUPRC must be added via
  `sklearn.metrics.average_precision_score`.
- The repo's AUPRC lives inside `_validate_classifier_seen_unseen`, which also applies SSL
  **seen/unseen** logic that **does not apply** here (these external designs were never in
  GraPPI's SSL pre-training). The new driver will **not** reuse that validator.
- New driver computes, for each held-out target:
  - `AUROC` via `roc_auc_score`, `AUPRC` via `average_precision_score`
    (both guarded for the single-class edge case).
- It also saves **raw per-sample logits + probabilities** (`exp_target_loto_raw_scores.csv`)
  so any other metric (F1, precision@recall, max-F1 thresholds, pooled curves) can be
  recomputed later without retraining.
- **Model selection / early stopping: the held-out target's AUPRC** (your choice), `mode='max'`.

---

## 7. What is new vs. reused

**Reused by import (unchanged):** `load_pretrained_encoder`, `precompute_embeddings`,
`get_pool_input_dim`, `create_poolhead_model`, `FineTuneEarlyStopping`,
`train_epoch_classifier`, `validate_classifier`, `compute_classification_metrics`
(+ `average_precision_score`).

**New code required:**
1. **Experimental data loader.** Inputs exist **only as PDBs** under
   `../GraPPI_data/exp_binder_data/grappi_inputs_classification/pdbs/`. Build GraPPI graphs
   once (ESM, dist=8, chains `B,A`), attach `y ∈ {0,1}` and a `target_id`, and cache to
   disk. Existing `load_classifier_data_cv` is curated-DB-specific and is not reused.
2. **Split driver** implementing §3–§4: seeded pre-held-out test target, then
   `StratifiedKFold(5)` on the pooled 14-target samples.
3. **Per-fold WBCE** (fixed `pos_weight=8.3122`) and plain AUROC/AUPRC reporting, selection
   on val AUPRC.
4. **Config + logging + output directory** for checkpoints, split JSON, per-fold and
   aggregate metrics, and cached graphs.

No existing training behavior changes; the work is additive.

---

## 8. File / artifact layout

- **Code / config / tests → this folder** `GraPPI/exp_target_retraining/`
  - `analyze_target_sizes.py` — per-target size similarity analysis (the §3 decision gate).
  - `cache_graphs.py` — build + cache graph **and** ESM features for the labeled
    experimental complexes (CPU multiprocess graph build, single-GPU ESM pass).
  - `train_exp_target_wbce.py` — target-aware WBCE training driver (split, folds,
    frozen-encoder precompute, per-fold training, AUROC/AUPRC reporting).
  - `config_exp_target_wbce.yaml` — all knobs for the driver.
  - `loto_results_summary.ipynb` and `loto_results.py` — interactive comparison
    of finished GraPPI and ESM runs, including per-target AP, AUROC, prevalence,
    lift, median/IQR, and raw-score precision–recall plots.
  - `tests/test_exp_target_split.py` — unit tests for the split/metric/label logic.
- **Outputs → `GraPPI_data`:**
  - Cached graphs: `../GraPPI_data/exp_binder_data/grappi_graphs_esm_8A/graphs_esm_8A.pkl`.
    The combined pickle is written only after all CPU graph construction and the
    ESM pass finish; the folder can remain empty while the job is running.
    Graph caches do **not** go under `trained_data_ssl_finetune`.
  - Training: `../GraPPI_data/trained_data_ssl_finetune/2layers_10hdim_esm_exp_target_loto_wbce/`
    — per-target checkpoints `disc_binder_exp_target_loto_<TARGET>_best.pt`,
    `exp_target_loto_summary.json` (per-target + median/IQR + pooled metrics),
    `exp_target_loto_per_target.csv`, `exp_target_loto_raw_scores.csv` (raw logits +
    probabilities for every held-out design), `exp_target_loto_splits.json`.
  - ESM baseline: `../GraPPI_data/trained_data_ssl_finetune/esm1280_exp_target_loto_wbce/`
    — the same metric/score filenames, with separate ESM-head checkpoints.

### ESM-only comparison

Use `--mode esm` with the same config and graph cache. This bypasses both the
GraPPI encoder and ESM inference. It follows the existing baseline feature-slicing
pattern to retain only `x[:, :1280]` (or `:480` for an ESM-480 cache), excluding the
five structural columns. Graph edges and coordinates are not used by this mode.
The residue set is the same as in the cache; this is not a separate full-sequence
baseline. The trainable cross-attention pooling and classifier remain the same;
the input projection changes from GraPPI's embedding width to ESM's width.
The existing `_slice_esm_features` helper is not called: its `nt in graph`
guard does not detect node stores in the installed PyG version. The local
adapter validates both node stores and then slices them explicitly.

Both modes use the same sorted 15 target folds, seed, WBCE, validation-AUPRC
selection, and per-target / macro / pooled metrics and raw-score CSV schema.
The baseline has its own output directory, so it does not overwrite GraPPI
results. `--output-dir PATH` can override either mode's output location.

### Cache startup and progress

CPU graph construction uses a spawn-based process pool, avoiding inherited
PyTorch/native-library thread state from forked workers. Each worker is limited
to one numeric-library thread. The progress bar advances for each completed
structure, not ordered batches of four. Worker imports still have a startup
cost. Workers serialize graph results as ordinary bytes before transferring them,
avoiding a persistent shared-memory file descriptor per tensor in the parent
process. This prevents descriptor exhaustion on typical 1,024-file limits.
Changing the script does not change an already-running process: stop the
old job with Ctrl+C in its terminal before restarting it with the command below.

### How to run (from the repo root, PPI env)

```bash
# 0. (optional) GPU-free unit tests
/home/zhisong/.conda/envs/PPI/bin/python -m unittest discover \
    -s exp_target_retraining/tests -v

# 1. Build the graph+ESM cache once (~3669 complexes). CPU graph build + 1 GPU ESM pass.
#    Not training. Writes to GraPPI_data/exp_binder_data/grappi_graphs_esm_8A/.
/home/zhisong/.conda/envs/PPI/bin/python exp_target_retraining/cache_graphs.py \
    --workers 20 --cuda-id 0

# 2. In one terminal: GraPPI classifier on GPU 0.
/home/zhisong/.conda/envs/PPI/bin/python exp_target_retraining/train_exp_target_wbce.py \
    --config exp_target_retraining/config_exp_target_wbce.yaml --mode grappi --cuda-id 0

# 3. In another terminal: ESM baseline on GPU 1, using the SAME cache.
/home/zhisong/.conda/envs/PPI/bin/python exp_target_retraining/train_exp_target_wbce.py \
    --config exp_target_retraining/config_exp_target_wbce.yaml --mode esm --cuda-id 1
```

The two training commands can run concurrently once the cache is complete.
`--cuda-id` overrides `system.cuda_id` for that command only; if omitted, the
config value is used, with CPU fallback when CUDA is unavailable. An explicitly
requested GPU must be available. IDs refer to CUDA-visible devices and may be
renumbered by `CUDA_VISIBLE_DEVICES`. Each process loads the cache independently.

### Results notebook

Open `loto_results_summary.ipynb`, select the PPI Python kernel, and run all
cells. The notebook reads the final JSON/CSV files from both output directories;
it shows which runs are still awaiting exports and works with one completed run.
It also reads `../GraPPI_data/exp_binder_data/final_dataset.csv` and recomputes
the two [Figure 3B AF3 features](https://www.biorxiv.org/content/10.1101/2025.08.14.670059v2.full)
on the same evaluated samples. These are deposited-feature results, not
published figure values: `AF3_ipSAE` uses `af3_ipSAE_min`, and
`AF3_interaction` uses the negative of
`af3_ipSAE_min * af3_rosetta_interface_dG_SASA_ratio` so that higher ranks are
more binder-like. AF3 feature scores are not probabilities.

The notebook recomputes AP and AUROC from the saved model probabilities or AF3
ranking scores, checks counts and sample/label consistency, and refuses a
direct comparison when cohorts or model training settings differ. Missing or
non-finite AF3 values for evaluated samples raise an error. If the source CSV
is missing, model results remain available and the notebook reports the
missing file. If no model is ready, available AF3 results use the labeled
reference dataset. `TARGET_TO_PLOT` selects the PR curve. Edit the paths in the
setup cell if you used `--output-dir` or moved the data.

The target used for each reported score also selected that fold's best epoch, so
these are potentially optimistic validation scores, not blind-test estimates.
The prevalence reference is not exactly the finite-sample expected random AP.
Pooled predictions come from different fold models and may not be calibrated
comparably across targets. AF3 features use re-predicted structures while GraPPI
used input structures. The paper states 3,766 designs and 436 binders, whereas
the checksum-verified Zenodo CSV has 3,676 rows, including 3,669 labeled
designs and 394 binders; the discrepancy is unexplained. The notebook also
identifies two published Figure 4 results separately; these use different
features or a different model and the paper's stated cohort, and the paper's
logistic model used nested target-group cross-validation.

---

## 9. Locked decisions (your answers)

1. CV scheme: **leave-one-target-out (LOTO)** — 15 folds, each target validated once; no
   separate held-out test target (supersedes the earlier sample-level 5-fold design).
2. Resolved by #1.
3. Labels **0 = non-binder, 1 = binder**; label smoothing 0.
4. Early-stop / select on the **held-out target's AUPRC**.
5. `w_+` = **global 3275/394 = 8.3122**, fixed across folds.
6. Encoder **fully frozen**, precomputed embeddings reused.
7. Output dir `2layers_10hdim_esm_exp_target_loto_wbce`.
8. **Anthropic set on hold.**
9. Report **per-target AUROC/AUPRC + median/IQR across targets + pooled metrics**, and save
   **raw per-sample scores** for arbitrary later metrics.

### Validation

Tests cover target folds, labels, metrics, exact cached-ESM feature extraction,
an end-to-end synthetic CPU baseline run (including score/checkpoint outputs and
encoder bypass), spawn-worker graph construction, and graph transfers under a
reduced file descriptor limit. The results notebook is tested with incomplete,
single-model, and paired runs, including synthetic exports from the actual
training driver and checks for inconsistent labels, metrics, settings, and
AF3 feature cohorts.
These checks do not run the full experimental cache or the 15-fold training job.
