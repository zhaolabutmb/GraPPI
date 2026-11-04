# Train on the full experimental set, test on the Anthropic set

Written for: whoever runs or reviews this experiment later (including future you).

This folder trains the GraPPI binder classifier on **all 3,669 labeled
experimental (Zenodo) designs** and evaluates it on the **Anthropic
claude-protein-binder-design** set. It is a sibling study to
[`exp_target_retraining/`](../exp_target_retraining/), which instead ran
15-fold leave-one-target-out within the experimental set; see that folder's
[chat history summary](../exp_target_retraining/CHAT_HISTORY_SUMMARY.md) for
the background.

Large data, caches, checkpoints and outputs live in the sibling `GraPPI_data`
directory, never in this repository.

## What is being compared

| Row | Encoder | Head | Trained here? |
|---|---|---|---|
| **GraPPI** | frozen `2layers_10hdim_esm` (2 HGT layers, hidden 2^10) | cross-attn pooling + 2-layer MLP | yes, on 3,669 exp designs |
| **ESM** | none — raw 1,280-dim ESM-2 node features | **same** head config | yes, on 3,669 exp designs |
| **Original head** | frozen `2layers_10hdim_esm` | published `2layers_10hdim_esm_jkmean` classifier | no — inference only |

The ESM row isolates what the graph encoder contributes: identical head
architecture, identical loss, identical training pool, identical test set, only
the input representation differs.

## Data

**Train — `../GraPPI_data/exp_binder_data`:** 3,669 labeled designs
(394 binders / 3,275 non-binders) across 15 targets. No held-out target and no
internal split: every labeled design is a training sample. The graph cache
built for the LOTO study is reused unchanged.

**Test — `../GraPPI_data/anthr_binder_data`:** 1,320 labeled designs
(354 binders / 966 non-binders) across 15 targets, from two design models
(Mythos Preview, Opus 4.8) and several campaigns. **1,189** have a released
`designed.cif` and therefore a prepared PDB; the remaining **131** are skipped,
as agreed — they are reported in the cache manifest and in each run's
`coverage` field, and are never scored as negatives.

Both datasets use binder chain A and merged target chain B, with GraPPI
receiving `B,A`, and both caches are built at an 8 Å interface cutoff.

### Target names common to both datasets

`EGFR` and `TrkA` appear verbatim in both label tables, and `IL-7Ra`/`IL7Ra`
and `PD-L1`/`Pdl1` are the same proteins under different spellings. The
training and test sets are therefore **not** disjoint in target identity, even
though the designs themselves are different. `train_exp_anthr.py` prints the
overlapping target names and records them in the run summary; treat the
affected targets accordingly when reading per-target results.

## Protocol

- **Loss:** weighted BCE, `BCEWithLogitsLoss(pos_weight = N_neg / N_pos)` with
  the weight computed on the **training set only**: 3275/394 = 8.3122. Hard 0/1
  labels, no label smoothing.
- **Encoder:** `../GraPPI_data/trained_data_ssl_finetune/2layers_10hdim_esm`,
  frozen. Embeddings are precomputed once per dataset; only the pooling and MLP
  head are trained.
- **Head:** mirrors `2layers_10hdim_esm_jkmean` — cross-attention pooling
  (2 queries, 4 heads), JK(mean), 2 classifier layers, hidden 256, dropout 0.4.
  Each run initializes a fresh head; it does **not** warm-start from the
  published MLP checkpoint.
- **Selection:** early stopping and best-epoch selection on the Anthropic set's
  average precision (`select_metric: auprc`, `select_on: test`).
- **Metrics:** AUROC and AP (scikit-learn `average_precision_score`), pooled
  and stratified by target, design model and campaign, plus median/IQR across
  groups. Groups with a single class (the Anthropic `MBP` target has zero
  positives) report NaN rather than a fabricated 0.5, and are excluded from the
  medians.

> **Limitation — read before quoting any number.**
> The Anthropic labels are used to pick the stopping epoch, so the Anthropic
> metrics for the GraPPI and ESM rows are **optimistic and not blind-test
> estimates**. This is the same limitation the LOTO study carries, and it is by
> design here. The **Original head** row is unaffected: it involves no training
> and its fold was chosen by the head's original training `fold_val_score`, so
> that row is a genuine held-out evaluation.
> For a blind estimate later, set `train.select_on: train` in the config, which
> stops on the training loss and never reads the test labels.

## Files

| File | Purpose |
|---|---|
| [`config_exp_train_anthr_test.yaml`](./config_exp_train_anthr_test.yaml) | All settings: paths, encoder, head, optimizer, selection |
| [`cache_graphs_anthr.py`](./cache_graphs_anthr.py) | Builds the Anthropic graph + ESM cache (handles that set's label schema) |
| [`train_exp_anthr.py`](./train_exp_anthr.py) | Trains on the full exp set, evaluates on Anthropic (`--mode grappi\|esm`) |
| [`score_original_head.py`](./score_original_head.py) | Scores Anthropic with the published head through the same cache and metrics |
| [`anthr_eval.py`](./anthr_eval.py) | Shared metrics, stratification and output writing |
| [`compare_results.py`](./compare_results.py) | Merges the three score sets into one comparison table (CPU only) |
| [`anthr_results_summary.ipynb`](./anthr_results_summary.ipynb) | Notebook: tables, per-target/model/campaign breakdowns, PR curves, training curves |
| [`tests/`](./tests/) | Offline unit tests for the pure logic |

## Running

All commands are run from the repository root with the PPI environment.

### 1. Build the Anthropic graph cache (once, ~GPU for the ESM pass)

The experimental cache already exists and is reused; only this one is new.

```bash
/home/zhisong/.conda/envs/PPI/bin/python exp_train_anthr_test/cache_graphs_anthr.py \
    --workers 20 --cuda-id 0
```

Writes `../GraPPI_data/anthr_binder_data/grappi_graphs_esm_8A/graphs_esm_8A.pkl`
plus a `cache_failures.csv` listing the 131 structureless designs and any build
failures. Expect ~1,189 cached graphs. Add `--limit 20` for a quick smoke test
first — but note that a limited run overwrites the pickle, so rebuild in full
before training.

### 2. Train and evaluate (the two rows can share a machine on separate GPUs)

```bash
# GraPPI
/home/zhisong/.conda/envs/PPI/bin/python exp_train_anthr_test/train_exp_anthr.py \
    --config exp_train_anthr_test/config_exp_train_anthr_test.yaml \
    --mode grappi --cuda-id 0

# ESM baseline, in another terminal
/home/zhisong/.conda/envs/PPI/bin/python exp_train_anthr_test/train_exp_anthr.py \
    --config exp_train_anthr_test/config_exp_train_anthr_test.yaml \
    --mode esm --cuda-id 1
```

`--cuda-id` is an index into the visible CUDA devices. `--seed` overrides the
config seed if you want repeat runs.

### 3. Score the published head

```bash
/home/zhisong/.conda/envs/PPI/bin/python exp_train_anthr_test/score_original_head.py \
    --config exp_train_anthr_test/config_exp_train_anthr_test.yaml --cuda-id 0
```

The fold is picked by maximum original training `fold_val_score`; pin a
specific one with `--head-checkpoint PATH`.

### 4. Build the comparison table

```bash
/home/zhisong/.conda/envs/PPI/bin/python exp_train_anthr_test/compare_results.py
```

This errors out if the three score sets do not cover exactly the same designs
with the same labels, rather than quietly comparing different cohorts.

### 5. Inspect the results

Open [`anthr_results_summary.ipynb`](./anthr_results_summary.ipynb), select the
**PPI** kernel, and Run All. It reads the completed exports and shows run
availability and provenance, the headline comparison, per-target /
per-design-model / per-campaign tables, per-target AP and AUROC plots,
precision–recall curves (`TARGET_TO_PLOT` picks the target), training curves
with the selected epoch marked, and score distributions by label. Missing runs
are skipped rather than raising, so it is usable while a run is still training.
It trains nothing and modifies no results; its last cell re-exports the same
comparison CSVs `compare_results.py` writes.

### Tests

```bash
/home/zhisong/.conda/envs/PPI/bin/python -m unittest discover -s exp_train_anthr_test/tests
```

## Outputs

GraPPI →
`../GraPPI_data/trained_data_ssl_finetune/2layers_10hdim_esm_exp_train_anthr_test/`
(the published-head scores land here too);
ESM →
`../GraPPI_data/trained_data_ssl_finetune/esm1280_exp_train_anthr_test/`.

Per run, with prefix `anthr_test`, `anthr_test_esm` or
`anthr_test_original_head`:

- `*_raw_scores.csv` — per-design label, logit, probability, target, design
  model, campaign. Any other metric can be recomputed from this without
  retraining.
- `*_summary.json` — protocol, config, checkpoint, best epoch, coverage,
  pooled and stratified metrics, the stated limitation, and (for the published
  head) checkpoint SHA-256s and the git commit.
- `*_per_target.csv`, `*_per_design_model.csv`, `*_per_campaign.csv`
- `*_epoch_log.csv` — per-epoch training loss and test AUROC/AP (trained rows only)
- `*_best.pt` — best-epoch head weights (trained rows only)

`compare_results.py` writes `anthr_comparison_summary.csv` and
`anthr_comparison_per_*.csv` into
`../GraPPI_data/trained_data_ssl_finetune/anthr_comparison/`.

## Reference points

An earlier run of the published head over this dataset, through the
`anthr_binder_data/scripts/validate_anthropic_binders.py` CLI pipeline, scored
1,186 designs at AUROC **0.4711** and AP **0.2440** against a scored-sample
prevalence of **0.2715**. Step 3 above reproduces that evaluation inside this
pipeline, on the shared cache; small differences in the scored count are
expected if a structure fails graph construction here but not there, so compare
the `coverage` fields before comparing the metrics.
