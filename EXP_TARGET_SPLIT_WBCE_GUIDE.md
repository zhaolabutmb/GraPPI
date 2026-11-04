# Guidance: Target-split retraining of the GraPPI binder classifier on the experimental (Zenodo 3,766) set, with weighted BCE

**Status:** design/guidance only. This document records the agreed plan so the
work can be implemented later. **No code was run and no code was changed** when
this file was written. Items marked **[CONFIRM]** need a decision from the user
before implementation.

---

## 1. Objective

Retrain / fine-tune the GraPPI discriminative-binder classifier head on the
experimental binder set (Zenodo record 15722219, "3,766 experimentally
characterised binders"), using a **split by protein target** so that the
validation and test targets are never seen in training. The GraPPI encoder stays
frozen; only the pooling + classifier head is trained. The training loss is
switched from plain BCE to **weighted BCE** (positive-class weighting) to handle
the strong class imbalance.

Deliverables of the eventual run (per fold): validation **AUROC** and **AUPRC**,
and test **AUROC** and **AUPRC**.

---

## 2. Dataset facts (verified from `final_dataset.csv` / per-target results)

- Labeled & scored complexes: **3,669** (394 binders / 3,275 non-binders).
  7 rows have an empty `binder` label and are excluded (consistent with the
  existing validation pipeline).
- Label source: the `binder` column (`True` → positive/binder = 1,
  `False` → negative/non-binder = 0).
- Input convention (already used by the inference pipeline): binder = chain **A**,
  merged target = chain **B**, GraPPI chain order `B,A`.
- **Number of protein targets: 15** (confirmed — matches the user's count), via
  the `target_id` column.

### 2.1 Per-target composition

| target_id | n | n_pos | n_neg | % of all data |
|---|---:|---:|---:|---:|
| FGFR2 | 2,123 | 193 | 1,930 | 57.9% |
| EGFR | 434 | 28 | 406 | 11.8% |
| IL7Ra | 171 | 38 | 133 | 4.7% |
| TrkA | 128 | 9 | 119 | 3.5% |
| InsulinR | 117 | 20 | 97 | 3.2% |
| SARS_CoV2_RBD | 99 | 9 | 90 | 2.7% |
| VirB8 | 99 | 9 | 90 | 2.7% |
| Mdm2 | 96 | 55 | 41 | 2.6% |
| pMHC_SILSY1 | 96 | 2 | 94 | 2.6% |
| Pdl1 | 95 | 12 | 83 | 2.6% |
| IL2Ra | 66 | 6 | 60 | 1.8% |
| sntx | 49 | 7 | 42 | 1.3% |
| pMHC_NY1 | 41 | 1 | 40 | 1.1% |
| LTK | 33 | 3 | 30 | 0.9% |
| IL10Ra | 22 | 2 | 20 | 0.6% |
| **Total** | **3,669** | **394** | **3,275** | 100% |

### 2.2 Imbalance summary (drives the loss design)

- Overall positive rate = 394 / 3,669 = **10.7%**.
- Overall negative:positive ratio = 3,275 / 394 = **8.31**.
  (The user's ≈9.4 estimate used round numbers 3,200/340; the exact overall value
  is 8.31, and the per-fold value will differ — see §5.)

### 2.3 Target-size skew (important — read before choosing the split)

The targets are **very** uneven. **FGFR2 alone is 57.9% of all complexes and 49%
of all positives.** EGFR + FGFR2 together are ~70% of the data. Several targets
have ≤3 positives (pMHC_NY1 has 1, IL10Ra and pMHC_SILSY1 have 2). This has two
direct consequences:

1. Whichever large target lands in **validation** or **test** dominates that
   fold's metric; a single target in the held-out set is a high-variance
   estimate.
2. Some targets contain essentially no positives, so a test/val fold built from
   only those targets can have an undefined or degenerate AUROC/AUPRC.

This is exactly the "validation samples too many" problem the user asked to be
flagged — see §4.3.

---

## 3. Split design (by target)

Split the **15 targets** (not individual complexes) into:

- **12 training targets**
- **2 validation targets**
- **1 test target**

12 + 2 + 1 = 15. The split is **random**, controlled by the **system seed
(`system.seed`, default 42)**. Splitting by target guarantees no target leaks
between train / val / test.

### 3.1 **[CONFIRM] What "5-fold" means here**

The request says *"2 for 5-fold cross validation ... Each fold will also produce
test AUROC and AUPRC."* Two reasonable interpretations exist; please confirm:

- **Interpretation A (assumed default): 5 independent random target partitions.**
  For `fold = 1..5`, draw a fresh random 12/2/1 target partition from a
  seeded RNG (`seed = 42`, then fold-specific sub-seeds, e.g. `42 + fold`).
  Each fold trains a fresh head on its 12 train targets, reports val AUROC/AUPRC
  on its 2 val targets and test AUROC/AUPRC on its 1 test target, then results
  are averaged ± std across the 5 folds. This matches "each fold also produces
  test metrics" because every fold has its own test target. It is a
  *grouped Monte-Carlo CV* at the target level (test/val targets may repeat
  across folds by chance).

- **Interpretation B: one fixed test target, then 5-fold rotation of the rest.**
  Hold out **1** test target once (seeded). From the remaining 14 targets, run a
  grouped 5-fold rotation that yields 12 train / 2 val per fold. Note 14 targets /
  2-per-val = 7 non-overlapping folds, so "5 folds" would **not** cover all 14 val
  targets exhaustively; it would be 5 seeded draws. The single test target is then
  constant across folds (so "each fold produces test metrics" means re-evaluating
  the same test target with each fold's model).

Interpretation A is implemented unless you choose B. Either way the RNG is the
system seed (42).

### 3.2 Determinism

- Use one seeded `numpy.random.Generator` / `random.Random(seed)` to shuffle the
  sorted list of 15 target IDs, then slice 12/2/1. Record the exact target → split
  assignment for every fold in the run's JSON log so the split is reproducible and
  auditable.

---

## 4. Validation-size check ("report if validation samples are too many")

### 4.1 Why a check is needed

The nominal target ratio is train:val = 12:2 = **6:1**. But because targets differ
hugely in size, the realized **sample** ratio can be far from that. Example
extremes:

- If **FGFR2 (2,123)** lands in validation, val alone is larger than the ~1,546
  remaining training samples → realized train:val ≈ **0.7:1** (val bigger than
  train).
- If validation is two tiny targets (e.g. IL10Ra + LTK = 55 samples), realized
  train:val ≈ **65:1**.

### 4.2 What to compute and report per fold

For each fold, after the target assignment, compute and log:

- `n_train_samples`, `n_val_samples`, `n_test_samples`
- `n_pos` / `n_neg` in each split
- realized `train:val` sample ratio and realized `val` fraction
- a **warning flag** when the validation split is disproportionately large.

### 4.3 **[CONFIRM] Warning threshold**

The user referenced a **train:val ≈ 5:1** sample ratio as the yardstick. Proposed
rule (please confirm the exact numbers):

- **Flag** a fold if realized `train:val` sample ratio `< 5:1`
  (equivalently, validation is `> ~16.7%` of train+val samples). Optionally also
  flag if the ratio is extreme in the other direction (e.g. `> 20:1`), or if a
  val/test target has `0` positives (AUPRC/AUROC undefined).
- The flag is **reported, not auto-corrected**, unless you want mitigation.

### 4.4 **[CONFIRM] Optional mitigations** (only if you want the split constrained)

If unconstrained random target splits produce too many degenerate/oversized folds,
options are:

1. **Pin the largest target(s) to training.** e.g. always keep FGFR2 (and maybe
   EGFR) in the 12 training targets; draw val/test only from the smaller targets.
2. **Reject-and-resample:** redraw a fold's partition (with the next sub-seed)
   until the realized train:val ratio is ≥ 5:1 and val/test each contain ≥1
   positive and ≥1 negative. Log every rejected draw.
3. **Leave fully random** and simply report the flags (default if no mitigation is
   requested).

Default assumption: **option 3 (report only)** unless you pick 1 or 2.

---

## 5. Loss function: weighted BCE

### 5.1 Target behaviour

Replace plain BCE with weighted BCE that up-weights the positive term:

```
L_WBCE = -(1/N) Σ_i [ w+ · y_i · log(p_i) + (1 - y_i) · log(1 - p_i) ]
w+ = N_neg / N_pos
```

This is exactly `nn.BCEWithLogitsLoss(pos_weight=w+)` — PyTorch's `pos_weight`
multiplies the positive term of the logit-space BCE, matching the formula above.
No custom loss class is required.

### 5.2 How `w+` is computed — **per training fold, from hard labels**

`w+` must be derived from the **actual** positive/negative counts of **that fold's
12 training targets only** (not the whole dataset, and never from val/test):

```python
w_pos = n_neg_train / max(1, n_pos_train)
criterion_cls = nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor([w_pos], device=device)
)
```

- Overall reference value is `3275/394 = 8.31`; per fold it will vary (e.g. if
  FGFR2's 193 positives are in the test/val target, the training positive count
  drops and `w+` rises). Log the realized `w_pos` for every fold.
- **[CONFIRM]** Optionally cap/clamp `w_pos` (e.g. to ≤ 10–15) to avoid an extreme
  weight when a fold's training positives are scarce. Default: **no clamp** (use
  the exact ratio).

### 5.3 Interaction with label smoothing — **[CONFIRM]**

The current `config_disc_binder.yaml` sets `label_smoothing: 0.1`. The training
code softens hard targets to `y·(1-ε)+0.5·ε` *before* the loss
(`finetune_training.py` and the validator). `pos_weight` still multiplies the
(now softened) positive term, so the two interact. Recommendation: for a clean
weighted-BCE objective matching the formula in §5.1, **set `label_smoothing: 0.0`**
for these runs. Confirm whether to keep 0.1 or set 0.0.

### 5.4 Where the loss is defined (code locations to change)

`nn.BCEWithLogitsLoss()` is instantiated in **two** places in
`utils/finetune_modules/finetune_disc_binder_module.py`:

- CV path: `_run_disc_binder_cv`, around **line 335**.
- Single train/test path: `_run_disc_binder_single_split` (the non-CV path),
  around **line 917**.

The new target-split routine (see §7) should build its own weighted criterion
**inside the per-fold loop** (because `w_pos` is fold-dependent), rather than once
before the loop. The existing `train_epoch_classifier` /
`validate_classifier` already accept a `criterion` argument and need no change —
`pos_weight` lives inside the criterion.

> Note: the *existing* `disc_binder` pipeline (curated-DB `load_classifier_data_cv`
> with `StratifiedKFold` / `fixed_pos`) is a **different** workflow. Do not repurpose
> it in place; add a new target-split path (§7) so the published
> `2layers_10hdim_esm_jkmean` checkpoints and their global/fixed_pos logs are not
> overwritten.

---

## 6. Encoder and classifier configuration (exact values to reuse)

### 6.1 Encoder (frozen) — `2layers_10hdim_esm`

From `.../trained_data_ssl_finetune/2layers_10hdim_esm/ssl_edge_config.json`:

| field | value |
|---|---|
| checkpoint | `ssl_edge_best.pt` |
| `embedding_type` | **esm** |
| `num_layers` | 2 |
| `hidden_dim_power` | 10 → hidden_dim = 1024 |
| `hgt_heads` | 4 |
| `message_style` | gated_src |
| `dropout` (SSL) | 0.2 |

The encoder stays **frozen** (consistent with `freeze_encoder: True`); embeddings
are precomputed once per fold as in `_run_disc_binder_cv`.
Note: `embedding_type: esm` is supported by the classifier CV path (only the
`+esm`/`only_esm` variants are rejected). The experimental complexes must be built
into graphs with ESM node features (see §8).

### 6.2 Classifier head — config from `2layers_10hdim_esm_jkmean`

From `.../2layers_10hdim_esm_jkmean/disc_binder_jkmean_5fold_summary.json`, reuse:

| field | value |
|---|---|
| `pool_mode` | cross_attn |
| `use_jk` | true |
| `jk_mode` | mean |
| `num_layers` | 2 |
| `hidden_dim_power` | 10 |
| `seed` | 42 |
| `n_folds` | 5 |

Other head hyperparameters from `config_disc_binder.yaml` (e.g.
`classifier_layers`, `cross_attn_queries`, `cross_attn_heads`, `ft_hdim`,
`dropout`, `lr`, `weight_decay`, scheduler, `patience`, `n_epochs`,
`batch_size`) are reused unchanged **except** the loss (§5) and, if confirmed,
`label_smoothing` and `cv_split_mode`.

---

## 7. Implementation plan (new code path, no run yet)

1. **New split mode** `cv_split_mode: target_split` (or a dedicated function
   `_run_disc_binder_target_split`) that:
   - loads the experimental pool (graphs + binary labels + `target_id`) instead of
     the curated-DB `load_classifier_data_cv`;
   - builds the 12/2/1 **target-level** partition for each of the 5 folds from the
     system seed (§3), per Interpretation A unless B is chosen;
   - logs the realized sample ratios and the validation-size warning (§4);
   - precomputes frozen-encoder embeddings;
   - trains a fresh head per fold with **weighted BCE** (`w_pos` per fold, §5);
   - evaluates **val** AUROC/AUPRC and **test** AUROC/AUPRC per fold;
   - writes a summary JSON (per-fold + mean ± std for val and test) and the exact
     target assignments.
2. **Outputs** go to a **new** save directory (do **not** reuse
   `2layers_10hdim_esm_jkmean`, which holds the published checkpoints). Suggested:
   a dedicated folder such as
   `.../trained_data_ssl_finetune/2layers_10hdim_esm_exp_target_wbce/`.
3. **[CONFIRM]** Early-stopping / checkpoint-selection metric. The published
   pipeline used `auprc_unseen`, which depends on the SSL seen/unseen split and
   is **not meaningful** for an external experimental set (all experimental
   samples are "unseen"). Proposed default: drive early stopping and best-model
   selection on **validation AUPRC** (appropriate for strong imbalance), while
   still reporting AUROC. Confirm AUPRC vs AUROC as the selection metric.

---

## 8. Data / graph preparation requirements

- The 3,669 experimental complexes must be represented as GraPPI graphs built the
  **same way** as the encoder's training data: distance threshold `dist = 8`,
  `embedding_type = esm` (ESM node features), binder chain A + target chain B.
  These are the same inputs already used for the classification inference run
  (binder chain A, target chain B, order `B,A`), so the existing extracted inputs
  can be reused if they are saved as graph/setting objects; otherwise they must be
  rebuilt with ESM embeddings.
- Each graph must carry: the binary label `y` (1 binder / 0 non-binder) and its
  `target_id` (for grouping). `target_id` comes from `final_dataset.csv`
  / the existing `grappi_inputs_classification/index.csv`.
- **[CONFIRM]** `target_chain_range`: the dataset provides sub-chain ranges within
  target chain B. The current inference pipeline passes the **whole** merged target
  chain B to GraPPI and does **not** crop by `target_chain_range`. Keep that same
  behaviour for training (recommended, for consistency with how the classifier will
  be used) unless you want the target cropped to a specific sub-chain.

---

## 9. Reproducibility and logging

- Single source of randomness: `system.seed = 42` (target shuffling, fold
  sub-seeds, any resampling). Record every realized split.
- Per-fold log should include: target assignment (train/val/test IDs), sample and
  pos/neg counts per split, realized train:val ratio + warning flag, `w_pos` used,
  per-epoch train/val loss and metrics, and final val/test AUROC & AUPRC.
- Summary log: mean ± std of val AUROC, val AUPRC, test AUROC, test AUPRC across
  the 5 folds.

---

## 10. Caveats to keep in mind when interpreting results

- **Target-size skew (FGFR2).** With only 15 targets and one target at ~58% of the
  data, both val and test metrics are high-variance; a single fold's test AUROC can
  swing widely depending on which target is held out. Report per-fold values, not
  just the mean.
- **Tiny-positive targets.** Targets with 1–3 positives can make a held-out
  AUROC/AUPRC undefined or unstable; these should trigger the §4 warning.
- **External-data distribution shift.** The encoder was SSL-pretrained on the
  curated DB; the experimental complexes are AF2/ColabFold models of de-novo
  designs. Frozen-encoder embeddings may not transfer perfectly; this is a modeling
  caveat, not a bug.
- **Weighted BCE changes calibration.** Up-weighting positives raises predicted
  probabilities; threshold-free metrics (AUROC/AUPRC) are unaffected, but any
  fixed-0.5-threshold accuracy/F1 will shift. Prefer AUROC/AUPRC for comparison.

---

## 11. Open questions to confirm before implementation

1. **Fold structure** — Interpretation A (5 random target partitions) vs B (fixed
   test target + 5 rotations)? (§3.1)
2. **Validation-size warning threshold** — flag when train:val < 5:1? Any upper
   bound? (§4.3)
3. **Split constraint / mitigation** — fully random (report only), pin FGFR2 (and
   EGFR?) to train, or reject-and-resample? (§4.4)
4. **`label_smoothing`** — set to 0.0 for clean weighted BCE, or keep 0.1? (§5.3)
5. **`w_pos` clamp** — exact ratio (default) or cap at some max? (§5.2)
6. **Early-stopping / selection metric** — validation AUPRC (default) or AUROC?
   (§7)
7. **`target_chain_range`** — keep whole target chain B (default) or crop to the
   sub-chain range? (§8)
8. **Output location** — confirm a new save dir (not `2layers_10hdim_esm_jkmean`).
