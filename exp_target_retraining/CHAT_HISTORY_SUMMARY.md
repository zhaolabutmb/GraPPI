# Chat history summary: experimental binder validation and retraining

This is a condensed record of the discussions and results through 2026-11-04,
not a verbatim transcript. The current leave-one-target-out protocol supersedes
the earlier split proposals described below. Command paths such as
`../GraPPI_data/...` are relative to the repository root; Markdown links are
relative to this file.

## 1. Goal and data

- Evaluate the GraPPI binder classifier against experimental labels from the
  [Zenodo binder-design meta-analysis](https://zenodo.org/records/15722219),
  then retrain its classifier and compare with an ESM-only baseline.
- Keep large datasets, caches, checkpoints, and outputs in the sibling
  `GraPPI_data` directory. The retraining scripts and notebook are in this
  folder.
- The paper reports 3,766 designs and 436 binders. The checksum-verified
  deposited CSV stored as `../GraPPI_data/exp_binder_data/final_dataset.csv`
  contains 3,676 rows: **3,669 labeled designs**, including **394 binders**
  and **3,275 non-binders**, plus seven unlabeled rows. The reason for the
  discrepancy is unknown.
- Labels are `1 = binder` and `0 = non-binder`. Zenodo input structures use
  binder chain A and merged target chain B; GraPPI receives `B,A`. The original
  validation did not crop chain B using `target_chain_range`.

The user asked for the number of examples and positives/negatives per target:

| Target | Samples | Positive | Negative |
|---|---:|---:|---:|
| FGFR2 | 2,123 | 193 | 1,930 |
| EGFR | 434 | 28 | 406 |
| IL7Ra | 171 | 38 | 133 |
| TrkA | 128 | 9 | 119 |
| InsulinR | 117 | 20 | 97 |
| SARS_CoV2_RBD | 99 | 9 | 90 |
| VirB8 | 99 | 9 | 90 |
| Mdm2 | 96 | 55 | 41 |
| pMHC_SILSY1 | 96 | 2 | 94 |
| Pdl1 | 95 | 12 | 83 |
| IL2Ra | 66 | 6 | 60 |
| sntx | 49 | 7 | 42 |
| pMHC_NY1 | 41 | 1 | 40 |
| LTK | 33 | 3 | 30 |
| IL10Ra | 22 | 2 | 20 |
| **Total** | **3,669** | **394** | **3,275** |

Targets are highly imbalanced: FGFR2 accounts for about 58% of samples, and
several targets have very few positives.

## 2. Earlier validation and similarity investigation

### Original GraPPI classifier on the Zenodo data

The initial inference request specified the `2layers_10hdim_esm` encoder and a
classifier checkpoint from `2layers_10hdim_esm_jkmean`. The fold-2 head was
chosen by its original training validation score. The retained historical
prediction CSV has no inference manifest, so the exact checkpoint cannot be
independently proven from that CSV alone.

For the 3,669 labeled designs, the original pooled results were AUROC
**0.5090** and average precision (AP) **0.1169**, versus a positive prevalence
of **0.1074**. Approximately 69.5% of binders and 70.4% of non-binders had
scores of at least 0.9. These observations describe this selected experimental
dataset; no statistical equivalence test was performed.

The user asked whether similar structures with different labels might help
explain the overlap. A full opposite-label structural screen compared
1,290,350 positive/negative pairs at the binder level and followed up
candidate complexes. It found **473 close opposite-label complex pairs**
involving **51 positive** and **237 negative** designs, with a median absolute
GraPPI score difference of **0.00418**. The pairs are not independent; some
designs appear in many pairs. Missing directional alignments and screening
thresholds limit recall, so the count is a lower bound under the stated
criteria. Structural similarity does not establish that it caused the model's
performance or that experimental labels are wrong.

An illustrative FGFR2 pair differed by two binder substitutions, E39L and
Q43T, and had GraPPI scores 0.926903 and 0.924009. Details, thresholds,
limitations, and reproducible outputs are in the local
[preliminary validation report](../../GraPPI_data/exp_binder_data/preliminary_validation_report.md).

### Anthropic dataset

An earlier, separate workflow used the
[Anthropic protein-binder dataset](https://huggingface.co/datasets/Anthropic/claude-protein-binder-design),
`designed.cif` structures, and the `binder_final` label. Of 1,320 labeled
designs, 1,186 were scored: AUROC **0.4711**, AP **0.2440**, and scored-sample
prevalence **0.2715**.

Affinity validation used the `6layers_9hdim_esm` encoder and an ensemble of
five SVR heads. For 322 scored designs, the ensemble had Pearson correlation
**0.2852** and MAE **1.3878 kcal/mol**. GraPPI's affinity output represents
**-DeltaG**, so it was negated for comparison with physical DeltaG; measured
KD was converted to DeltaG at 298.15 K. The Anthropic dataset was subsequently
placed **on hold** for the current retraining work. Its scripts and further
details remain in the local
[Anthropic dataset folder](../../GraPPI_data/anthr_binder_data/README.md).

## 3. How the retraining design evolved

1. The first proposal used a target-based 12-train / 2-validation / 1-test
   split with cross-validation.
2. The user then requested a statistical check of target sizes to guide the
   split. The target sizes were very unequal: maximum/minimum ratio **96.5**,
   coefficient of variation **2.09**, and Gini coefficient **0.663**.
   A sample-level five-fold approach with a held-out test target was
   considered at that stage.
3. After discussing the paper's per-target analysis, the user chose the final
   approach: **15-fold leave-one-target-out (LOTO) validation**. In each fold,
   one target is the validation target and the other 14 targets are used for
   training. Every target is held out once. There is **no separate blind test
   target** in this final design.

The size analysis is reproducible with
[`analyze_target_sizes.py`](./analyze_target_sizes.py); its earlier split
decision is historical and does not override LOTO.

## 4. Final GraPPI and ESM training setup

- **GraPPI:** use the pretrained `2layers_10hdim_esm` encoder, freeze it, and
  train the cross-attention pooling and MLP classifier head. The head
  architecture mirrors `2layers_10hdim_esm_jkmean`. Each fold constructs a new
  head; it does **not** initialize the head from the original MLP checkpoint.
- **ESM baseline:** use the cached 1,280-dimensional ESM node embeddings from
  the same interface graph cache and bypass the GraPPI graph encoder. Use the
  same target folds and classifier architecture/settings.
- **Loss:** weighted binary cross-entropy with logits, hard labels, and fixed
  global `pos_weight = 3275 / 394 = 8.3122` across folds.
- **Selection:** early stopping and best-epoch selection use the held-out
  target's AP/AUPRC.
- **Outputs:** per-target AUROC and AP; median and interquartile range across
  targets; pooled metrics; and per-sample labels, logits, and probabilities for
  later analysis.

The user asked for AUROC and AUPRC; the earlier term “AUPRO” was interpreted as
AUROC. Here AP is scikit-learn's `average_precision_score`, not trapezoidal PR
area.

**Important limitation:** the held-out target's labels were also used to
choose its best epoch. The global positive weight was calculated using all
labeled targets. The reported GraPPI and ESM results can therefore be
optimistic and are **not independent blind-test estimates**. A future blind
comparison would need model selection and any data-derived weighting confined
to training/inner-validation data, followed by evaluation on unseen targets.

## 5. Cache, troubleshooting, and commands

The shared graph cache is
`../GraPPI_data/exp_binder_data/grappi_graphs_esm_8A/graphs_esm_8A.pkl`
(approximately 5.4 GB). It contains structural graphs and ESM embeddings, so
the ESM baseline does not need another ESM forward pass. The cache is under
the dataset folder, not `trained_data_ssl_finetune`.

The initial cache build appeared stuck at `0/3669`; multiprocessing was
changed to spawned CPU workers with thread limits. A later run failed with
`received 0 items of ancdata` / `BrokenProcessPool` because tensor transfers
exhausted file descriptors. Worker results are now serialized to bytes before
transfer; a 120-structure, 20-worker check passed under a 1,024-descriptor
limit.

The user requested explicit GPU IDs so GraPPI and ESM could run concurrently.
These are the commands discussed, from the repository root, using the local PPI
environment. **Both runs have already completed**; rerunning training can
replace outputs in the default directories.

```bash
# Build the shared cache if it is not already available.
/home/zhisong/.conda/envs/PPI/bin/python exp_target_retraining/cache_graphs.py \
    --workers 20 --cuda-id 0

# In separate terminals after the cache is ready:
/home/zhisong/.conda/envs/PPI/bin/python exp_target_retraining/train_exp_target_wbce.py \
    --config exp_target_retraining/config_exp_target_wbce.yaml --mode grappi --cuda-id 0

/home/zhisong/.conda/envs/PPI/bin/python exp_target_retraining/train_exp_target_wbce.py \
    --config exp_target_retraining/config_exp_target_wbce.yaml --mode esm --cuda-id 1
```

`--cuda-id` refers to CUDA-visible device IDs. The default output directories
are:

- GraPPI: `../GraPPI_data/trained_data_ssl_finetune/2layers_10hdim_esm_exp_target_loto_wbce/`
- ESM: `../GraPPI_data/trained_data_ssl_finetune/esm1280_exp_target_loto_wbce/`

Each contains `exp_target_loto_raw_scores.csv`, per-target metrics, split and
summary JSON files, and best checkpoints.

## 6. Completed comparison

Both LOTO runs completed on the same 3,669 samples and 15 targets. The notebook
also recomputes two AF3 Figure 3B feature rankings from the deposited CSV on
that matched cohort:

| Method | Median target AP | AP Q25–Q75 | Median target AUROC | Pooled AP |
|---|---:|---:|---:|---:|
| GraPPI | 0.30981 | 0.16089–0.36464 | 0.67040 | 0.22698 |
| ESM | 0.26619 | 0.17801–0.58521 | 0.63247 | 0.17963 |
| AF3 ipSAE_min | 0.52743 | 0.34196–0.67021 | 0.81481 | 0.34969 |
| AF3 interaction | 0.51970 | 0.37657–0.64276 | 0.81728 | 0.42219 |

The user asked about a fold with AUROC 0.7222 and AP 0.3098. It was **IL2Ra**:
66 designs, six positives, and prevalence 0.0909. On that same target ESM AP
was 0.46712, AF3 ipSAE_min AP was 0.78783, and AF3 interaction AP was 0.78016.
Prevalence is a useful PR reference, but finite-sample expected random-ranking
AP is not always exactly equal to prevalence.

### How the AF3 results relate to the paper

- `AF3_ipSAE` uses `af3_ipSAE_min`, with higher values ranked as more
  binder-like.
- `AF3_interaction` uses
  `-af3_ipSAE_min * af3_rosetta_interface_dG_SASA_ratio`: the paper's raw
  product is lower-is-better, so it is negated for ranking metrics. These are
  arbitrary feature scores, **not probabilities**.
- The table above recomputes the paper's Figure 3B features on the matched
  deposited cohort, aggregated across targets in the style of Figure 3D. It
  does **not** copy the numerical Figure 3 values.
- Separately, the paper reports median AP **0.58** (IQR 0.33–0.66) for
  AF3 LIS × input interface shape complementarity and **0.57** (IQR 0.40–0.66)
  for its AF3 individual-feature logistic model. These are paper-reported
  results for different features/model and the paper's stated cohort, not
  matched-cohort values. See the [preprint](https://www.biorxiv.org/content/10.1101/2025.08.14.670059v2.full).
- The paper's logistic model used nested target-group cross-validation. AF3
  features use re-predicted structures, while the GraPPI models used input
  structures. Pooled predictions from different LOTO models may also be
  calibrated differently across targets.

The earlier original-classifier AP of 0.1169 is **pooled**, whereas the
0.30981 GraPPI result above is a **median of target APs** after retraining and
held-out-target epoch selection. They should not be treated as directly
equivalent estimates.

## 7. Notebook, metrics, and local environment

- Open [`loto_results_summary.ipynb`](./loto_results_summary.ipynb), select the
  PPI kernel, and run all cells. It shows per-target counts, AP/AUROC, lift,
  aggregate metrics, and plots for the two models and two AF3 features.
  `TARGET_TO_PLOT` selects the target for the PR curve.
- The user requested the `aggregates` table be transposed: methods are columns
  and metrics are rows. `AP_iqr` means `AP_q75 - AP_q25`, the interquartile
  range of target-level AP; for GraPPI it is approximately **0.2038**.
- The notebook compares sample IDs, targets, and labels before combining
  results, and errors on missing/non-finite AF3 values rather than silently
  changing the cohort. Earlier checks in this chat ran all notebook cells on
  the real data and passed **24** focused results/notebook tests.
- [`README.md`](./README.md) contains the detailed training instructions;
  [`config_exp_target_wbce.yaml`](./config_exp_target_wbce.yaml) contains the
  settings; [`cache_graphs.py`](./cache_graphs.py),
  [`train_exp_target_wbce.py`](./train_exp_target_wbce.py), and
  [`loto_results.py`](./loto_results.py) implement the pipeline and reporting.
- The repository-local `.conda/` is a separate Conda environment, not the PPI
  environment used for these runs. At inspection it contained Python 3.15.2
  and occupied about 226 MB; PPI used Python 3.12.9. The root `.gitignore` now
  includes `.conda/`. Do not upload the environment binaries to GitHub; the
  tracked `environment.yml` is the starting point for sharing dependencies.
