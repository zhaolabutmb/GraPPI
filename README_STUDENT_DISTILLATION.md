# Sequence Student Model via Knowledge Distillation

> Design document for future implementation. Records the agreed pipeline for training a
> **sequence-only student model** that reproduces GraPPI complex embeddings, enabling
> downstream inference from individual protein sequences (or structures) instead of
> requiring a pre-formed protein complex.

---

## 1. Motivation

The pretrained GraPPI encoder operates on **protein complexes** converted to heterogeneous
graphs. Its self-supervised objective (masked edge prediction, including inter-molecular
`receptor↔ligand` edges) bakes the **interface geometry** into the embedding. Consequently,
every downstream fine-tuning task requires a **complex** as input.

Biologists typically have **sequences** (abundant) rather than co-crystallized complex
structures (scarce). A student model that maps two individual proteins to the *same
embedding space* as GraPPI would let all downstream tasks run from sequence input — a major
usability win for adoption.

**Goal:** the student produces a per-residue embedding dict
`{receptor:[N_rec, H], ligand:[N_lig, H]}` that is a **drop-in replacement** for the frozen
GraPPI encoder output, so the *existing* fine-tuning stack (JK → cross-attention pooling →
task head) works unchanged.

---

## 2. Teacher / Student concept

```
Teacher:  complex het-graph ──GraPPI encoder (frozen, last HGT layer)──▶  dict {rec:[N,H], lig:[N,H]}
                                                                              ▲  (distillation target)
Student:  two ESM sequences ──ESM2(frozen)+cross-attn──────────────────▶  dict {rec:[N,H], lig:[N,H]}

Downstream (use_jk=False): dict ──▶ cross-attn pool ──▶ 1-D ──▶ task head
```

- The teacher's pretrained artifact is **only the encoder** (`EdgeEnhancedHGT`). The
  cross-attention **pooling** that yields a 1-D vector is **not** pretrained — it is trained
  per downstream task. Therefore the **distillation target is a per-residue dict**, **not** a
  compressed 1-D vector. `H = 2^hidden_dim_power`.
- **JK-net: not used.** To keep things simple we distill against the **final HGT layer**
  output only (`return_all_layers=False`) and set **`use_jk=False` in every downstream task**.
  This avoids the extra computation of per-layer embeddings and sidesteps the question of how
  a layerless student would reproduce JK. (Multi-layer / JK enrichment is deferred; it can be
  revisited later via per-layer distillation if needed.)
- The distillation target is exactly the tensor the downstream **pooling+head consume as node
  features** — i.e. the frozen encoder's final-layer output.

---

## 3. Training objectives

### v1 — Distillation only (primary, necessary)

Per-residue regression toward the teacher's exact vectors (1:1 residue correspondence):

```
L_distill = mean_over_residues[ (1 - cos(s_i, t_i)) + λ · || s_i - t_i ||² ]
```

- `s_i` = student residue embedding, `t_i` = teacher residue embedding.
- Cosine aligns direction; MSE pins magnitude → places the student at the teacher's
  **absolute coordinates**, which the downstream heads (trained on teacher embeddings)
  expect.
- Because every residue has a distinct target, representation collapse is unlikely — this
  is why v1 does not need contrastive.

### v2 — Optional contrastive regularizer

CLIP-style InfoNCE on a **mean-pooled summary** (one vector per complex) with **in-batch
negatives** (other complexes' teacher embeddings). Prevents collapse and keeps complexes
separated. Use **foldseek clusters** (`nr_dimer_table`, `dimer_cluster_info`) to avoid
same-cluster false negatives in a batch. Add only if under-separation is observed.

---

## 4. Data reuse (no new ESM computation for training)

Reuse the existing **ESM graph pickles** so sequence and structure are already aligned:

- Load with the SSL machinery: `load_sttgs_from_dir`, `get_db_path_ssl`,
  `filter_samples_for_ssl`, `set_seed` (from `SSL_train_module.py`).
- Each `StructureToHeteroGraph.protein_graph` is a PyG `HeteroData`.
- Node feature layout for `embedding_type='esm'`: `x` is `[N, 1285]` =
  **ESM-2 650M `[:, :1280]`** + 5 structural dims. For `esm480`: `[N, 485]`,
  **ESM `[:, :480]`**.
- **Student input** = `x[:, :esm_dim]` per protein. **Teacher input** = full graph.
- **1:1 correspondence** is guaranteed by extracting the student ESM slice from the *same*
  graph the teacher consumes → receptor residue *i* ↔ receptor residue *i*, no alignment
  code needed.

**Teacher / graph consistency:** the teacher checkpoint's `node_in_dim` must match the
graph feature dim. Cleanest setup: use an **ESM-trained GraPPI teacher** (`node_in_dim=1285`)
with the `esm` pickle set — one graph set serves both teacher (full `x`) and student
(`x[:, :1280]`). Teacher checkpoint path is provided via the **config file**.

**Train/val split:** reuse the SSL 80/20 split with `split_seed = 42`.

**No masking:** unlike SSL, feed the **complete unmasked** complex graph to the teacher so it
produces its true embedding. The student sees **no edges** — only the two ESM tensors.

---

## 5. Teacher embedding cache

The teacher is frozen → compute each complex's dict **once** and reuse across all epochs.

- **Precompute to a disk cache** keyed by `pdb_name` (mirrors the fine-tune
  `precompute_embeddings` pattern). Optionally keep on CPU pinned memory.
- **Move per-batch to GPU** during training. Do **not** keep the entire set resident in VRAM
  (per-residue dicts for ~10k complexes at H=512–1024 can be several GB).
- Load frozen encoder via `load_pretrained_encoder(...)` (eval mode, `requires_grad=False`);
  auto-load architecture from the checkpoint (as fine-tune configs do with `null` keys).
- **Build the target from the final HGT layer:** run `encoder.forward(..., return_all_layers=False)`
  → target dict `[N, H]`. Cache this dict (this is what the student must match and what
  downstream pooling consumes with `use_jk=False`).

---

## 6. Student architecture

A small **cross-encoder** over the two proteins' ESM residue embeddings (no GNN needed —
inputs are already numerical and ESM2 encodes evolutionary + implicit contact info):

1. Project ESM `esm_dim → H` (match teacher hidden dim).
2. `n_blocks` of `[ per-protein self-attention → cross-attention to partner → FFN ]`.
3. Output per-residue `[N, H]` for each tower → assemble the dict.

- No extra positional encoding (ESM already carries position).
- Keep modest (e.g. `n_blocks=2–4`, `n_heads=4–8`) given ~9.6k training complexes.
- **Order convention:** feed tower A→`receptor`, B→`ligand` matching the teacher dict keys
  per sample. Add **swap augmentation** (swap inputs + swap target keys) in a later
  iteration for order robustness.

**Future inference `esm_encoding` (not part of training):** run ESM2 live on a raw sequence
pair. Only here does the ESM2 **~1022 residue input limit** apply — hence the training cap
below keeps train/inference consistent and avoids truncation logic.

---

## 7. Batching & padding

- Transformer/cross-attention over variable-length sequences needs padding (unlike the GNN,
  which batches as one disconnected graph).
- **Dynamic per-batch padding** (pad to the max length *within the batch*), **not** a global
  fixed size — attention is O(L²), so padding to the dataset max is wasteful.
- **Length bucketing** (group similar-length complexes) to minimize padding overhead.
- **Two padding masks per sample** (one per protein tower) for the cross-attention.
- Padding is correctness-neutral **when masked properly**: `key_padding_mask` excludes pad
  tokens from attention, and the per-residue loss is computed only over real residues.
- **Inference:** trim each `[L_max, H]` output back to its real residue count via stored
  lengths → the dict.
- Optional **token-budget batching** (shrink batch size for long buckets so
  `batch_size × L_max²` stays ~constant) if long buckets risk OOM.

### Max-length cap — `max_residues_per_protein` (config, default 1000)

A per-protein ceiling (distinct from padding size) for memory safety and inference
consistency with ESM2's ~1022 limit. Complexes with either protein above the cap are either
excluded or placed in `batch_size=1` buckets.

Justified by the current dataset distribution (see §11): **≤ 2.0%** of samples in any split
exceed 1000 residues on the larger protein, so a 1000-residue cap (2000 total/complex)
discards very little data.

---

## 8. Loss & early stopping (reuse existing utilities)

- **Loss:** `get_loss('mse')` from `utils/Quantity_compute/Loss_fun.py` for the MSE term;
  cosine via `torch.nn.functional.cosine_similarity`.
- **Early stopping:** `SSLEarlyStopping` (`utils/Training_modules/ssl_training.py`) or
  `FineTuneEarlyStopping` (`utils/Training_modules/save_training.py`) — both save best state
  on improvement. Stop on **validation distillation loss**.
- **Training loop:** mirror `train_epoch_ssl` / `validate_ssl` / scheduler / early-stop from
  `SSL_train_module.py`.
- **Validation metrics:** mean per-residue cosine similarity and MSE; optionally cosine of
  the mean-pooled summary.

---

## 9. Config sketch (new `student` section)

```yaml
student:
  teacher_ckpt: ../GraPPI_data/trained_data_ssl_finetune/<esm_teacher>/ssl_edge_best.pt
  embedding_type: esm            # esm|esm480 -> esm_dim 1280|480
  max_residues_per_protein: 1000 # per-protein cap (2000 total/complex)

  # student architecture
  n_blocks: 3
  n_heads: 8
  dropout: 0.2
  hidden_dim: null               # auto = teacher hidden_dim (H)
  # target = final HGT layer (return_all_layers=False); downstream uses use_jk=False

  # loss
  mse_weight: 1.0                # λ
  cosine_weight: 1.0
  contrastive_weight: 0.0        # enable in v2

  # training
  batch_size: 20
  lr: 0.0003
  weight_decay: 0.001
  clip_max_norm: 5.0
  patience: 50
  n_epochs: 1000
  split_seed: 42

  teacher_cache_dir: ../GraPPI_data/student_teacher_cache/
  save_dir: ../GraPPI_data/trained_data_student/
```

---

## 10. Reusable components

| Purpose            | Component / file |
|--------------------|------------------|
| Data loading       | `load_sttgs_from_dir`, `get_db_path_ssl`, `filter_samples_for_ssl`, `set_seed` — `SSL_train_module.py` |
| Frozen teacher     | `load_pretrained_encoder`, `EdgeEnhancedHGT.forward(..., return_all_layers=False)` — `model_collection/FineTuneModels.py`, `model_collection/EMP_HGT.py` |
| Loss               | `get_loss` — `utils/Quantity_compute/Loss_fun.py` |
| Early stopping     | `SSLEarlyStopping`, `FineTuneEarlyStopping` — `utils/Training_modules/` |
| Loop structure     | `train_epoch_ssl` / `validate_ssl` — `SSL_train_module.py`, `utils/Training_modules/ssl_training.py` |

---

## 11. Dataset size statistics (8Å graphs, current)

Basis for the default 1000-residue-per-protein cap.

| Folder                     | N     | Complex min/med/mean/max | Max-protein min/med/mean/max | > 1000 (max-protein) |
|----------------------------|-------|--------------------------|------------------------------|----------------------|
| SSL (pos)                  | 11593 | 21 / 373 / 451 / 4766    | 11 / 245 / 304 / 4529        | 2.0% |
| Negative (PrePPI)          | 15000 | 82 / 252 / 299 / 1475    | 45 / 144 / 197 / 1258        | 0.2% |
| Negative (random mut)      | 8555  | 21 / 347 / 419 / 4210    | 11 / 225 / 276 / 3905        | 1.5% |
| Negative (swapped Ab/Ag)   | 272   | 292 / 589 / 596 / 827    | 223 / 431 / 419 / 438        | 0.0% |
| Binding affinity (unmut)   | 2903  | 42 / 439 / 492 / 3426    | 21 / 317 / 351 / 3302        | 2.0% |
| Binding affinity (mutant)  | 6577  | 57 / 409 / 473 / 3397    | 52 / 262 / 318 / 3119        | 2.0% |

Distribution plots: `database_process/dataset_stats.ipynb`.

---

## 12. Implementation roadmap

1. **Length distribution** (done — `dataset_stats.ipynb`). Set bucket boundaries; confirm cap.
2. **Teacher cache builder** — precompute final-layer dicts (`return_all_layers=False`)
   and save per-`pdb_name` to disk.
3. **Student dataloader** — extract `x[:, :esm_dim]` per protein; dynamic padding + bucketing
   + two masks; apply `max_residues_per_protein`; reuse `split_seed=42`.
4. **Student model** — ESM projection + cross-encoder → per-residue dict.
5. **Trainer** — cosine+MSE distillation loss (masked), early stopping, mirror SSL loop.
6. **Validation** — per-residue cosine/MSE; sanity-check padded vs per-complex equivalence.
7. **Downstream check** — feed student dict into the fine-tune pooling+head with
   **`use_jk=False`**; compare to teacher upper bound and to the existing `only_esm` baseline.
8. **v2 (optional)** — contrastive regularizer + swap augmentation.

---

## 13. Open decisions

1. **Teacher checkpoint** — chosen ESM-trained GraPPI (so one graph set serves both). *(via config)*
2. **Teacher cache** — disk cache → GPU per batch. *(agreed)*
3. **v1 scope** — distillation-only first; contrastive in v2. *(agreed)*
4. **max_residues_per_protein** — default 1000 (configurable). *(agreed)*
5. **JK-net** — not used; distill against **final HGT layer**, `use_jk=False` in all downstream
   tasks. Per-layer/JK enrichment deferred. *(agreed)*

---

## 14. Implementation (v1) — files & usage

**New files**
- `model_collection/StudentModels.py` — `SequenceStudentEncoder` (ESM projection + cross-encoder
  blocks: per-protein self-attention → cross-attention → FFN), `create_student_model`,
  `distillation_loss` (masked cosine + MSE over real residues).
- `utils/Training_modules/student_data_loader.py` — `build_teacher_cache` (frozen encoder →
  per-name fp16 `.pt`), `StudentDistillDataset` (lazy ESM slice + teacher target),
  `pad_and_mask` / `collate_student`, `BucketBatchSampler` (length bucketing).
- `student_train_module.py` — `run_student_training(config, device)`: loads the pos graph set,
  builds the teacher cache, 80/20 split (`split_seed=42`), trains with early stopping, saves
  `student_best.pt` + `student_config.json` + `student_train_val_split.json`.
- `model_configs/config_student.yaml` — `training_task: [student]` with `data` + `student`
  sections.

**Shared dispatch (drop-in encoder swap)**
- `model_collection/FineTuneModels.py` adds:
  - `load_student_encoder(checkpoint_path, device)` — rebuilds the student from
    `student_config.json`, frozen/eval.
  - `precompute_embeddings_student(...)` — runs the student on `x[:, :esm_dim]` per protein and
    writes `graph[nt].x = [N, H]` (same contract as `precompute_embeddings`; no residue cap at
    inference).
  - `build_embedder(...)` — returns `(precompute_fn, hidden_dim, use_jk)` for either
    `encoder_type='ssl'` (GraPPI) or `encoder_type='student'` (student → `use_jk=False`,
    `hidden_dim` from the checkpoint).

**Downstream selection.** Every finetune module (`dg_reg`, `mutation`, `disc_binder`, and the
three `ml_*` variants) now reads `encoder_type` from its config section (default `'ssl'`).
Data loading, labels, CV folds, and metrics are unchanged — only the embedding source differs.

```yaml
# In the task config section (e.g. dg_reg / mutation / disc_binder):
dg_reg:
  encoder_type: student   # 'ssl' (default) or 'student'
  pretrained_path: ../GraPPI_data/trained_data_student/<student_dir>/student_best.pt
data:
  embedding_type: esm     # student requires esm/esm480 graphs
```

**Run**
```bash
python train_unified.py -config model_configs/config_student.yaml     # train the student
# then point a finetune config's pretrained_path at student_best.pt with encoder_type: student
```

**Verified:** all modules import; student forward + masked loss are finite; padded-batch output
is identical to per-complex output for real residues (≈1e-6, floating-point only) — so batching
does not affect learning.
