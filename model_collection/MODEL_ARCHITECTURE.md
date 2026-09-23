# EntroPPI Model Architecture

This document provides a comprehensive overview of the EntroPPI model architecture, including the core encoder (`EdgeEnhancedHGT`), SSL pre-training models, and the precomputed-embedding fine-tuning pipeline.

**Source files**: `model_collection/EMP_HGT.py`, `model_collection/SSLModels.py`, `model_collection/FineTuneModels.py`

---

## Table of Contents

1. [Core Architecture: EdgeEnhancedHGT](#1-core-architecture-edgeenhancedhgt)
2. [Self-Supervised Learning Models](#2-self-supervised-learning-models)
3. [Application of the Pretrained Encoder](#3-application-of-the-pretrained-encoder)
4. [Training Protocol and Parameters](#4-training-protocol-and-parameters)
5. [Machine Learning Fine-Tuning Pipeline](#5-machine-learning-fine-tuning-pipeline)

---

## 1. Core Architecture: EdgeEnhancedHGT

The **EdgeEnhancedHGT** (Heterogeneous Graph Transformer with 1-Layer Edge Message Passing) is the foundational encoder used by all models. It processes heterogeneous protein-protein interaction graphs with explicit edge feature modeling.

**File**: `model_collection/EMP_HGT.py`

### 1.1 Graph Structure

A protein complex is represented as a **heterogeneous graph**:

$$
\mathcal{G} = \left(\{V^{(\tau)}\}_{\tau \in \mathcal{T}},\; \{E^{(r)}\}_{r \in \mathcal{R}}\right)
$$

where $\mathcal{T} = \{\text{receptor},\, \text{ligand}\}$ is the set of node types and
$\mathcal{R}$ is the set of directed relation types $r = (\tau_s, \phi, \tau_t)$ with source type $\tau_s$, target type $\tau_t$, and relation label $\phi$.

| Element | Description |
|---------|-------------|
| **Node types** $\mathcal{T}$ | `receptor`, `ligand` |
| **Relation types** $\mathcal{R}$ | `(receptor,→,receptor)`, `(ligand,→,ligand)`, `(receptor,→,ligand)`, `(ligand,→,receptor)` |
| **Node features** $\mathbf{x}^{(\tau)}_v$ | base=25, esm=1285, esm480=485 dimensions |
| **Edge features** $\mathbf{e}^{(r)}_{uv}$ | 8 dimensions: 5-bin distance histogram + 3D direction vector |

### 1.2 Model Overview

```
Input Graph → Node Projection → Edge Message Passing → HGT Layers → Node Embeddings (dict)
```

### 1.3 Mathematical Formulation

#### Step 1: Node Feature Projection

All node types share a single linear projection to a unified hidden dimension:

$$
\tilde{\mathbf{h}}^{(\tau)}_v = \mathbf{W}_{\text{node}}\, \mathbf{x}^{(\tau)}_v, \quad \forall v \in V^{(\tau)},\; \tau \in \mathcal{T}
$$

where:
- $\mathbf{x}^{(\tau)}_v \in \mathbb{R}^{d_{\text{node}}}$ is the input feature of node $v$ of type $\tau$
- $\tilde{\mathbf{h}}^{(\tau)}_v \in \mathbb{R}^{d_{\text{hidden}}}$ is the projected embedding
- $\mathbf{W}_{\text{node}} \in \mathbb{R}^{d_{\text{hidden}} \times d_{\text{node}}}$ is the **shared** projection matrix (same weights for all $\tau \in \mathcal{T}$; receptor and ligand nodes share the same input dimension $d_{\text{node}}$)

#### Step 2: Edge Message Passing (Two Modes)

For each directed edge $(u, v) \in E^{(r)}$ with relation $r = (\tau_s, \phi, \tau_t)$, source $u \in V^{(\tau_s)}$, and target $v \in V^{(\tau_t)}$:

**Mode A: Additive Message Passing** (`message_style: additive`)

$$
\mathbf{m}_{uv}^{(r)} = \text{LayerNorm}\left(\mathbf{W}_r^{(2)} \cdot \text{GELU}\left(\text{LayerNorm}\left(\mathbf{W}_r^{(1)} \mathbf{e}^{(r)}_{uv}\right)\right)\right)
$$

where:
- $\mathbf{e}^{(r)}_{uv} \in \mathbb{R}^{d_{\text{edge}}}$ is the edge feature for relation type $r$
- $\mathbf{W}_r^{(1)} \in \mathbb{R}^{d_{\text{bottleneck}} \times d_{\text{edge}}}$ and $\mathbf{W}_r^{(2)} \in \mathbb{R}^{d_{\text{hidden}} \times d_{\text{bottleneck}}}$ are relation-specific weight matrices
- $d_{\text{bottleneck}} = 16$ by default

**Mode B: Gated Source Features** (`message_style: gated_src`, **default**)

$$
g_{uv}^{(r)} = \sigma\left(\mathbf{W}_r^{(2)} \cdot \text{GELU}\left(\text{LayerNorm}\left(\mathbf{W}_r^{(1)} \mathbf{e}^{(r)}_{uv}\right)\right)\right) \in \mathbb{R}
$$

$$
\mathbf{m}_{uv}^{(r)} = g_{uv}^{(r)} \cdot \tilde{\mathbf{h}}^{(\tau_s)}_u
$$

where $\sigma$ is the sigmoid activation. The gate $g_{uv}^{(r)}$ is a scalar that modulates the source node embedding $\tilde{\mathbf{h}}^{(\tau_s)}_u \in \mathbb{R}^{d_{\text{hidden}}}$, conditioned on edge features of relation $r$.

#### Step 3: Message Aggregation with Per-Relation Scaling

For each target node $v \in V^{(\tau_t)}$, messages from all incoming relation types are aggregated:

$$
\mathbf{a}_v^{(r)} = \sum_{u \in \mathcal{N}_r(v)} \mathbf{m}_{uv}^{(r)}, \qquad
\tilde{\mathbf{a}}_v^{(r)} = \sigma(\alpha_r) \cdot \mathbf{a}_v^{(r)}
$$

$$
\mathbf{h}_v^{(0,\tau_t)} = \tilde{\mathbf{h}}^{(\tau_t)}_v + \sum_{r \in \mathcal{R}(\cdot,\cdot,\tau_t)} \tilde{\mathbf{a}}_v^{(r)}
$$

where:
- $\mathcal{N}_r(v)$ is the set of source neighbours of $v$ under relation $r = (\tau_s, \phi, \tau_t)$
- $\mathcal{R}(\cdot, \cdot, \tau_t) = \{r \in \mathcal{R} \mid r = (\tau_s, \phi, \tau_t)\}$ selects only relations whose target type is $\tau_t$
- $\alpha_r \in \mathbb{R}$ is a learnable scalar per relation type, initialized small

#### Step 4: HGT Layers

The edge-enhanced node embeddings $\mathbf{H}^{(0)} = \{\mathbf{H}^{(0,\tau)}\}_{\tau \in \mathcal{T}}$ are further refined by a stack of $L$ HGT blocks. Each block receives the full heterogeneous node embedding collection and the graph structure (edge indices per relation type), and returns updated node embeddings. Internally, HGTConv applies type-specific and relation-specific Q/K/V projections and relation-aware multi-head attention, guided by the metagraph metadata supplied at construction. The residual addition, shared LayerNorm, and edge dropout are applied externally in the encoder's forward pass:

$$
\mathbf{H}^{(l)} = \text{LayerNorm}\!\left(\mathbf{H}^{(l-1)} + \text{HGTConv}^{(l)}\!\left(\mathbf{H}^{(l-1)},\, \mathcal{G}\right)\right)
$$

where $\mathbf{H}^{(l)} = \{\mathbf{H}^{(l,\tau)}\}_{\tau \in \mathcal{T}}$ collects node embeddings for all types after layer $l$, and $\mathcal{G}$ denotes the heterogeneous graph structure. LayerNorm uses shared weights but is applied independently to each node type's embedding matrix. Optional gradient checkpointing is available.

---

## 2. Self-Supervised Learning Models

**File**: `model_collection/SSLModels.py`

Two SSL models are provided, both using EdgeEnhancedHGT as the encoder with a masked edge prediction (MEP) objective.

### 2.1 SSLEdgePredictor (Inter-Molecular Only)

Masks and reconstructs inter-molecular edges only.

```
Input Graph → Mask inter-molecular edges → Encoder → Edge feature construction → Binary prediction
```

### 2.2 SSLInterIntraEdgePredictor (Inter + Intra)

The primary SSL model. Jointly masks inter-molecular and intra-molecular edges, training with a combined loss:

$$
\mathcal{L}_{\text{SSL}} = \beta \cdot \mathcal{L}_{\text{inter}} + (1 - \beta) \cdot \mathcal{L}_{\text{intra}}
$$

where $\beta$ is controlled by `intra_edge_beta` (default 0.65). Intra-molecular masking targets edges on residues adjacent to the interface.

### 2.3 Edge Feature Construction

Node embeddings from the final HGT layer carry explicit type superscripts. Edge features are constructed by concatenating endpoint embeddings with their difference and element-wise product.

**Inter-molecular edges** $(u, v) \in E^{(\text{rec}\to\text{lig})}$, with $u \in V^{(\text{receptor})}$ and $v \in V^{(\text{ligand})}$:

$$
\mathbf{f}^{(\text{inter})}_{uv} = \left[\mathbf{h}^{(\text{receptor})}_u \parallel \mathbf{h}^{(\text{ligand})}_v \parallel |\mathbf{h}^{(\text{receptor})}_u - \mathbf{h}^{(\text{ligand})}_v| \parallel \mathbf{h}^{(\text{receptor})}_u \odot \mathbf{h}^{(\text{ligand})}_v\right] \in \mathbb{R}^{4d_{\text{hidden}}}
$$

**Intra-molecular edges** $(u, v) \in E^{(\tau\to\tau)}$, with $u, v \in V^{(\tau)}$ for $\tau \in \mathcal{T}$:

$$
\mathbf{f}^{(\text{intra},\tau)}_{uv} = \left[\mathbf{h}^{(\tau)}_u \parallel \mathbf{h}^{(\tau)}_v \parallel |\mathbf{h}^{(\tau)}_u - \mathbf{h}^{(\tau)}_v| \parallel \mathbf{h}^{(\tau)}_u \odot \mathbf{h}^{(\tau)}_v\right] \in \mathbb{R}^{4d_{\text{hidden}}}
$$

### 2.4 Edge Prediction Head

A 2-layer MLP (default `predictor_layers=2`). The first layer applies LayerNorm, GELU, and dropout; the final layer is a plain linear projection to a scalar logit. The input is the typed edge feature $\mathbf{f}^{(r)}_{uv}$ (inter or intra, as above):

$$
\mathbf{z} = \text{Dropout}\left(\text{GELU}\left(\text{LayerNorm}\left(\mathbf{W}_1 \mathbf{f}^{(r)}_{uv}\right)\right)\right) \in \mathbb{R}^{d_{\text{hidden}}}
$$

$$
\hat{y}_{uv} = \mathbf{w}_2^\top \mathbf{z} \in \mathbb{R}
$$

where $\mathbf{W}_1 \in \mathbb{R}^{d_{\text{hidden}} \times 4d_{\text{hidden}}}$ and $\mathbf{w}_2 \in \mathbb{R}^{d_{\text{hidden}}}$. The `SSLInterIntraEdgePredictor` uses two **separate** heads of this form — one for inter-edges (relation $\text{receptor} \to \text{ligand}$) and one for intra-edges (relations $\tau \to \tau$) — sharing only the encoder.

### 2.5 Loss

Binary cross-entropy with positive (masked true edges) and negative (sampled non-edges) samples:

$$
\mathcal{L}_{\text{edge}} = -\frac{1}{|\mathcal{E}_{\text{pos}}| + |\mathcal{E}_{\text{neg}}|} \sum_{(u,v)} \left[y_{uv} \log \sigma(\hat{y}_{uv}) + (1 - y_{uv}) \log(1 - \sigma(\hat{y}_{uv}))\right]
$$

---

## 3. Application of the Pretrained Encoder

**Source files**: `model_collection/FineTuneModels.py`, `model_collection/EMP_HGT.py`

Once the SSL encoder has been trained, its only output is a per-node embedding dictionary $\{\tau : \mathbf{H}^{(\tau)} \in \mathbb{R}^{N_\tau \times d_{\text{enc}}}\}$ (Section 1). All three downstream tasks consume this representation through a common application stack consisting of (i) an offline aggregation across HGT layers (Jumping Knowledge), and (ii) an online trainable *Pool + Head* module that maps the node-level embeddings to a task-specific scalar prediction. The encoder itself is **frozen** throughout this stage; only the components introduced in this section carry trainable parameters.

### 3.1 Precomputed-Embedding Pipeline

```
SSL Encoder (frozen) ──► precompute_embeddings() ──► graph.x = node embeddings
                                                            │
                                                     ┌──────┴───────┐
                                                     │  Pool + Head │  (trainable)
                                                     └──────────────┘
```

The encoder is loaded once via `load_pretrained_encoder()`, placed in `eval()` mode with all parameters frozen, and applied to every graph in batches inside `precompute_embeddings()`. The resulting node embeddings (optionally JK-aggregated; see Section 3.2) are written back into each graph as `graph[τ].x` and the encoder is deleted from GPU memory. Subsequent Pool+Head training reads only these stored tensors, which decouples representation learning from task-specific training and substantially reduces GPU memory pressure.

The pipeline also supports auxiliary ESM-2 sequence embeddings. For `embedding_type ∈ {+esm, +esm480}` the per-residue ESM vector is concatenated to the encoder output at the node level ($d_{\text{in}} = d_{\text{enc}} + d_{\text{ESM}}$); for `only_esm` the encoder output is replaced entirely by the ESM embedding.

### 3.2 Jumping Knowledge (JK-Net) Aggregation

When `use_jk: True`, the encoder is invoked with `return_all_layers=True`, returning the $(L+1)$ intermediate outputs $\{\mathbf{H}^{(0)}, \mathbf{H}^{(1)}, \ldots, \mathbf{H}^{(L)}\}$ (the initial edge-message-passing output followed by each of the $L$ HGT layers). These are collapsed into a single per-node tensor inside `precompute_embeddings()` via `_aggregate_jk_layers()`:

$$
\mathbf{H}^{(\text{JK}),(\tau)} = \text{Agg}\!\left(\mathbf{H}^{(0),(\tau)}, \mathbf{H}^{(1),(\tau)}, \ldots, \mathbf{H}^{(L),(\tau)}\right), \quad \tau \in \mathcal{T}
$$

| `jk_mode` | Aggregation | Output dim |
|-----------|-------------|------------|
| `sum` | $\sum_l \mathbf{H}^{(l)}$ | $d_{\text{enc}}$ |
| `mean` | $\tfrac{1}{L+1} \sum_l \mathbf{H}^{(l)}$ | $d_{\text{enc}}$ |
| `max` | $\max_l \mathbf{H}^{(l)}$ (element-wise) | $d_{\text{enc}}$ |
| `concat` | $[\mathbf{H}^{(0)} \parallel \cdots \parallel \mathbf{H}^{(L)}]$ | $d_{\text{enc}} \times (L+1)$ |

**Why JK aggregation is beneficial with a frozen encoder.** The classical motivations for JK-Net — mitigating over-smoothing during training and supporting per-node adaptive depth via gradient flow — apply primarily to the optimisation phase of a GNN. Here the encoder is already trained and frozen, so those effects are fixed. The benefit retained in the inference/fine-tuning regime is *multi-scale representation*: each HGT layer encodes a different spatial scale of the complex (the initial edge-MP output reflects direct-contact geometry; intermediate layers capture secondary-structure patches and interface motifs; the deepest layer captures global fold context). Aggregating across layers — analogously to feature pyramid networks in computer vision — supplies the downstream Pool+Head with a richer, task-agnostic multi-resolution descriptor, and the mean variant additionally reduces the variance of the input representation.

### 3.3 Pool + Head Module

After JK aggregation, each graph carries a precomputed node-embedding dictionary $\{\tau : \mathbf{H}^{(\tau)} \in \mathbb{R}^{N_\tau \times d_{\text{in}}}\}$. The Pool+Head module — implemented in `_PoolHeadBase` and shared by all three task-specific models — transforms this into a scalar prediction in three stages: an input projection, a cross-attention pooler, and a progressive-bottleneck MLP head.

#### 3.3.1 Input projection

A shared linear projection maps the per-node feature dimension to the head's working width $d_{\text{ft}}$:

$$
\tilde{\mathbf{H}}^{(\tau)} = \text{GELU}\!\left(\text{LN}\!\left(\mathbf{W}_{\text{proj}}\, \mathbf{H}^{(\tau)}\right)\right) \in \mathbb{R}^{N_\tau \times d_{\text{ft}}}
$$

$d_{\text{in}}$ depends on the JK mode and embedding type (e.g. $d_{\text{enc}}$ for `jk_mean`, $d_{\text{enc}}\,(L+1)$ for `jk_concat`, with $d_{\text{ESM}}$ added under `+esm`). The head width $d_{\text{ft}}$ is determined at runtime by

$$
d_{\text{ft}} = \min\!\bigl(\texttt{ft\_hdim}_{\text{cfg}},\; \lfloor d_{\text{in}} / 2 \rfloor\bigr)
$$

so the YAML value (typically 512) acts as an upper bound; the actual head width is always at most half of $d_{\text{in}}$. The projection is therefore active in all practical configurations.

#### 3.3.2 Cross-attention aggregation

The projected node embeddings are reduced to a fixed-size graph vector by the `CrossAttentionAggregator` (defined in `EMP_HGT.py` and used exclusively by the Pool+Head module). Learnable query banks $\mathbf{Q}_{\tau} \in \mathbb{R}^{Q \times d_{\text{ft}}}$ — one per chain — attend over the node embeddings of their own chain (Stage 1), then attend across to the partner chain's nodes (Stage 2), before the two attended summaries are concatenated (Stage 3):

**Stage 1 — Independent query-to-node attention.** For each $\tau \in \{\text{rec}, \text{lig}\}$:

$$
\mathbf{A}_{\tau}^{(1)} = \text{MultiHeadAttention}\!\left(\mathbf{Q}_{\tau},\; \tilde{\mathbf{H}}^{(\tau)},\; \tilde{\mathbf{H}}^{(\tau)}\right) \in \mathbb{R}^{Q \times d_{\text{ft}}}
$$

**Stage 2 — Cross-chain interaction (residual + LN).**

$$
\mathbf{A}_{\text{rec}}^{(2)} = \text{LN}\!\left(\mathbf{A}_{\text{rec}}^{(1)} + \text{MultiHeadAttention}\!\bigl(\mathbf{A}_{\text{rec}}^{(1)},\, \tilde{\mathbf{H}}^{(\text{lig})},\, \tilde{\mathbf{H}}^{(\text{lig})}\bigr)\right)
$$

$$
\mathbf{A}_{\text{lig}}^{(2)} = \text{LN}\!\left(\mathbf{A}_{\text{lig}}^{(1)} + \text{MultiHeadAttention}\!\bigl(\mathbf{A}_{\text{lig}}^{(1)},\, \tilde{\mathbf{H}}^{(\text{rec})},\, \tilde{\mathbf{H}}^{(\text{rec})}\bigr)\right)
$$

**Stage 3 — Flatten and concatenate.**

$$
\mathbf{g} = \text{LN}\!\left(\bigl[\text{flatten}(\mathbf{A}_{\text{rec}}^{(2)}) \;\Vert\; \text{flatten}(\mathbf{A}_{\text{lig}}^{(2)})\bigr]\right) \in \mathbb{R}^{d_{\text{g}}}, \qquad d_{\text{g}} = 2 \cdot Q \cdot d_{\text{ft}}
$$

The number of queries $Q$ is `cross_attn_queries` (2 for ΔG and binder discrimination, 4 for ΔΔG in the production configs). Node sequences are padded to the maximum graph size in the batch with attention masks applied so that padded positions do not contribute. (`_PoolHeadBase` also exposes a fallback `pool_mode ∈ {attn, mean, max}` for ablations, but this is not used in any current production config.)

#### 3.3.3 Progressive-bottleneck MLP head

The graph vector $\mathbf{g}$ (or, for ΔΔG, the combined mut/wt vector — see Section 3.5) is mapped to a scalar by an $L$-layer MLP with a halving bottleneck. Letting $\mathbf{g}_{\text{in}}$ denote the head input,

$$
\mathbf{z}_0 = \text{Dropout}\!\left(\text{GELU}\!\left(\text{LN}\!\left(\mathbf{W}_0\, \mathbf{g}_{\text{in}}\right)\right)\right) \in \mathbb{R}^{d_{\text{ft}}}
$$

$$
\mathbf{z}_k = \text{Dropout}\!\left(\text{GELU}\!\left(\text{LN}\!\left(\mathbf{W}_k\, \mathbf{z}_{k-1}\right)\right)\right) \in \mathbb{R}^{d_{\text{ft}}\,/\,2^k}, \quad k = 1, \ldots, L-2
$$

$$
\hat{y} = \mathbf{w}_{L-1}^{\top}\, \mathbf{z}_{L-2} \in \mathbb{R}
$$

The final linear has no activation. With $L = 3$ (all production configs), the per-layer widths are $d_{\text{in,head}} \to d_{\text{ft}} \to d_{\text{ft}}/2 \to 1$, where $d_{\text{in,head}} = d_{\text{g}}$ for ΔG and binder discrimination, or $3\, d_{\text{g}}$ for ΔΔG (Section 3.5).

### 3.4 ΔG Prediction (`PoolHeadRegressor`)

Predicts the binding free energy ΔG of a single complex.

$$
\mathbf{g} = \text{CrossAttn}\!\left(\text{Proj}\bigl(\mathbf{H}^{(\text{rec})}\bigr),\; \text{Proj}\bigl(\mathbf{H}^{(\text{lig})}\bigr)\right), \qquad \hat{y}_{\Delta G} = \text{MLP}_{\text{reg}}(\mathbf{g})
$$

**Loss** (default `loss_fun: logcosh`; alternatives: `mse`, `l1`, `huber`):

$$
\mathcal{L}_{\Delta G} = \frac{1}{B}\sum_{i=1}^{B}\log\cosh\!\bigl(\hat{y}_i - y_i\bigr)
$$

Trained with 5-fold stratified cross-validation; primary validation metric: Pearson $r$.

### 3.5 ΔΔG Mutation Prediction (`PoolHeadDDGRegressor`)

Predicts the change in binding affinity, $\Delta\Delta G = \Delta G_{\text{mut}} - \Delta G_{\text{wt}}$, from **paired** mutant and wild-type graphs. Each graph is independently projected and pooled with shared weights:

$$
\mathbf{g}_{\text{mut}} = \text{CrossAttn}\!\left(\text{Proj}\bigl(\mathbf{H}^{(\text{rec})}_{\text{mut}}\bigr),\; \text{Proj}\bigl(\mathbf{H}^{(\text{lig})}_{\text{mut}}\bigr)\right), \quad
\mathbf{g}_{\text{wt}} = \text{CrossAttn}\!\left(\text{Proj}\bigl(\mathbf{H}^{(\text{rec})}_{\text{wt}}\bigr),\; \text{Proj}\bigl(\mathbf{H}^{(\text{lig})}_{\text{wt}}\bigr)\right)
$$

The two graph vectors are combined according to `ddg_input_mode`:

- `concat_diff` (default):

$$\mathbf{x} = \bigl[\mathbf{g}_{\text{mut}} - \mathbf{g}_{\text{wt}} \;\Vert\; \mathbf{g}_{\text{mut}} \;\Vert\; \mathbf{g}_{\text{wt}}\bigr] \in \mathbb{R}^{3 d_{\text{g}}}$$

the first MLP linear becomes $\mathbf{W}_0 \in \mathbb{R}^{d_{\text{ft}} \times 3 d_{\text{g}}}$.

- `diff`:

$$\mathbf{x} = \mathbf{g}_{\text{mut}} - \mathbf{g}_{\text{wt}} \in \mathbb{R}^{d_{\text{g}}}$$

$$
\hat{y}_{\Delta\Delta G} = \text{MLP}_{\text{ddg}}(\mathbf{x}) \in \mathbb{R}
$$

Trained with 5-fold random or group KFold CV; loss and primary metric identical to ΔG.

### 3.6 Binder Discrimination (`PoolHeadClassifier`)

Binary classification of true binders vs. decoys.

$$
\mathbf{g} = \text{CrossAttn}\!\left(\text{Proj}\bigl(\mathbf{H}^{(\text{rec})}\bigr),\; \text{Proj}\bigl(\mathbf{H}^{(\text{lig})}\bigr)\right), \qquad \hat{l} = \text{MLP}_{\text{cls}}(\mathbf{g})
$$

**Loss** (BCE with label smoothing $\varepsilon$, default `label_smoothing = 0.1`):

$$
\mathcal{L}_{\text{cls}} = -\frac{1}{B}\sum_{i=1}^{B}\bigl[\tilde{y}_i \log\sigma(\hat{l}_i) + (1-\tilde{y}_i)\log\bigl(1 - \sigma(\hat{l}_i)\bigr)\bigr], \qquad \tilde{y}_i = (1-\varepsilon)\,y_i + \tfrac{\varepsilon}{2}
$$

Two training modes are supported, controlled by `n_folds` in the disc-binder config (or the `-nfold` CLI flag of `finetune_disc_binder_module.py`):

- **Single split mode** (`n_folds = 0`, default): a fixed train/test split is taken from the SSL split file, early stopping is driven by the held-out test AUROC, and a single checkpoint `disc_binder{_emb}{_pool}{_jkmode}_best.pt` is saved.
- **K-fold CV mode** (`n_folds > 0`, typically 5): a single pool of all positives plus matched negatives (all SWAP samples plus a half/half PrePPI / random-mutation top-up) is partitioned with `StratifiedKFold`; no held-out test set is used. A fresh Pool+Head is trained per fold with its own early stopping on the fold's validation AUROC, and per-fold checkpoints `disc_binder{_emb}{_pool}{_jkmode}_fold{k}_best.pt` are written alongside a `_{n}fold_summary.json` reporting the per-fold and mean ± std AUROC.

In both modes the primary validation metric is AUROC.

### 3.7 Factory: `create_poolhead_model()`

All three Pool+Head models are instantiated through a single factory:

```python
model = create_poolhead_model(
    model_type='regressor',        # 'classifier' | 'regressor' | 'ddg_regressor'
    pool_input_dim=...,            # d_in (from JK + embedding_type)
    hidden_dim=512,                # ft_hdim (upper bound)
    metadata=METADATA,
    pool_mode='cross_attn',
    cross_attn_queries=2,
    cross_attn_heads=4,
    regressor_layers=3,
)
```

---

## 4. Training Protocol and Parameters

This section specifies the optimisation procedure and hyperparameters used to train all models. A unified notation is used throughout:

| Symbol | Meaning |
|--------|---------|
| $\boldsymbol{\theta}_t \in \mathbb{R}^P$ | Trainable parameters at optimiser step $t$ |
| $\mathbf{g}_t = \nabla_{\boldsymbol{\theta}} \mathcal{L}(\boldsymbol{\theta}_t)$ | Gradient of the task loss |
| $\eta_t$ | Learning rate at step $t$ |
| $\lambda$ | Decoupled weight-decay coefficient |
| $c_t$ | Gradient-norm clipping threshold at step $t$ |
| $T_0,\, T_{\text{mult}}$ | Cosine warm-restart period and multiplier |

### 4.1 Hardware and Software Environment

All experiments were carried out on a single workstation equipped with an **NVIDIA GeForce RTX 5090** GPU running **CUDA 12.8**. The software stack is PyTorch (CUDA 12.8 build) with PyTorch Geometric for heterogeneous graph operations. Mixed-precision training is enabled by default on CUDA devices via `torch.amp.autocast` with **bfloat16**, which reduces memory footprint and improves throughput on the RTX 5090's tensor cores with no observable accuracy loss for the tasks reported here. Model and SSL pre-training runs additionally use gradient checkpointing on the HGT stack when `use_checkpoint: True`, trading recomputation for further memory savings.

### 4.2 Optimiser and Parameter Update

All training stages — SSL pre-training and the three downstream Pool+Head tasks — use the **AdamW** optimiser with decoupled weight decay. A single per-step update combines the gradient step and the weight-decay term:

$$
\boldsymbol{\theta}_{t+1} = \bigl(1 - \eta_t \lambda\bigr)\, \boldsymbol{\theta}_t \;-\; \eta_t \,\widehat{\mathbf{g}}_t
$$

where $\widehat{\mathbf{g}}_t$ is the Adam-rescaled (bias-corrected first and second moment) version of the clipped gradient (Section 4.4). The learning rate $\eta_t$ follows the schedule in Section 4.3 and the weight-decay coefficient $\lambda$ is task-specific (Section 4.7).

### 4.3 Learning-Rate Schedule

The learning rate is annealed by **cosine annealing with warm restarts** (Loshchilov & Hutter, 2017):

$$
\eta_t = \eta_{\min} + \tfrac{1}{2}\bigl(\eta_{\max} - \eta_{\min}\bigr)\!\left[1 + \cos\!\left(\frac{T_{\text{cur}}}{T_i}\,\pi\right)\right]
$$

$\eta_{\max}$ is the configured initial learning rate `lr`, $\eta_{\min}$ is `scheduler_eta_min` ($10^{-6}$ for all downstream tasks). $T_{\text{cur}}$ is the number of epochs since the last restart, and the restart period grows geometrically as $T_{i+1} = T_{\text{mult}}\,T_i$, starting from $T_0$ = `scheduler_t0` = 10 epochs with $T_{\text{mult}}$ = 2.

### 4.4 Adaptive Gradient Clipping

Gradients are clipped by global norm before being passed to the optimiser. To accommodate larger gradient magnitudes early in training while tightening as gradients stabilise, the clipping threshold is updated adaptively after each step:

$$
c_{t+1} = \max\!\bigl(c_{\min},\; \alpha\,\lVert\mathbf{g}_t\rVert_2\bigr)
$$

with $c_{\min} = 5.0$ (configured by `clip_max_norm`) and $\alpha = 1.25$. The clipped gradient $\mathbf{g}_t \leftarrow \mathbf{g}_t \cdot \min\!\bigl(1,\, c_t / \lVert\mathbf{g}_t\rVert_2\bigr)$ is then fed into the AdamW update of Section 4.2.

### 4.5 Regularisation

Three complementary regularisers are used:

- **Dropout**: applied inside the encoder (`emb_drp_rate`, `nn_drp_rate`) and inside every Pool+Head layer (`dropout`, configured per task).
- **Edge dropout**: at training time, a fraction `edge_drp_rate` of the edges in each non-final HGT layer is randomly removed via `dropout_edge`, encouraging the encoder to remain robust to perturbed contact graphs.
- **Label smoothing**: the binder-discrimination loss uses smoothing $\varepsilon = 0.1$ (Section 3.6).

### 4.6 Early Stopping

Training is halted when the task's validation metric has not improved for `patience` consecutive epochs. The monitored metric differs by task:

| Task | Validation metric | Direction |
|------|-------------------|-----------|
| SSL pre-training | edge-prediction loss | minimise |
| ΔG regression (`dg_reg`) | Pearson $r$ | maximise |
| ΔΔG mutation (`mutation`) | Pearson $r$ | maximise |
| Binder discrimination (`disc_binder`) | AUROC | maximise |

The checkpoint with the best validation metric is retained as the final model.

### 4.7 Per-Task Hyperparameters

The values below correspond to the production configs in `model_configs/`.

| Hyperparameter | SSL | ΔG | ΔΔG | Binder Disc. |
|----------------|:---:|:---:|:---:|:---:|
| Batch size | 40 | 30 | 40 | 50 |
| Max epochs | 1500 | 1000 | 1000 | 1000 |
| Initial LR $\eta_{\max}$ | $1\!\times\!10^{-4}$ | $5\!\times\!10^{-4}$ | $1\!\times\!10^{-4}$ | $2\!\times\!10^{-4}$ |
| Min LR $\eta_{\min}$ | $0$ | $10^{-6}$ | $10^{-6}$ | $10^{-6}$ |
| Weight decay $\lambda$ | $10^{-3}$ | $5\!\times\!10^{-3}$ | $5\!\times\!10^{-3}$ | $2\!\times\!10^{-2}$ |
| Clip floor $c_{\min}$ | 5.0 | 5.0 | 5.0 | 5.0 |
| Restart period $T_0$ | 10 | 10 | 10 | 10 |
| Restart multiplier $T_{\text{mult}}$ | — | 2 | 2 | 2 |
| Patience | 50 | 50 | 50 | 20 |
| Mixed precision (bf16) | yes | yes | yes | yes |

---

## 5. Machine Learning Fine-Tuning Pipeline

**Source files**: `finetune_ml_dG_module.py`, `finetune_ml_ddG_module.py`, `finetune_ml_disc_binder_module.py`, `utils/Training_modules/ml_utils.py`  
**Config**: `model_configs/config_ml_full_pipeline.yaml`

As a complement to the neural Pool+Head pipeline (Section 3), all three downstream tasks can also be solved by replacing the trainable MLP head with classical **scikit-learn** estimators trained on frozen encoder embeddings. The encoder pre-processing steps (loading, precomputation, JK aggregation — Sections 3.1–3.2) are identical; the only change is what happens after the per-graph feature vector is formed.

### 5.1 Pipeline Overview

```
SSL Encoder (frozen)
       │
       ▼
 precompute_embeddings()   ← identical to Pool+Head pipeline
       │
       ▼
  mean_pool_graph()        ← mean over all receptor + ligand nodes
       │
       ▼
 [X_train, y_train]  →  sklearn fit()  →  [X_test]  →  sklearn predict()
```

The encoder is loaded once, all graphs are embedded, then the encoder is deleted from GPU memory. All subsequent operations run entirely on CPU.

### 5.2 Feature Extraction by Task

#### ΔG Regression and Binder Discrimination

For each graph $\mathcal{G}_i$, the receptor and ligand node embeddings are stacked and mean-pooled to a single fixed-length vector:

$$
\mathbf{x}_i = \frac{1}{N^{(\text{rec})}_i + N^{(\text{lig})}_i} \sum_{\tau \in \mathcal{T}} \sum_{v \in V^{(\tau)}_i} \mathbf{h}^{(\text{JK},\tau)}_v \;\in\; \mathbb{R}^{d_{\text{JK}}}
$$

where $\mathbf{h}^{(\text{JK},\tau)}_v$ is the JK-aggregated node embedding from Section 3.2 and $d_{\text{JK}}$ is the resulting per-node dimension. In the production config (`jk_mode: concat`, $L$ HGT layers), $d_{\text{JK}} = d_{\text{enc}} \times (L+1)$; for sum/mean/max modes, $d_{\text{JK}} = d_{\text{enc}}$. See the JK mode table in Section 3.2 for the full mapping.

#### ΔΔG Regression

For each mutant–wildtype pair $(\mathcal{G}_{\text{mut}}, \mathcal{G}_{\text{wt}})$, both graphs are pooled independently and concatenated:

$$
\mathbf{x}_i = \bigl[\mathbf{x}_{\text{mut},i} \;\Vert\; \mathbf{x}_{\text{wt},i}\bigr] \in \mathbb{R}^{2\,d_{\text{JK}}}
$$

> **Note**: This differs from the Pool+Head `concat_diff` mode (Section 3.5), which additionally includes the difference $\mathbf{g}_{\text{mut}} - \mathbf{g}_{\text{wt}}$ as a third block. The ML pipeline uses the simpler two-block concatenation, as the tree-based and kernel models can learn the difference implicitly from the two components.

### 5.3 Models

Four model families are evaluated for each task. Regressors are used for ΔG and ΔΔG; classifiers for binder discrimination. The table below lists the scikit-learn class and the key hyperparameters drawn from the config.

| Model | Regressor class | Classifier class | Key hyperparameters |
|-------|----------------|-----------------|---------------------|
| **Random Forest** | `RandomForestRegressor` | `RandomForestClassifier` | `n_estimators`, `max_depth`, `min_samples_leaf`, `max_features` |
| **Gradient Boosting** | `HistGradientBoostingRegressor` | `HistGradientBoostingClassifier` | `max_iter` (`=n_estimators`), `max_depth`, `learning_rate` |
| **Decision Tree** | `DecisionTreeRegressor` | `DecisionTreeClassifier` | `max_depth`, `min_samples_leaf`; classifier adds `class_weight='balanced'` |
| **SVM** | `SVR` (RBF kernel) | `SVC` (RBF kernel) | `C`, `epsilon` (SVR only); classifier adds `class_weight='balanced'`, `probability=True` |

All models are initialised with `random_state = seed` (from `system.seed`) and `n_jobs` for parallelism.

### 5.4 Per-Task Hyperparameters

The `ml` block in the config is keyed by task name. If a task key is absent, the module falls back to top-level `ml` values.

#### ΔG Regression (`ml.dg_reg`) — ~2.9 k train+val, ~160 test

Small dataset; parameters are regularised more aggressively (shallower trees, larger `min_samples_leaf`, lower GBT learning rate).

| Parameter | Value | Applies to |
|-----------|-------|------------|
| `n_jobs` | 20 | RF |
| `rf_n_estimators` | 300 | RF |
| `rf_max_depth` | 10 | RF |
| `rf_min_samples_leaf` | 10 | RF |
| `rf_max_features` | `sqrt` | RF |
| `gbt_n_estimators` | 200 | GBT (`max_iter`) |
| `gbt_max_depth` | 4 | GBT |
| `gbt_learning_rate` | 0.05 | GBT |
| `dt_max_depth` | 8 | DT |
| `dt_min_samples_leaf` | 15 | DT |
| `svm_C` | 10 | SVR |
| `svm_epsilon` | 0.1 | SVR |

#### ΔΔG Regression (`ml.mutation`) — ~5 k train+val, ~140 test

Moderate dataset; regularisation is relaxed relative to `dg_reg`.

| Parameter | Value | Applies to |
|-----------|-------|------------|
| `n_jobs` | 20 | RF |
| `rf_n_estimators` | 300 | RF |
| `rf_max_depth` | 12 | RF |
| `rf_min_samples_leaf` | 5 | RF |
| `rf_max_features` | `sqrt` | RF |
| `gbt_n_estimators` | 300 | GBT |
| `gbt_max_depth` | 5 | GBT |
| `gbt_learning_rate` | 0.1 | GBT |
| `dt_max_depth` | 10 | DT |
| `dt_min_samples_leaf` | 10 | DT |
| `svm_C` | 10 | SVR |
| `svm_epsilon` | 0.1 | SVR |

#### Binder Discrimination (`ml.disc_binder`) — ~18.5 k train, ~4.6 k test

Large dataset; deeper trees, more estimators, lighter SVM regularisation (`C=1`).

| Parameter | Value | Applies to |
|-----------|-------|------------|
| `n_jobs` | 20 | RF |
| `rf_n_estimators` | 500 | RF |
| `rf_max_depth` | 15 | RF |
| `rf_min_samples_leaf` | 5 | RF |
| `rf_max_features` | `sqrt` | RF |
| `gbt_n_estimators` | 500 | GBT |
| `gbt_max_depth` | 6 | GBT |
| `gbt_learning_rate` | 0.1 | GBT |
| `dt_max_depth` | 15 | DT |
| `dt_min_samples_leaf` | 20 | DT |
| `svm_C` | 1 | SVC |

### 5.5 Evaluation Protocols

| Task | Split strategy | Folds | Validation metric | Test metric |
|------|---------------|-------|-------------------|-------------|
| ΔG regression | 5-fold `StratifiedKFold` (stratified on binned $\Delta G$) | 5 | Pearson $r$ (per fold) | Pearson $r$, Spearman $r$, MAE |
| ΔΔG regression | 5-fold `KFold` (`random` or `group` strategy, configurable via `fold_strategy`) | 5 | Pearson $r$ (per fold) | Pearson $r$, Spearman $r$, MAE |
| Binder discrimination | 5-fold `StratifiedKFold` and Single train/test split (no CV) | 5 | Unseen AUPRC (per fold) | Accuracy, F1, AUROC |

For the two regression tasks, each fold trains an independent model on the fold's training partition and evaluates on both the fold's validation partition and the held-out test set. Results are aggregated as mean ± std across the 5 folds and saved to a JSON file (`ml_dg_5fold_results.json` / `ml_ddg_5fold_results.json`). The binder-discrimination task trains once on the full training set and writes `ml_disc_binder_results.json`.

### 5.6 Evaluation Metrics

**Regression** (ΔG and ΔΔG):

$$
r_P = \frac{\sum_i (\hat{y}_i - \bar{\hat{y}})(y_i - \bar{y})}{\sqrt{\sum_i(\hat{y}_i-\bar{\hat{y}})^2 \sum_i(y_i-\bar{y})^2}}, \qquad
\text{MAE} = \frac{1}{N}\sum_i |\hat{y}_i - y_i|
$$

Spearman $r_S$ is computed analogously on rank-transformed predictions. Primary ranking metric: Pearson $r_P$.

**Classification** (binder discrimination):

$$
\text{AUROC} = \int_0^1 \text{TPR}\,d(\text{FPR}), \qquad
F_1 = \frac{2 \cdot \text{precision} \cdot \text{recall}}{\text{precision} + \text{recall}}
$$

Primary ranking metric: AUROC.

---

## Summary

### Model Hierarchy

```
EdgeEnhancedHGT (Core Encoder, frozen after SSL)
    │
    ├── SSLInterIntraEdgePredictor (pre-training)
    │   └── Masked inter + intra edge prediction
    │
    └── Precomputed embeddings
        │
        ├── Pool + Head models (Section 3)
        │   ├── PoolHeadRegressor      (ΔG regression, 5-fold CV)
        │   ├── PoolHeadDDGRegressor   (ΔΔG mutation, paired mut/wt)
        │   └── PoolHeadClassifier     (binder discrimination)
        │
        └── ML models (Section 5) — mean-pool features + sklearn
            ├── RandomForest   (RF, regressor / classifier)
            ├── GradientBoosting (HistGBT, regressor / classifier)
            ├── DecisionTree   (DT, regressor / classifier)
            └── SVM            (SVR / SVC, RBF kernel)
```

### Key Design Decisions

1. **Edge-Enhanced Message Passing**: Explicit edge feature modeling (gated or additive) before HGT layers
2. **Per-Relation Scaling**: Learnable $\alpha_r$ importance weights for each of the 4 edge types
3. **Frozen Encoder + Precomputed Embeddings**: Decouples representation learning from task-specific training, reducing memory and enabling fast iteration
4. **JK-Net**: Aggregates intermediate HGT layer outputs so early and late representations both contribute
5. **Cross-Attention Aggregation**: Learnable queries provide adaptive, task-specific graph-level pooling
6. **ΔΔG Dual-Graph Architecture**: Paired mutant/wild-type processing with difference-based feature combination

---

*For implementation details, see the source code in `model_collection/`. For training configuration, see `model_configs/` and [README_UNIFIED_TRAINING.md](README_UNIFIED_TRAINING.md).*
