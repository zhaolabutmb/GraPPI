# Methods: training on experimental binder data, testing on the Anthropic set

<!--
Draft for a manuscript Methods section. Numbers are taken from the run
summaries in
../GraPPI_data/trained_data_ssl_finetune/{2layers_10hdim_esm,esm1280}_exp_train_anthr_test/
and from the two dataset label tables. Sentences marked TODO still need a
decision or a number that does not exist yet.
-->

## Datasets

We trained on the experimental binder-design meta-analysis deposited at Zenodo
(record 15722219) and tested on release v1.0 of the Anthropic
`claude-protein-binder-design` dataset (CC BY 4.0). The deposited
experimental table contains 3,669 designs with a binary experimental outcome,
comprising 394 binders and 3,275 non-binders distributed over 15 protein
targets; target sizes are highly unequal, with FGFR2 alone accounting for
approximately 58% of designs. All 3,669 labeled designs were used for
training, with no held-out target and no internal validation split.

The Anthropic release contains 1,440 de novo miniprotein binders of 50–120
residues against 16 targets, designed by two autonomous model-driven design
agents and characterised by surface plasmon resonance and biolayer
interferometry at two independent vendors. We used the consolidated
`binder_final` outcome, which labels 1,320 designs as 354 binders and 966
non-binders; the 120 mature GDF-8 designs excluded from the release for
inconclusive assay behaviour are absent from this table. Because this dataset
reports experimental binding but no experimentally determined structures, we
used its `designed.cif` files, which carry the design-time coordinates of the
binder together with the target assembly. We note that these are generative
design-model outputs rather than co-folded or experimentally determined
structures, and that their atomic completeness is heterogeneous: of the 1,189
designs we prepared, 556 binder chains contain backbone atoms only (19 of them
Cα traces) while 633 contain side chains.

For both datasets the binder was assigned to chain A and all remaining chains
were merged and renumbered into target chain B, and GraPPI received the
complex as receptor B and ligand A. Interface graphs were constructed with an
8 Å residue-pair distance cutoff. Of the 1,320 labeled Anthropic designs, 131
have no released `designed.cif` and a further 3 yielded no interface at the
8 Å cutoff, leaving 1,186 scored designs (322 binders, 964 non-binders;
positive prevalence 0.2715). Designs without a usable structure were reported
as coverage and excluded from all metrics; they were never imputed as
non-binders.

No design occurs in both datasets, and no test label was used to fit model
parameters, so the evaluation is free of sample-level leakage. Four of the
fifteen test targets are nonetheless represented in the training set — EGFR and
TrkA under identical names, and IL-7Ra and PD-L1 under the training set's
`IL7Ra` and `Pdl1` spellings — accounting for 338 of the 1,186 scored designs
(28.5%). For these targets the evaluation measures generalisation to unseen
binders against a previously seen target rather than to an unseen target, a
weaker claim than for the remaining eleven. Empirically we observed no
advantage for them: their per-target AUROC values (EGFR 0.534, IL-7Ra 0.565,
PD-L1 0.520, TrkA 0.519) fall at or below the median across all targets
(0.543). We report metrics per target rather than pooled over this split,
because the previously seen targets have a higher positive prevalence (0.343
versus 0.243) and include the two most binder-rich targets, so pooling them
inflates average precision through base rate alone.

## Model

The GraPPI encoder is a heterogeneous graph transformer pretrained by
self-supervised edge masking. We used the published two-layer checkpoint with
hidden dimension 2^10 = 1,024, four attention heads and gated-source message
passing, with Jumping-Knowledge mean aggregation over layer outputs. The
encoder was kept frozen throughout and its embeddings were precomputed once per
dataset, so training updated only the task head. Node features were the 1,280
dimensional per-residue embeddings of ESM-2 (`esm2_t33_650M_UR50D`)
concatenated with five structural features, with the ESM block replacing the
20-dimensional one-hot amino-acid encoding.

The classifier head mirrors the architecture of the published binder
classifier: cross-attention pooling with two learnable query vectors and four
attention heads, followed by a two-layer multilayer perceptron with hidden
dimension 256 and dropout 0.4. A fresh head was initialised for every run; no
run warm-started from the published classifier weights.

## Baselines

Two baselines share the head architecture, loss, optimiser settings, training
pool and test cohort, so that only the input representation differs.

The **ESM baseline** bypasses the graph encoder entirely and feeds the
1,280-dimensional ESM-2 node embeddings, read from the same interface-graph
cache, directly into the pooling and classifier head. It therefore isolates the
contribution of the graph encoder over the language-model features alone.

The **GraPPI-base baseline** substitutes the base-feature encoder checkpoint of
identical shape (two layers, hidden dimension 1,024) pretrained on the
25-dimensional structural node features instead of ESM input, isolating the
contribution of language-model features within the same graph architecture.
Because the ESM feature construction overwrites the one-hot amino-acid block,
the base node features cannot be recovered from the ESM graph cache and were
rebuilt from the input structures.

We additionally evaluated the **published classifier** without retraining. Its
fold was selected by the maximum validation score recorded during its original
training, never using the Anthropic labels, and it was scored through the same
interface-graph cache, frozen encoder and metric code as the trained models.

## Training

The classifier was trained with weighted binary cross-entropy on hard 0/1
labels, using a positive weight of 3,275/394 = 8.3122 computed from the
training set alone. We used AdamW with learning rate 2×10⁻⁴, weight decay 0.02,
gradient-norm clipping at 5.0 and a batch size of 50, with cosine annealing and
warm restarts (T₀ = 10, T_mult = 2, minimum learning rate 10⁻⁶) for up to 1,000
epochs with an early-stopping patience of 20 epochs. Mixed-precision bfloat16
was used on GPU. All runs used seed 42. <!-- TODO: repeat over several seeds if
a variance estimate is wanted; results below are single-seed. -->

**Model selection and its limitation.** Early stopping and best-epoch selection
used average precision on the Anthropic test set itself. The Anthropic metrics
reported for the trained models are therefore optimistic and do not constitute a
blind held-out estimate; the selected epochs were 6 of 26 for GraPPI and 10 of
30 for the ESM baseline. The published-classifier evaluation is unaffected by
this, as it involves no training and no label-driven selection. We state this
explicitly because the same caveat applies to our leave-one-target-out analysis,
and a fully blind comparison would require confining model selection to the
training data.

## Evaluation

We report the area under the receiver operating characteristic curve (AUROC)
and average precision (AP, scikit-learn `average_precision_score`), computed
over the scored designs pooled, and stratified by target, by design model and
by assay campaign. Across-stratum performance is summarised as the median and
interquartile range of the per-group values, which weights each group equally
rather than letting the largest group dominate. Groups containing a single
class, such as the Anthropic MBP target with zero binders, admit no ranking
metric; these are reported as undefined and excluded from the medians rather
than assigned a nominal value of 0.5. AP is additionally expressed as lift over
the positive prevalence of the corresponding group, the expected value for
random ranking. Per-design labels, logits and probabilities were saved so that
any further metric can be recomputed without retraining.

## Implementation

Experiments used Python 3.12.9, PyTorch 2.7.0 (CUDA 12.8), PyTorch Geometric
2.6.1 and scikit-learn 1.8.0. Code to reproduce the graph caches, the training
runs, the published-classifier scoring and the comparison tables is in the
`exp_train_anthr_test/` directory of the GraPPI repository.
<!-- TODO: add the public repository URL and the commit hash used for the
final runs. -->
