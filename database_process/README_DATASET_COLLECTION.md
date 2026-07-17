# EntroPPI Dataset Collection

This document describes how the datasets used by EntroPPI were assembled for
self-supervised pre-training (**SSL**) and the three downstream tasks:
binding affinity prediction (**ΔG**), change of affinity upon mutation
(**ΔΔG**), and binder vs. non-binder discrimination (**binder discrimination**).

All scripts and notebooks referenced live in
[database_process/](.) and the resulting tables are written to
[saved_tables/](saved_tables/).

---

## 1. High-level overview

```
            ┌─────────────────────────────────────┐
            │   PDB hetero-dimer mining (RCSB)    │
            │   + in-house pre-collected (5,037)  │
            └──────────────────┬──────────────────┘
                               │ >20k hetero-dimers
                               ▼
            ┌─────────────────────────────────────┐
            │  Structural complex clustering      │
            │  (Foldseek easy-multimercluster,    │
            │   cov=0.8, TM-score=0.8)            │
            └──────────────────┬──────────────────┘
                               │  Dimer set (cluster representatives)
                               ▼
            ┌─────────────────────────────────────┐
            │  Multi-chain affinity complexes     │
            │  PPB-Affinity (ATLAS, SAbDab,       │
            │   PDBbind, SKEMPI)                  │
            │   + new SAbDab + ProAffinity-GNN    │
            └──────────────────┬──────────────────┘
                               │  Affinity set
                               ▼
            ┌─────────────────────────────────────┐
            │  Merge & de-duplicate complexes     │
            │  → SSL set                          │
            └──────────────────┬──────────────────┘
                               │
        ┌──────────────────────┼──────────────────────┐
        ▼                      ▼                      ▼
   ΔG task              ΔΔG task            Binder discrimination
   (affinity set,       (PPB-Affinity        (SSL binders + swap
    S90/S79 test)        mutations,           negatives + PrePPI
                         SKEMPI test)         negatives + interface
                                              mutagenesis + AbEpiTope)
```

Test complexes for every task are **excluded from the SSL set** so that no
test sample is ever seen during pre-training or fine-tuning.

> **Structure source for non-experimental complexes.** Structures used
> in the **mutation set**, the **swap** and **PrePPI** binder-
> discrimination negatives, and **AbEpiTope** are predicted with
> **AlphaFold2-multimer**. The **interface-mutagenesis** negatives are
> *not* re-folded — they are produced by editing the graph of the
> corresponding wild-type complex (see §7.3). Experimental PDBs are
> used directly only for the dimer set, the WT entries of the affinity
> set, and the WT reference of SKEMPI mutations.

---

## 2. Building the dimer set

### 2.1 Mining hetero-dimers from RCSB PDB

Script: [download_dimer_pdbs.py](download_dimer_pdbs.py)

The RCSB PDB Search API is queried with the following criteria:

| Filter | Value |
|---|---|
| `rcsb_struct_symmetry.oligomeric_state` | `Hetero 2-mer` |
| `exptl.method` | `X-RAY DIFFRACTION`, `ELECTRON MICROSCOPY`, or `SOLUTION NMR` |
| `rcsb_entry_info.resolution_combined` | ≤ 4.0 Å (X-ray / cryo-EM; NMR has no resolution filter) |
| `entity_poly.rcsb_entity_polymer_type` | `Protein` |
| `rcsb_entry_info.deposited_polymer_entity_instance_count` | ≥ 2 |

Hits are paginated 10,000 per request. For every hit:

1. The `.pdb` file is downloaded; if RCSB no longer serves a legacy PDB
   format (large structures), the `.cif` is downloaded and converted via
   Biopython's `MMCIFParser` → `PDBIO`. A `ProteinSelect` filter keeps
   only standard amino-acid residues (and water), avoiding HETATMs with
   >3-character residue names that would corrupt the fixed-width PDB
   columns.
2. **Protein chains** are identified as chains containing at least one
   `CA` atom.
3. **Dimer chain assignment.** A `Hetero 2-mer` entry can still
   contain more than two protein chains in the deposited coordinates,
   because crystallographic asymmetric units often include several
   copies of the same biological dimer arranged by crystal-packing
   symmetry. To single out the biological dimer, the helper
   `determine_dimer_chains()` in [download_dimer_pdbs.py](download_dimer_pdbs.py)
   tries three strategies in order:

   1. **Exactly 2 protein chains** — both kept directly
     (`method = "exact_2_chains"`).
   2. **RCSB biological-assembly REST endpoint** —
     `get_assembly_chain_info()` queries
     `https://data.rcsb.org/rest/v1/core/assembly/{pdb}/1` and parses
     `pdbx_struct_assembly_gen[*].asym_id_list`. If exactly two of the
     returned chain IDs match the protein chains observed in the
     downloaded structure, those two are kept
     (`method = "assembly_api"`). This uses the curator-annotated
     biological assembly 1, which is the most reliable source of the
     true biological unit.
   3. **`REMARK 350 BIOMT` fallback** — when the REST endpoint is
     unavailable, returns more than two chains, or returns chain IDs
     that do not match the deposited protein chains,
     `parse_remark350_chains()` reads the `BIOMOLECULE: 1` block of
     the downloaded PDB file and collects the chain IDs listed under
     `APPLY THE FOLLOWING TO CHAINS:` / `AND CHAINS:`. If this yields
     exactly two chains, they are kept (`method = "remark350"`). This
     covers entries where the REST call fails (network issues, missing
     fields) but the biological-assembly information is still encoded
     in the PDB header.
   4. **First-two-protein-chains fallback** — if neither source yields
     exactly two matching chains, the first two protein chains in the
     structure are kept (`method = "first_two_fallback"`). PDB files
     conventionally list the biological-unit chains first, so this is
     a reasonable last resort, but entries tagged with this method are
     subsequently re-validated in §2.2 (sequence-identity and
     residue-contact checks) and any failures are written to
     `questioning_pdbs.csv` for manual inspection.

   The selection method is recorded per entry so that downstream code
   can audit how each dimer was resolved.

The metadata is written to
[saved_tables/new_dimer_db_table.csv](saved_tables/) with columns
`PDB, all_chains, chain_1, chain_2, homo_hetero`.

### 2.2 Validating the dimer chain assignment

Script: [check_dimer_chains.py](check_dimer_chains.py)

For every candidate dimer the script enforces:

- **Hetero check** – pairwise sequence identity between `chain_1` and
  `chain_2` must be `< 0.95`.
- **Interaction check** – the two chains must form **≥ 5 inter-chain
  residue contacts**, where a contact is at least one heavy-atom pair
  within **6.5 Å** (KD-tree query, `scipy.spatial.cKDTree`).

For entries with more than two protein chains all pairs are scored and
the **most-interacting hetero pair** is kept. Failing entries are
written to `questioning_pdbs.csv` for manual inspection.

### 2.3 Combining with the in-house pre-collected dimers

The validated table is merged with an in-house dataset of 5,037
hetero-dimers in [get_dimer_db.ipynb](get_dimer_db.ipynb); when a PDB ID
appears in both, the freshly downloaded entry is preferred. The merged
set contains >20,000 hetero-dimer complexes.

### 2.4 Structural complex clustering (Foldseek)

Script: [cluster_dimers_foldseek.py](cluster_dimers_foldseek.py)

Redundancy reduction is performed directly at the **complex level**
with `foldseek easy-multimercluster`, so that the dimer geometry — not
just per-chain folds — drives the clustering:

| Parameter | Value |
|---|---|
| `-c` (coverage) | **0.80** |
| `--cov-mode` | 0 |
| `--multimer-tm-threshold` | **0.80** |
| `--exhaustive-search` | enabled (keeps small proteins) |

For every cluster the **medoid** (member with minimum average TM-distance
to the rest) is chosen as the cluster representative. The
representatives form the final **dimer set** and are stored in
`nr_dimer_table_foldseek_cov08_tms08_exh.csv`.

Pairwise structure-similarity matrices used for medoid selection are
produced by [compute_rmsd_tmscores.py](compute_rmsd_tmscores.py) using
USalign (`-chain1 chain_A,chain_B -chain2 chain_A,chain_B`), saved as
`wt_tmscore_matrix.npy` / `wt_rmsd_matrix.npy`.

---

## 3. Building the affinity set

The affinity set extends the dimer set with multi-chain complexes that
have measured binding affinities.

### 3.1 PPB-Affinity as the base

We start from the published **PPB-Affinity** database, which itself
unifies **ATLAS**, **SAbDab**, **PDBbind**, and **SKEMPI**.

- Split into a wild-type subset (`Mutations.isna()`) and a mutant subset.
- Standardize affinity units: convert `KD` (mM/µM/nM/pM/fM → M) to ΔG
  via $\Delta G = -RT\ln K_D$. If a temperature is missing, the room
  temperature default **T = 298.15 K** is used. Where `delta_g`
  (kcal/mol) is already reported, it is used as is.

### 3.2 Adding fresh SAbDab entries

Notebook: [update_new_pdb_samples.ipynb](update_new_pdb_samples.ipynb)

Each of the four source databases (ATLAS, SAbDab, PDBbind, SKEMPI) is
re-checked for new releases. **SAbDab** is currently the only one with
substantial updates, so we add the new antibody–antigen entries that:

- have a protein or peptide antigen (carbohydrate / nucleic-acid /
  hapten / unknown antigens are rejected),
- have a non-null `affinity` or `delta_g`,
- are not already present in the affinity set.

Where a PDB already exists in PPB-Affinity, the new and old affinity
values are correlated; conflicting values are reconciled by averaging.

### 3.3 Importing missing complexes from ProAffinity-GNN

The affinity set is then compared against the dataset used in
**ProAffinity-GNN**; any complex present there but missing from ours is
added.

The resulting table is the **affinity set**.

---

## 4. SSL set

The SSL set is the union of the **dimer set** and the **affinity set**
with overlapping complexes removed (overlap is checked on PDB ID and
chain assignment). All test complexes from every downstream task — see
§7 — are removed from the SSL set so that the encoder never sees a test
complex during pre-training.

---

## 5. ΔG task dataset

The ΔG fine-tuning task reuses the **affinity set** for training/CV and
the standard public benchmarks **S90** and **S79** as test sets. The
two test sets are excluded from the affinity set used in training (see
[split_datasets_for_train_test.ipynb](split_datasets_for_train_test.ipynb)
and [get_test_dataset_for_all_models.ipynb](get_test_dataset_for_all_models.ipynb)).

---

## 6. ΔΔG task dataset

The ΔΔG task uses the **mutation set** from PPB-Affinity (mutant rows
with `Mutations` such as `A_L36K, B_R42E`) and the public **SKEMPI
test** as the held-out set. Mutant complex structures are predicted
with **AlphaFold2-multimer** rather than rotamer-only modeling, so the
backbone is allowed to relax around the mutation.

The associated wild-type PDBs of all SKEMPI test mutations are
themselves removed from training and from the SSL set.

---

## 7. Binder-discrimination dataset

For the binary binder/non-binder task we need positive **and** negative
complexes. Positives are the SSL binders. Negatives come from three
complementary sources, each predicted with **AlphaFold2-multimer**.

### 7.1 Negative source 1 — swap negatives

Notebook: [get_swap_data.ipynb](get_swap_data.ipynb)

Starting from the affinity set, ligand and receptor chains are swapped
between complexes to create mismatched pairs. To avoid trivial near-
duplicate negatives, the WT complexes are first clustered:

- USalign all-vs-all TM-score / RMSD matrices.
- Combined distance $d = (1 - \mathrm{TM}) + \beta \cdot \mathrm{RMSD}_{\text{norm}}$.
- Hierarchical clustering with average linkage at threshold **0.65**.
- The cluster medoid is chosen as the representative; one swap is
  generated per cluster.

The output is `swapped_table.csv` (~3,000 swapped complexes).

### 7.2 Negative source 2 — PrePPI low-score domain pairs

Notebooks/scripts:
[get_preppi_negative.ipynb](get_preppi_negative.ipynb) and
[IXN2Pdb.py](IXN2Pdb.py)

We use **PrePPI-AF (human)** to harvest domain–domain pairs predicted
to be **non-binders** (low PrePPI score):

1. From the PrePPI IXN table, keep pairs with PrePPI score in
   $[5, 10]$ (low-confidence predictions).
2. Cross-reference PrePPI's experimental lookup (`expDB`). A candidate
   negative pair is **discarded if the two parent proteins are known to
   bind in any experiment**, even when the specific domain pair has no
   experimental evidence.
3. For each remaining pair, the two domains are aligned by USalign onto
   the PrePPI template structure (`IXN2Pdb.py`) to build a candidate 3D
   complex.
4. To avoid trivial negatives, the candidate complex must satisfy:

   | Filter | Value |
   |---|---|
   | Min inter-chain residue contacts (heavy atoms < 6.5 Å) | **≥ 5** |
   | Max clashing residue pairs (heavy atoms < 2 Å) | **≤ 5** |
   | Max total residues in complex | 1,600 |
   | Max residues per chain | 1,300 |

   This rejects pairs that either fail to form any interface or are
   simply sterically incompatible.

Up to ~15,000 PrePPI negatives are produced (`preppi_db.csv`, with
`affinity = -1`).

### 7.3 Negative source 3 — interface mutagenesis negatives

Unlike the swap and PrePPI negatives, the interface-mutagenesis
negatives are generated **directly at the graph level** rather than as
new PDB files. For each real (wild-type) complex we reuse the
heterogeneous graph already built from its experimental structure and
apply the following procedure:

1. Identify the binding-interface residues on each of the two
   partners (the residues participating in inter-chain edges).
2. Randomly select **5 interface residues on each partner** (10 per
   complex) and mutate each selected residue to a different amino acid
   sampled uniformly from the remaining 19 types.
3. **Update only the node features** of the mutated residues so that
   the amino-acid identity (one-hot) and physicochemical features
   reflect the new residue type. The graph topology, the edge index,
   and the edge attributes (distance histograms and direction
   vectors) are left unchanged, since no new 3D structure is
   predicted.

The rationale is to mimic aggressive interface mutagenesis at the
level the model actually consumes: the encoder sees a complex whose
interface chemistry has been heavily perturbed while the backbone
geometry it was trained on remains intact. This avoids the
computational cost of re-folding every mutated complex with
AlphaFold2-multimer, and avoids contaminating the negatives with
structure-prediction artefacts.

The mutated graphs are produced on-the-fly in the data pipeline (no
separate PDB files are written); see the random-mutation logic in
[../utils/gen_graphs_unified.py](../utils/gen_graphs_unified.py)
(`if_random_mut=True`, `num_mut_on_each=5`).

### 7.4 Public test set — AbEpiTope

Notebook: [get_AbEpiTope_samples.ipynb](get_AbEpiTope_samples.ipynb)

**AbEpiTope** is taken from public literature as an additional
held-out benchmark for binder discrimination. It contains **272 swapped
antibody–antigen complexes built from 17 Ab–Ag complexes**, all
predicted with AlphaFold2-multimer.

### 7.5 Binder-discrimination train/test split and balancing

The data loader
[utils/Training_modules/classifier_data_loader.py](../utils/Training_modules/classifier_data_loader.py)
(`load_classifier_data`) balances positives and negatives **per split**
so that the classifier sees an equal number of binders and non-binders
in both train and test.

**Positives.** The positive set is split following the SSL train/val
split (`ssl_train_val_split.json`). Let `n_pos_train` and `n_pos_test`
denote the resulting positive counts.

**Training negatives — target total = `n_pos_train`.**

| Source | Quota for train |
|---|---|
| Swap (Source 1) | `n_pos_train // 3`, randomly sampled |
| Interface mutagenesis (Source 3) | `n_pos_train // 3`, randomly sampled from the **parent-WT-in-SSL-train** pool only — mutants whose wild-type parent lies in the SSL train split, to prevent SSL↔fine-tuning leakage |
| PrePPI (Source 2) | Fills the gap: `n_pos_train − swap_train − rand_mut_train` |

**Test negatives — target total = `n_pos_test`.**

| Source | Quota for test |
|---|---|
| AbEpiTope (`swapped_abag_sthg`) | **All** entries go to test |
| Interface mutagenesis | `(n_pos_test − |AbEpiTope|) // 2`, drawn from the test-eligible pool (parent WT in SSL test split) |
| PrePPI | The other half of `(n_pos_test − |AbEpiTope|)`, plus any random-mutant shortfall and any odd remainder |

Leftover swap samples are **not** carried into the test split; only
the train portion of Source 1 is used. Parent-aware splitting of the
interface-mutagenesis negatives ensures that no mutant whose wild-type
parent appeared in SSL pre-training train ever leaks into the
discriminator test split (and vice versa).

In short, Sources 1 and 3 contribute fixed fractions (1/3 of
`n_pos_train` each for training), AbEpiTope contributes its full set
to testing, and Source 2 (PrePPI), being the largest and most
plentiful negative pool, is used as the **filler** that closes the
balancing gap in both splits. The corresponding complexes are
removed from training in
[split_datasets_for_train_test.ipynb](split_datasets_for_train_test.ipynb)
and [get_test_dataset_for_all_models.ipynb](get_test_dataset_for_all_models.ipynb).

---

## 8. Test set summary

| Task | Training source | Test set |
|---|---|---|
| SSL | SSL set (dimer set ∪ affinity set, minus all test complexes) | 20 % held-out of SSL set |
| ΔG | Affinity set | **S90**, **S79** |
| ΔΔG | PPB-Affinity mutations | **SKEMPI test** |
| Binder discrimination | SSL positives + balanced negatives from swap, interface-mutagenesis, PrePPI (see §7.5) | SSL test positives + AbEpiTope (all) + interface-mut. + PrePPI, balanced 1:1 with positives |

All test complexes — for every task — are excluded from the SSL set, so
the encoder never sees them during pre-training or fine-tuning.

---

## 9. Key thresholds at a glance

| Stage | Parameter | Value |
|---|---|---|
| RCSB query | resolution (X-ray / EM) | ≤ 4.0 Å |
| RCSB query | oligomeric state | Hetero 2-mer |
| Chain validation | sequence identity (hetero) | < 0.95 |
| Chain validation | min inter-chain residue contacts | ≥ 5 (heavy atoms < 6.5 Å) |
| Foldseek (complex) | coverage / TM-score | **0.80 / 0.80** |
| Affinity ΔG default temperature | T | 298.15 K |
| Swap clustering | hierarchical-clustering distance threshold | 0.65 |
| PrePPI negatives | PrePPI score | 5 – 10 |
| PrePPI negatives | min contacts / max clashes | 5 / 5 |
| PrePPI negatives | max complex / chain size | 1,600 / 1,300 residues |
| Interface mutagenesis | random mutations per partner | 5 |

---

## 10. Manuscript-style description

The paragraphs below are drafted in a tone suitable for the
"Materials and Methods / Data" section of a manuscript. They can be
lifted as is or edited to match the journal voice.

### 10.1 Construction of the hetero-dimer set

To assemble a structurally diverse pool of protein–protein complexes,
we first mined hetero-dimeric structures from the RCSB Protein Data
Bank (PDB). The RCSB Search API was queried for entries whose
oligomeric state was annotated as `Hetero 2-mer`, whose polymer type
was protein, and which were determined either by X-ray diffraction or
cryo-electron microscopy at a resolution of 4.0 Å or better, or by
solution NMR. For each hit, the experimental structure was downloaded
in PDB format, falling back to mmCIF (and converting to PDB with
Biopython) when the legacy PDB format was not available. Only standard
amino-acid residues were retained, eliminating heteroatom ligands
whose residue names exceed three characters and would otherwise
corrupt the fixed-width PDB columns. Protein chains were identified as
chains containing at least one Cα atom; when more than two protein
chains were present (i.e., the asymmetric unit contained several
symmetry-related copies of the same biological dimer), the biological
dimer was resolved by a three-stage fallback. The RCSB
biological-assembly REST endpoint
(`/core/assembly/{pdb}/1`) was queried first, and the curator-
annotated assembly was accepted when it returned exactly two chains
matching the deposited protein chains; otherwise the `REMARK 350
BIOMT` records of the downloaded PDB file were parsed as a fallback;
and if neither source produced an unambiguous pair, the first two
protein chains in the deposited coordinates were retained as a last
resort, with the resulting dimer subsequently re-validated by the
sequence-identity and residue-contact checks described below.

Because the RCSB annotation of "Hetero 2-mer" does not guarantee a
biologically interacting pair, every candidate dimer was further
validated by two criteria. First, pairwise sequence identity between
the two assigned chains was required to be below 0.95 to ensure that
the pair was genuinely hetero. Second, the two chains were required to
form at least five inter-chain residue contacts, where a contact was
defined as any heavy-atom pair within 6.5 Å (computed efficiently with
a KD-tree). For structures with more than two protein chains, the
chain pair maximizing the number of contacts while satisfying the
hetero criterion was selected. Candidates failing either filter were
flagged for manual inspection rather than silently kept. The mined and
validated dimers were then merged with an in-house pre-collected
dataset of 5,037 hetero-dimers, yielding more than 20,000 candidate
hetero-dimeric complexes.

To remove redundancy, the candidates were clustered directly at the
complex level with Foldseek's `easy-multimercluster`, using a
coverage threshold of 0.80 and a multimer TM-score threshold of 0.80
(exhaustive search enabled to retain small proteins). Clustering at
the complex level — rather than at the level of individual chains or
on sequence — ensured that both partners and their relative
orientation contributed to the similarity measure, which we found
more appropriate for protein–protein interaction data than
sequence-only clustering. For each Foldseek cluster, the medoid (the
member minimizing the average TM-distance to the rest of the cluster)
was retained as the cluster representative. Pairwise TM-score and RMSD
matrices used both for medoid selection and for downstream analyses
were precomputed with USalign. The resulting set of cluster
representatives constitutes the **dimer set**.

### 10.2 Construction of the affinity set

To augment the dimer set with complexes carrying measured binding
affinities — and in particular with complexes containing more than two
chains — we built on the recently published PPB-Affinity database,
which unifies entries from ATLAS, SAbDab, PDBbind, and SKEMPI. We
parsed PPB-Affinity into a wild-type subset and a mutant subset, and
standardized all affinity measurements to ΔG (kcal/mol). When ΔG was
not provided directly, KD was converted via $\Delta G = -RT\ln K_D$
using the reported temperature, or 298.15 K when temperature was
missing. We then re-checked each of the four upstream databases for
recent releases; of these, SAbDab provided substantial new entries.
New SAbDab entries were added when the antigen was a protein or
peptide and a valid affinity value was available; for complexes that
overlapped with PPB-Affinity, conflicting affinity values were
reconciled by averaging. Finally, we compared the resulting set to the
dataset used in ProAffinity-GNN and incorporated any complexes present
there but absent from our collection. The final table constitutes the
**affinity set**.

### 10.3 SSL set

The self-supervised pre-training set, hereafter the **SSL set**, was
defined as the union of the dimer set and the affinity set after
removing complexes that appear in both. Crucially, every complex
belonging to any downstream test set — the SSL held-out test split
(20 % of the SSL set), the S90 and S79 ΔG benchmarks, the SKEMPI
mutational test set, and the binder-discrimination test set (see
below) — was removed from the SSL set prior to pre-training. This
guarantees that the encoder never observes a test complex, in either
pre-training or fine-tuning.

### 10.4 ΔG and ΔΔG fine-tuning datasets

For the binding-affinity (ΔG) regression task, the affinity set was
used for training and cross-validation, while the standard public
benchmarks S90 and S79 were held out as independent test sets. For
the mutational binding-affinity (ΔΔG) task, the mutation subset of
PPB-Affinity served as the training set, and the SKEMPI test split
was used as the held-out benchmark. All mutant complex structures
were generated with FoldX.

### 10.5 Binder-discrimination dataset

The binder-discrimination task requires both positive examples
(genuine binders) and negative examples (non-binding complexes).
Positives are taken from the SSL set. Because no large-scale curated
dataset of negative protein–protein complexes is available, we
constructed negatives from three complementary sources.

*Source 1 — swap negatives.* Starting from the affinity set, we
generated mismatched complexes by swapping ligand and receptor
between different complexes. To avoid trivially similar negatives,
the wild-type complexes were first clustered with hierarchical
clustering using a combined TM-score/RMSD distance, average linkage,
and a distance threshold of 0.65. One swap was generated per cluster,
yielding approximately 3,000 swapped complexes.

*Source 2 — PrePPI low-score domain pairs.* We mined the human
PrePPI-AF database for domain–domain pairs assigned a low PrePPI score
(score between 5 and 10), which the PrePPI model itself deems unlikely
to bind. To guard against false negatives, any pair was discarded
whenever the two parent proteins were experimentally reported to
interact in any form — even when the specific domain pair had no
experimental evidence. The retained domain pairs were then assembled
into 3D complexes by USalign-based alignment to the PrePPI template
structures. To exclude trivially non-binding complexes, the assembled
complex was required to display at least five inter-chain residue
contacts (heavy-atom distance < 6.5 Å) and no more than five clashing
residue pairs (heavy-atom distance < 2 Å); chain and complex sizes
were further capped at 1,300 and 1,600 residues respectively to keep
the inputs computationally tractable. This procedure yielded up to
~15,000 high-quality PrePPI-derived negatives.

*Source 3 — interface mutagenesis.* In contrast to the swap and
PrePPI negatives, the interface-mutagenesis negatives were not
realised as new three-dimensional structures. For each real complex,
we identified the binding-interface residues on each partner,
randomly selected five interface residues per partner (ten per
complex), and mutated each to a different amino acid sampled
uniformly from the remaining nineteen. The corresponding graph node
features — the amino-acid one-hot encoding and the physicochemical
descriptors — were updated accordingly, while the graph topology, the
edge index and the edge attributes (distance histograms and unit
direction vectors) of the wild-type complex were retained. This
graph-level perturbation mimics aggressive interface mutagenesis at
the level the encoder actually consumes, presenting it with a complex
whose interface chemistry has been heavily disrupted while the
underlying backbone geometry remains intact, and avoids the
computational cost and the structure-prediction artefacts that would
be incurred by re-folding every mutated complex.

*External benchmark — AbEpiTope.* As a stringent external test set
for binder discrimination, we used the publicly available AbEpiTope
collection, which contains 272 swapped antibody–antigen complexes
derived from 17 original Ab–Ag complexes and predicted with
AlphaFold2-multimer.

The final binder-discrimination dataset is balanced 1:1 between
positives and negatives in each split. Positives follow the SSL
train/val split. For training, negatives are assembled as one third
from Source 1 (swap) and one third from Source 3 (interface
mutagenesis), both randomly sampled to match `n_pos_train // 3`,
with the remaining third filled by Source 2 (PrePPI). Interface-
mutagenesis sampling is restricted to mutants whose wild-type parent
lies in the SSL training split, to prevent leakage between SSL
pre-training and discriminator fine-tuning. For testing, the entire
AbEpiTope set is assigned to the test split, the remaining slots are
split equally between interface mutagenesis (drawn from the
test-eligible parent pool) and PrePPI, and PrePPI also absorbs any
shortfall so that the total negative count equals `n_pos_test`. The
corresponding complexes were removed from the training pool. Across
all four tasks, no test complex appears anywhere in the SSL set or in
any training fold, ensuring strict isolation between training and
evaluation data.

---

## 11. Reproduction order

```
1.  download_dimer_pdbs.py        # mine RCSB hetero-dimers
2.  check_dimer_chains.py         # validate dimer chain assignment
3.  compute_rmsd_tmscores.py      # USalign matrices
4.  cluster_dimers_foldseek.py    # complex-level NR  → dimer set
5.  db_process.ipynb              # parse PPB-Affinity
6.  update_new_pdb_samples.ipynb  # add new SAbDab    → affinity set
7.  get_dimer_db.ipynb            # merge dimer + affinity → SSL set
8.  get_swap_data.ipynb           # swap negatives
9.  get_preppi_negative.ipynb + IXN2Pdb.py   # PrePPI negatives
10. get_mutant_stru.ipynb (+ multiprocess_mut.py)
                                  # interface-mutagenesis negatives
                                  # & PPB-Affinity mutation structures
11. get_AbEpiTope_samples.ipynb   # AbEpiTope test set
12. split_datasets_for_train_test.ipynb
    get_test_dataset_for_all_models.ipynb
                                  # carve out S90 / S79 / SKEMPI test
                                  # and binder-discrimination test
```
