"""
Classifier data loader for binary PPI classification fine-tuning.

Positive samples (label=1): from pos_sthg_8A (or pos_sthg_{embedding_type}_8A)
  - Train/test split follows the SSL pre-training split (ssl_train_val_split.json).

Negative samples (label=0): assembled from multiple sources:
  - swapped_sthg_8A: Chain-swapped decoys. 1/3 of n_pos_train randomly sampled to train;
    remaining go to test.
  - random_mut_sthg_8A: Random-mutated decoys (~3041 total). 1/3 of n_pos_train randomly
    sampled to train; remaining go to test.
  - swapped_abag_sthg_8A: Antibody-antigen swapped decoys. ALL go to test.
  - preppi_sthg_8A: PrePPI predicted non-interactors. Fills the remainder so that total
    negatives equal total positives in train and test separately. If swapped or random mutants
    cannot fill their 1/3 quota, PrePPI compensates the shortfall in train; likewise for test.

All splits use set_seed for reproducibility.
"""

import os
import json
import random
import pickle
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

from tqdm import tqdm

from .common_utils import (
    set_seed,
    load_sttgs_from_dir,
    filter_samples_finetune,
)


# ============================================================================
# Path helpers
# ============================================================================
def _get_pos_dir(pdb_root: str, dist: str, embedding_type: str) -> str:
    """Return path to the positive-sample folder."""
    if embedding_type in ['base', '+esm', '+esm480', 'only_esm', 'only_esm480']:
        return os.path.join(pdb_root, f'pos_sthg_{dist}A')
    else:
        return os.path.join(pdb_root, f'pos_sthg_{embedding_type}_{dist}A')


def _get_neg_dir(pdb_root: str, folder_prefix: str, dist: str, embedding_type: str) -> str:
    """
    Return path to a negative-sample folder.
    folder_prefix: one of 'capri_sthg', 'swapped_sthg', 'swapped_abag_sthg', 'preppi_sthg'
    """
    if embedding_type in ['base', '+esm', '+esm480', 'only_esm', 'only_esm480']:
        return os.path.join(pdb_root, f'{folder_prefix}_{dist}A')
    else:
        return os.path.join(pdb_root, f'{folder_prefix}_{embedding_type}_{dist}A')


# ============================================================================
# File listing helpers (do NOT pickle-load; just list names)
# ============================================================================
def _list_pkl_names(directory: str) -> List[str]:
    """Return sorted list of sample names (without .pkl) in *directory*."""
    if not os.path.isdir(directory):
        print(f"  [WARN] Directory not found: {directory}")
        return []
    names = sorted(
        f.replace('.pkl', '').lower()
        for f in os.listdir(directory) if f.endswith('.pkl')
    )
    return names


# ============================================================================
# CAPRI helpers (kept for future use; not used in the current training pipeline)
# ============================================================================
def _extract_capri_pdb_id(name: str) -> str:
    """
    Extract the PDB id from a CAPRI decoy name.
    Format: xxxx_n_m  →  pdb_id = xxxx
    Or just: xxxx      →  pdb_id = xxxx  (native structure, no underscores)
    """
    parts = name.split('_')
    return parts[0]


def _subsample_capri(names: List[str], ratio: int, seed: int) -> Dict[str, List[str]]:
    """
    For each PDB keep every `ratio`-th decoy (1/ratio) after shuffling within that PDB.

    Returns:
        by_pdb: dict mapping pdb_id → list of kept decoy names.
    """
    set_seed(seed)
    by_pdb: Dict[str, List[str]] = defaultdict(list)
    for n in names:
        by_pdb[_extract_capri_pdb_id(n)].append(n)

    kept_by_pdb: Dict[str, List[str]] = {}
    for pdb_id in sorted(by_pdb):
        decoys = sorted(by_pdb[pdb_id])
        random.shuffle(decoys)
        n_keep = max(1, len(decoys) // ratio)
        kept_by_pdb[pdb_id] = decoys[:n_keep]
    return kept_by_pdb


def _split_capri(
    by_pdb: Dict[str, List[str]],
    train_ratio: float,
    seed: int,
) -> Tuple[List[str], List[str]]:
    """
    Split subsampled CAPRI decoys by PDB id.

    PDB ids are shuffled, then the first `train_ratio` fraction of PDB ids
    go to train and the rest to test.  All decoys belonging to a PDB stay
    in the same split.
    """
    set_seed(seed)
    pdb_ids = sorted(by_pdb.keys())
    random.shuffle(pdb_ids)
    split_idx = int(len(pdb_ids) * train_ratio)

    train_names: List[str] = []
    test_names: List[str] = []
    for pdb_id in pdb_ids[:split_idx]:
        train_names.extend(by_pdb[pdb_id])
    for pdb_id in pdb_ids[split_idx:]:
        test_names.extend(by_pdb[pdb_id])
    return train_names, test_names


# ============================================================================
# Swapped split helper (kept for future use; not used in the current pipeline)
# ============================================================================
def _extract_swapped_pdb_ids(name: str) -> Tuple[str, str]:
    """
    Extract both PDB ids from a swapped decoy name.
    Format: xxxx_l_yyyy_r  or  xxxx_r_yyyy_l
    Returns (first_pdb, second_pdb).
    """
    parts = name.split('_')
    # parts: [xxxx, l/r, yyyy, r/l]
    return parts[0], parts[2]


def _split_swapped_by_pdb(names: List[str], train_ratio: float, seed: int) -> Tuple[List[str], List[str]]:
    """
    Split swapped decoys by the *first* PDB id.
    All samples sharing the same first PDB go to the same split.
    """
    set_seed(seed)
    by_pdb: Dict[str, List[str]] = defaultdict(list)
    for n in names:
        first_pdb, _ = _extract_swapped_pdb_ids(n)
        by_pdb[first_pdb].append(n)

    pdb_ids = sorted(by_pdb.keys())
    random.shuffle(pdb_ids)
    split_idx = int(len(pdb_ids) * train_ratio)

    train_names: List[str] = []
    test_names: List[str] = []
    for pdb_id in pdb_ids[:split_idx]:
        train_names.extend(by_pdb[pdb_id])
    for pdb_id in pdb_ids[split_idx:]:
        test_names.extend(by_pdb[pdb_id])

    return train_names, test_names


# ============================================================================
# Random-mutant naming helper
# ============================================================================
def _extract_rand_mut_pdb_id(name: str) -> Optional[str]:
    """
    Extract the parent PDB id from a random-mutant sample name.
    Expected format: xxx_mut  →  pdb_id = xxx
    Returns None if the name does not match the expected format.
    """
    if name.endswith('_mut'):
        return name[:-4]   # strip trailing '_mut'
    return None


# ============================================================================
# Train/test random sampling helper
# ============================================================================
def _sample_for_train_test(names: List[str], n_train: int, seed: int) -> Tuple[List[str], List[str]]:
    """
    Randomly select up to *n_train* samples for training; the remainder go to test.
    If len(names) <= n_train, all samples go to train and test receives none.
    """
    set_seed(seed)
    available = list(names)
    random.shuffle(available)
    n_take = min(n_train, len(available))
    return available[:n_take], available[n_take:]


# ============================================================================
# PrePPI fill helper
# ============================================================================
def _fill_preppi(
    preppi_names: List[str],
    n_train_needed: int,
    n_test_needed: int,
    seed: int,
) -> Tuple[List[str], List[str]]:
    """
    Randomly select samples from PrePPI to fill the gap between current
    negative count and the target (equal to positive count).

    If not enough PrePPI samples exist, use as many as available and print a
    warning.
    """
    set_seed(seed)
    total_needed = n_train_needed + n_test_needed
    available = list(preppi_names)
    random.shuffle(available)

    if total_needed > len(available):
        print(f"  [WARN] PrePPI has {len(available)} samples but need {total_needed}. "
              f"Using all available.")
        # Distribute proportionally
        if total_needed == 0:
            return [], []
        train_frac = n_train_needed / total_needed
        n_train_take = int(len(available) * train_frac)
        return available[:n_train_take], available[n_train_take:]
    
    train_preppi = available[:n_train_needed]
    test_preppi = available[n_train_needed: n_train_needed + n_test_needed]
    return train_preppi, test_preppi


# ============================================================================
# Main entry point
# ============================================================================
def load_classifier_data(
    data_cfg: Dict,
    ssl_save_dir: str,
    save_dir: Optional[str] = None,
    seed: int = 42,
    min_nodes: int = 5,
) -> Tuple[Dict, Dict, Dict, Dict]:
    """
    Load and split data for binary PPI classifier fine-tuning.

    Negative sampling strategy (no CAPRI):
      Train negatives = n_pos_train total, split evenly as 1/3 each from:
        - Swapped decoys  (randomly sampled; remainder → test)
        - Random mutants  (randomly sampled; remainder → test)
        - PrePPI          (fills any shortfall from the above two, plus their share)
      Test negatives = n_pos_test total, drawn from:
        - Remaining Swapped decoys
        - Remaining Random mutants
        - All Swapped Ab-Ag decoys
        - PrePPI fills any shortfall to reach n_pos_test

    Args:
        data_cfg: data section from YAML config (pdb_root, dist, embedding_type …)
        ssl_save_dir: directory that contains ssl_train_val_split.json
                      (same folder as the SSL pretrain checkpoint)
        save_dir: directory where classifier_split_info.json will be saved.
                  Defaults to ssl_save_dir if not provided.
        seed: random seed for all splits
        min_nodes: minimum nodes per chain for filtering

    Returns:
        train_sttgs:  dict  {name: setting_obj}  positive + negative for training
        test_sttgs:   dict  {name: setting_obj}  positive + negative for testing
        train_labels: dict  {name: int}  0 or 1
        test_labels:  dict  {name: int}  0 or 1

    Also saves split info to save_dir/classifier_split_info.json.
    """
    pdb_root = data_cfg['pdb_root']
    dist = data_cfg['dist']
    embedding_type = data_cfg['embedding_type']

    print(f"\n{'='*70}")
    print("CLASSIFIER DATA LOADING")
    print(f"{'='*70}")
    print(f"  pdb_root       : {pdb_root}")
    print(f"  dist           : {dist}A")
    print(f"  embedding_type : {embedding_type}")
    print(f"  seed           : {seed}")
    print(f"  ssl_save_dir   : {ssl_save_dir}")

    # ------------------------------------------------------------------
    # 1. Load SSL train/val split  (determines positive train/test names)
    # ------------------------------------------------------------------
    split_path = os.path.join(ssl_save_dir, 'ssl_train_val_split.json')
    if not os.path.exists(split_path):
        raise FileNotFoundError(
            f"ssl_train_val_split.json not found at {split_path}. "
            "Run SSL pre-training first or point ssl_save_dir to the correct folder."
        )
    with open(split_path, 'r') as f:
        ssl_split = json.load(f)

    pos_train_names_ssl = set(n.lower() for n in ssl_split['train_names'])
    pos_test_names_ssl = set(n.lower() for n in ssl_split['val_names'])
    print(f"\n  SSL split loaded: {len(pos_train_names_ssl)} train / "
          f"{len(pos_test_names_ssl)} test positive names")

    # ------------------------------------------------------------------
    # 2. Load positive samples
    # ------------------------------------------------------------------
    pos_dir = _get_pos_dir(pdb_root, dist, embedding_type)
    print(f"\n--- Positive samples ---")
    print(f"  Loading from: {pos_dir}")
    pos_sttgs_all = load_sttgs_from_dir(pos_dir)
    pos_sttgs_all = filter_samples_finetune(pos_sttgs_all, min_nodes=min_nodes)
    print(f"  After filtering: {len(pos_sttgs_all)} samples")

    # Split positives following SSL split
    pos_train_sttgs: Dict = {}
    pos_test_sttgs: Dict = {}
    pos_skipped = 0
    for name, st in pos_sttgs_all.items():
        if name in pos_train_names_ssl:
            pos_train_sttgs[name] = st
        elif name in pos_test_names_ssl:
            pos_test_sttgs[name] = st
        else:
            pos_skipped += 1

    n_pos_train = len(pos_train_sttgs)
    n_pos_test = len(pos_test_sttgs)
    print(f"  Positive train: {n_pos_train}, test: {n_pos_test}")
    if pos_skipped > 0:
        print(f"  (Skipped {pos_skipped} positives not in SSL split – "
              "likely filtered out during SSL preprocessing)")

    # 1/3 quota per source for train
    quota_per_source = n_pos_train // 3

    # ------------------------------------------------------------------
    # 3. Swapped decoys: randomly sample quota_per_source for train;
    #    remainder goes to test.
    # ------------------------------------------------------------------
    swap_dir = _get_neg_dir(pdb_root, 'swapped_sthg', dist, embedding_type)
    print(f"\n--- Swapped negatives ---")
    print(f"  Loading from: {swap_dir}")
    swap_all_names = _list_pkl_names(swap_dir)
    print(f"  Total swapped files: {len(swap_all_names)}")

    swap_train_names, swap_test_names = _sample_for_train_test(
        swap_all_names, n_train=quota_per_source, seed=seed
    )
    print(f"  Quota (1/3 of {n_pos_train}): {quota_per_source}")
    print(f"  Swapped → train: {len(swap_train_names)}, test: {len(swap_test_names)}")

    # ------------------------------------------------------------------
    # 4. Random mutants: randomly sample quota_per_source for train;
    #    remainder goes to test.
    # ------------------------------------------------------------------
    rand_mut_dir = _get_neg_dir(pdb_root, 'random_mut_sthg', dist, embedding_type)
    print(f"\n--- Random-mutant negatives ---")
    print(f"  Loading from: {rand_mut_dir}")
    rand_mut_all_names = _list_pkl_names(rand_mut_dir)
    print(f"  Total random-mutant files: {len(rand_mut_all_names)}")

    # Assign each random mutant to the same split as its parent wildtype (xxx_mut → parent xxx)
    # to prevent data leakage: if the encoder saw wildtype xxx during SSL pre-training in the
    # train split, its mutant xxx_mut must not appear in the test split.
    rand_mut_train_pool: List[str] = []
    rand_mut_test_pool: List[str] = []
    rand_mut_unmatched: List[str] = []
    for n in rand_mut_all_names:
        parent = _extract_rand_mut_pdb_id(n)
        if parent is None:
            print(f"  [WARN] Unexpected random-mutant name format (no '_mut' suffix): {n}")
            rand_mut_unmatched.append(n)
        elif parent in pos_train_names_ssl:
            rand_mut_train_pool.append(n)
        elif parent in pos_test_names_ssl:
            rand_mut_test_pool.append(n)
        else:
            rand_mut_unmatched.append(n)

    print(f"  Train-eligible (parent in train split): {len(rand_mut_train_pool)}")
    print(f"  Test-eligible  (parent in test split) : {len(rand_mut_test_pool)}")
    if rand_mut_unmatched:
        print(f"  Unmatched (parent not in SSL split)   : {len(rand_mut_unmatched)} — excluded")

    # Sample up to quota_per_source from the train-eligible pool for train;
    # all test-eligible samples go to test.
    rand_mut_train_names, _ = _sample_for_train_test(
        rand_mut_train_pool, n_train=quota_per_source, seed=seed + 1
    )
    rand_mut_test_names = list(rand_mut_test_pool)
    print(f"  Quota (1/3 of {n_pos_train}): {quota_per_source}")
    print(f"  Random mutants → train: {len(rand_mut_train_names)}, test: {len(rand_mut_test_names)}")

    # ------------------------------------------------------------------
    # 5. Swapped Ab-Ag → all to test
    # ------------------------------------------------------------------
    abag_dir = _get_neg_dir(pdb_root, 'swapped_abag_sthg', dist, embedding_type)
    print(f"\n--- Swapped Ab-Ag negatives (all → test) ---")
    print(f"  Loading from: {abag_dir}")
    abag_all_names = _list_pkl_names(abag_dir)
    print(f"  Total Ab-Ag files: {len(abag_all_names)}  (all assigned to test)")

    # ------------------------------------------------------------------
    # 6. Determine test-negative counts:
    #    abag covers its share first; the remaining slots are split equally
    #    between random mutants (test-eligible pool) and PrePPI.
    #    Any random-mutant shortfall falls to PrePPI.
    # ------------------------------------------------------------------
    remaining_test = max(0, n_pos_test - len(abag_all_names))
    each_fill_test = remaining_test // 2
    extra_fill_test = remaining_test - 2 * each_fill_test  # 0 or 1; assigned to preppi

    # Sample up to each_fill_test from the test-eligible random-mutant pool
    rand_mut_test_selected, _ = _sample_for_train_test(
        rand_mut_test_names, n_train=each_fill_test, seed=seed + 3
    )
    rand_mut_test_shortfall = each_fill_test - len(rand_mut_test_selected)

    print(f"\n--- Test-negative fill (abag + equal random mutants + PrePPI) ---")
    print(f"  n_pos_test={n_pos_test}, abag={len(abag_all_names)}, remaining={remaining_test}")
    print(f"  Per-source quota (random mutants / PrePPI): {each_fill_test}")
    print(f"  Random mutants selected for test: {len(rand_mut_test_selected)}"
          + (f" (shortfall {rand_mut_test_shortfall} → PrePPI compensates)" if rand_mut_test_shortfall else ""))

    # ------------------------------------------------------------------
    # 7. PrePPI: fill remaining quota in train and test
    # ------------------------------------------------------------------
    neg_train_so_far = len(swap_train_names) + len(rand_mut_train_names)
    preppi_train_needed = max(0, n_pos_train - neg_train_so_far)
    preppi_test_needed  = each_fill_test + extra_fill_test + rand_mut_test_shortfall

    print(f"\n--- PrePPI fill ---")
    print(f"  Target neg train = {n_pos_train}  (have {neg_train_so_far}, need {preppi_train_needed} more)")
    print(f"  PrePPI needed for test: {preppi_test_needed}")

    preppi_dir = _get_neg_dir(pdb_root, 'preppi_sthg', dist, embedding_type)
    print(f"  Loading from: {preppi_dir}")
    preppi_all_names = _list_pkl_names(preppi_dir)
    print(f"  Total PrePPI files: {len(preppi_all_names)}")

    preppi_train_names, preppi_test_names = _fill_preppi(
        preppi_all_names,
        n_train_needed=preppi_train_needed,
        n_test_needed=preppi_test_needed,
        seed=seed + 2,
    )
    print(f"  PrePPI → train: {len(preppi_train_names)}, test: {len(preppi_test_names)}")

    # ------------------------------------------------------------------
    # 8. Aggregate negative name lists
    # ------------------------------------------------------------------
    neg_train_names = swap_train_names + rand_mut_train_names + preppi_train_names
    neg_test_names  = abag_all_names + rand_mut_test_selected + preppi_test_names

    print(f"\n--- Aggregate ---")
    print(f"  Neg train total: {len(neg_train_names)}")
    print(f"  Neg test  total: {len(neg_test_names)}")

    # ------------------------------------------------------------------
    # 9. Pickle-load only the samples we actually need
    # ------------------------------------------------------------------
    # Build a mapping: name → (directory, source_tag)
    neg_name_to_dir: Dict[str, Tuple[str, str]] = {}
    for n in swap_train_names:  # swap_test is unused; train portion only
        neg_name_to_dir[n] = (swap_dir, 'swapped')
    for n in rand_mut_train_names + rand_mut_test_selected:
        neg_name_to_dir[n] = (rand_mut_dir, 'random_mut')
    for n in abag_all_names:
        neg_name_to_dir[n] = (abag_dir, 'swapped_abag')
    for n in preppi_train_names + preppi_test_names:
        neg_name_to_dir[n] = (preppi_dir, 'preppi')

    def _load_selected(names: List[str], label: int, source_tag: str = 'pos') -> Tuple[Dict, Dict]:
        """Load pkl files for given names; return (sttgs_dict, labels_dict)."""
        sttgs = {}
        labels = {}
        for name in tqdm(names, desc=f"Loading {source_tag} ({label})", leave=False):
            if source_tag == 'pos':
                # Already loaded in pos_sttgs_all
                if name in pos_sttgs_all:
                    sttgs[name] = pos_sttgs_all[name]
                    labels[name] = label
            else:
                directory, _ = neg_name_to_dir[name]
                pkl_path = os.path.join(directory, name + '.pkl')
                if not os.path.exists(pkl_path):
                    # Try original case
                    pkl_path_upper = os.path.join(directory, name.upper() + '.pkl')
                    if os.path.exists(pkl_path_upper):
                        pkl_path = pkl_path_upper
                    else:
                        continue
                try:
                    with open(pkl_path, 'rb') as fh:
                        st = pickle.load(fh)
                    # Strip any extra attributes that are absent in other graph types;
                    # PyG collate raises KeyError if keys differ across graphs in a batch.
                    if hasattr(st.protein_graph, 'mut_info'):
                        del st.protein_graph.mut_info
                    sttgs[name] = st
                    labels[name] = label
                except Exception as e:
                    print(f"  [WARN] Failed to load {pkl_path}: {e}")
        return sttgs, labels

    # Load positive samples (already in memory)
    print(f"\n--- Loading samples into memory ---")
    pos_train_loaded, pos_train_labels = _load_selected(
        list(pos_train_sttgs.keys()), label=1, source_tag='pos'
    )
    pos_test_loaded, pos_test_labels = _load_selected(
        list(pos_test_sttgs.keys()), label=1, source_tag='pos'
    )

    # Load negative samples (need to read from disk)
    neg_train_loaded, neg_train_labels = _load_selected(
        neg_train_names, label=0, source_tag='neg'
    )
    neg_test_loaded, neg_test_labels = _load_selected(
        neg_test_names, label=0, source_tag='neg'
    )
    # Note: swap_test_names are intentionally excluded from test to keep
    # test negatives composed only of abag + random mutants + PrePPI.

    # Filter negative samples the same way
    neg_train_loaded = filter_samples_finetune(neg_train_loaded, min_nodes=min_nodes)
    neg_test_loaded = filter_samples_finetune(neg_test_loaded, min_nodes=min_nodes)
    # Update labels to match filtered
    neg_train_labels = {k: v for k, v in neg_train_labels.items() if k in neg_train_loaded}
    neg_test_labels = {k: v for k, v in neg_test_labels.items() if k in neg_test_loaded}

    # ------------------------------------------------------------------
    # 9. Merge into final dicts
    # ------------------------------------------------------------------
    train_sttgs = {**pos_train_loaded, **neg_train_loaded}
    test_sttgs  = {**pos_test_loaded,  **neg_test_loaded}
    train_labels = {**pos_train_labels, **neg_train_labels}
    test_labels  = {**pos_test_labels,  **neg_test_labels}

    print(f"\n{'='*70}")
    print("CLASSIFIER DATA SUMMARY")
    print(f"{'='*70}")
    print(f"  Train: {len(train_sttgs)} total  "
          f"({sum(v==1 for v in train_labels.values())} pos, "
          f"{sum(v==0 for v in train_labels.values())} neg)")
    print(f"  Test : {len(test_sttgs)} total  "
          f"({sum(v==1 for v in test_labels.values())} pos, "
          f"{sum(v==0 for v in test_labels.values())} neg)")

    # ------------------------------------------------------------------
    # 10. Save split info
    # ------------------------------------------------------------------
    def _source_map(names):
        """name → source folder tag."""
        m = {}
        for n in names:
            if n in neg_name_to_dir:
                m[n] = neg_name_to_dir[n][1]
            else:
                m[n] = 'pos'
        return m

    split_info = {
        'seed': seed,
        'quota_per_source_train': quota_per_source,
        'each_fill_test': each_fill_test,
        # Counts
        'n_pos_train': sum(v == 1 for v in train_labels.values()),
        'n_neg_train': sum(v == 0 for v in train_labels.values()),
        'n_pos_test':  sum(v == 1 for v in test_labels.values()),
        'n_neg_test':  sum(v == 0 for v in test_labels.values()),
        # Negative source breakdown – train
        'neg_train_swapped':    len([n for n in neg_train_labels if neg_name_to_dir.get(n, ('',''))[1] == 'swapped']),
        'neg_train_random_mut': len([n for n in neg_train_labels if neg_name_to_dir.get(n, ('',''))[1] == 'random_mut']),
        'neg_train_preppi':     len([n for n in neg_train_labels if neg_name_to_dir.get(n, ('',''))[1] == 'preppi']),
        # Negative source breakdown – test (abag + equal random_mut + preppi; no swapped)
        'neg_test_abag':       len([n for n in neg_test_labels if neg_name_to_dir.get(n, ('',''))[1] == 'swapped_abag']),
        'neg_test_random_mut': len([n for n in neg_test_labels if neg_name_to_dir.get(n, ('',''))[1] == 'random_mut']),
        'neg_test_preppi':     len([n for n in neg_test_labels if neg_name_to_dir.get(n, ('',''))[1] == 'preppi']),
        # Name lists
        'train_pos_names': sorted([n for n, v in train_labels.items() if v == 1]),
        'test_pos_names':  sorted([n for n, v in test_labels.items() if v == 1]),
        'train_neg_names': sorted([n for n, v in train_labels.items() if v == 0]),
        'test_neg_names':  sorted([n for n, v in test_labels.items() if v == 0]),
        # Source mapping for negatives
        'train_neg_sources': _source_map([n for n, v in train_labels.items() if v == 0]),
        'test_neg_sources':  _source_map([n for n, v in test_labels.items() if v == 0]),
    }

    _split_save_dir = save_dir if save_dir is not None else ssl_save_dir
    os.makedirs(_split_save_dir, exist_ok=True)
    info_path = os.path.join(_split_save_dir, 'classifier_split_info.json')
    with open(info_path, 'w') as f:
        json.dump(split_info, f, indent=2)
    print(f"\n  Split info saved to: {info_path}")

    return train_sttgs, test_sttgs, train_labels, test_labels


# ============================================================================
# CV pool loader (n-fold cross-validation, no train/test split)
# ============================================================================
def load_classifier_data_cv(
    data_cfg: Dict,
    save_dir: Optional[str] = None,
    seed: int = 42,
    min_nodes: int = 5,
) -> Tuple[Dict, Dict]:
    """
    Load a single pool of samples for n-fold CV (no train/test split, no
    SSL split required).

    Composition (total negatives == total positives):
      Positives : ALL samples in pos_dir.
      Negatives :
        - All SWAP samples (swapped_sthg)
        - random_mut : (n_pos - n_swap) / 2 randomly sampled
        - PrePPI     : (n_pos - n_swap) / 2 randomly sampled
                       (fills any random_mut shortfall too)

    `swapped_abag` is intentionally NOT included (reserved for held-out test
    in the non-CV pipeline).

    Returns:
        pool_sttgs  : dict {name: setting_obj}
        pool_labels : dict {name: int}   1 = positive, 0 = negative

    Also saves split info to `save_dir/classifier_cv_pool_info.json`
    (or `ssl_save_dir` substitute if save_dir is None).
    """
    pdb_root = data_cfg['pdb_root']
    dist = data_cfg['dist']
    embedding_type = data_cfg['embedding_type']

    print(f"\n{'='*70}")
    print("CLASSIFIER CV POOL LOADING")
    print(f"{'='*70}")
    print(f"  pdb_root       : {pdb_root}")
    print(f"  dist           : {dist}A")
    print(f"  embedding_type : {embedding_type}")
    print(f"  seed           : {seed}")

    # ------------------------------------------------------------------
    # 1. Positives — load all
    # ------------------------------------------------------------------
    pos_dir = _get_pos_dir(pdb_root, dist, embedding_type)
    print(f"\n--- Positive samples ---")
    print(f"  Loading from: {pos_dir}")
    pos_sttgs_all = load_sttgs_from_dir(pos_dir)
    pos_sttgs_all = filter_samples_finetune(pos_sttgs_all, min_nodes=min_nodes)
    n_pos = len(pos_sttgs_all)
    print(f"  After filtering: {n_pos} positives")

    if n_pos == 0:
        raise ValueError(f"No positives loaded from {pos_dir}.")

    # ------------------------------------------------------------------
    # 2. SWAP — take all
    # ------------------------------------------------------------------
    swap_dir = _get_neg_dir(pdb_root, 'swapped_sthg', dist, embedding_type)
    print(f"\n--- SWAP negatives (all included) ---")
    print(f"  Loading from: {swap_dir}")
    swap_all_names = _list_pkl_names(swap_dir)
    n_swap = len(swap_all_names)
    print(f"  Total SWAP files: {n_swap}")

    # ------------------------------------------------------------------
    # 3. Compute half-quota for PrePPI and random_mut
    # ------------------------------------------------------------------
    remaining = max(0, n_pos - n_swap)
    each_half = remaining // 2
    extra = remaining - 2 * each_half   # 0 or 1, assigned to PrePPI

    print(f"\n  Target negatives ({n_pos}) − SWAP ({n_swap}) = remaining {remaining}")
    print(f"  random_mut quota: {each_half}")
    print(f"  PrePPI quota    : {each_half + extra}")

    # ------------------------------------------------------------------
    # 4. random_mut — sample up to each_half from full pool
    # ------------------------------------------------------------------
    rand_mut_dir = _get_neg_dir(pdb_root, 'random_mut_sthg', dist, embedding_type)
    print(f"\n--- random_mut negatives ---")
    print(f"  Loading from: {rand_mut_dir}")
    rand_mut_all_names = _list_pkl_names(rand_mut_dir)
    print(f"  Total random_mut files: {len(rand_mut_all_names)}")

    rand_mut_take, _ = _sample_for_train_test(
        rand_mut_all_names, n_train=each_half, seed=seed + 1
    )
    rand_mut_shortfall = each_half - len(rand_mut_take)
    print(f"  random_mut selected: {len(rand_mut_take)}"
          + (f" (shortfall {rand_mut_shortfall} → PrePPI compensates)" if rand_mut_shortfall else ""))

    # ------------------------------------------------------------------
    # 5. PrePPI — sample (each_half + extra + rand_mut_shortfall)
    # ------------------------------------------------------------------
    preppi_needed = each_half + extra + rand_mut_shortfall
    preppi_dir = _get_neg_dir(pdb_root, 'preppi_sthg', dist, embedding_type)
    print(f"\n--- PrePPI negatives ---")
    print(f"  Loading from: {preppi_dir}")
    preppi_all_names = _list_pkl_names(preppi_dir)
    print(f"  Total PrePPI files: {len(preppi_all_names)}")
    print(f"  PrePPI needed: {preppi_needed}")

    preppi_take, _ = _sample_for_train_test(
        preppi_all_names, n_train=preppi_needed, seed=seed + 2
    )
    print(f"  PrePPI selected: {len(preppi_take)}")

    # ------------------------------------------------------------------
    # 6. Build dir map and load selected negatives
    # ------------------------------------------------------------------
    neg_name_to_dir: Dict[str, Tuple[str, str]] = {}
    for n in swap_all_names:
        neg_name_to_dir[n] = (swap_dir, 'swapped')
    for n in rand_mut_take:
        neg_name_to_dir[n] = (rand_mut_dir, 'random_mut')
    for n in preppi_take:
        neg_name_to_dir[n] = (preppi_dir, 'preppi')

    neg_names = list(swap_all_names) + list(rand_mut_take) + list(preppi_take)

    print(f"\n--- Loading negatives into memory ---")
    neg_sttgs: Dict = {}
    neg_labels: Dict[str, int] = {}
    for name in tqdm(neg_names, desc="Loading negatives", leave=False):
        directory, _ = neg_name_to_dir[name]
        pkl_path = os.path.join(directory, name + '.pkl')
        if not os.path.exists(pkl_path):
            pkl_path_upper = os.path.join(directory, name.upper() + '.pkl')
            if os.path.exists(pkl_path_upper):
                pkl_path = pkl_path_upper
            else:
                continue
        try:
            with open(pkl_path, 'rb') as fh:
                st = pickle.load(fh)
            if hasattr(st.protein_graph, 'mut_info'):
                del st.protein_graph.mut_info
            neg_sttgs[name] = st
            neg_labels[name] = 0
        except Exception as e:
            print(f"  [WARN] Failed to load {pkl_path}: {e}")

    neg_sttgs = filter_samples_finetune(neg_sttgs, min_nodes=min_nodes)
    neg_labels = {k: v for k, v in neg_labels.items() if k in neg_sttgs}

    # ------------------------------------------------------------------
    # 7. Merge positives + negatives into a single pool
    # ------------------------------------------------------------------
    pos_labels = {name: 1 for name in pos_sttgs_all}
    pool_sttgs = {**pos_sttgs_all, **neg_sttgs}
    pool_labels = {**pos_labels, **neg_labels}

    n_pos_final = sum(v == 1 for v in pool_labels.values())
    n_neg_final = sum(v == 0 for v in pool_labels.values())

    print(f"\n{'='*70}")
    print("CV POOL SUMMARY")
    print(f"{'='*70}")
    print(f"  Pool total: {len(pool_sttgs)}  "
          f"({n_pos_final} pos, {n_neg_final} neg)")
    n_swap_kept   = sum(1 for n in neg_labels if neg_name_to_dir.get(n, ('',''))[1] == 'swapped')
    n_rmut_kept   = sum(1 for n in neg_labels if neg_name_to_dir.get(n, ('',''))[1] == 'random_mut')
    n_preppi_kept = sum(1 for n in neg_labels if neg_name_to_dir.get(n, ('',''))[1] == 'preppi')
    print(f"  Neg sources — SWAP: {n_swap_kept}, "
          f"random_mut: {n_rmut_kept}, PrePPI: {n_preppi_kept}")

    # ------------------------------------------------------------------
    # 8. Save pool info
    # ------------------------------------------------------------------
    if save_dir is not None:
        def _source_map(names):
            m = {}
            for n in names:
                m[n] = neg_name_to_dir[n][1] if n in neg_name_to_dir else 'pos'
            return m

        pool_info = {
            'seed': seed,
            'n_pos': n_pos_final,
            'n_neg': n_neg_final,
            'neg_swapped':    n_swap_kept,
            'neg_random_mut': n_rmut_kept,
            'neg_preppi':     n_preppi_kept,
            'pos_names': sorted([n for n, v in pool_labels.items() if v == 1]),
            'neg_names': sorted([n for n, v in pool_labels.items() if v == 0]),
            'neg_sources': _source_map([n for n, v in pool_labels.items() if v == 0]),
        }
        os.makedirs(save_dir, exist_ok=True)
        info_path = os.path.join(save_dir, 'classifier_cv_pool_info.json')
        with open(info_path, 'w') as f:
            json.dump(pool_info, f, indent=2)
        print(f"\n  CV pool info saved to: {info_path}")

    return pool_sttgs, pool_labels
