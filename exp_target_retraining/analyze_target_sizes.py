#!/usr/bin/env python3
"""Target-size similarity analysis for the experimental binder set.

Decides between the two cross-validation schemes agreed for the target-aware
retraining (see README.md, section 3):

  * targets similar in size  -> leave-2-targets-out 7-fold CV (grouped by target)
  * otherwise                -> sample-level 5-fold random CV on the non-test targets

The decision rule flags the targets as "similar in size" only when ALL of:
    max/min ratio <= 3 , coefficient of variation <= 0.5 , chi2-vs-uniform p > 0.05.

This script only reads the labeled rows of final_dataset.csv and prints descriptive
statistics. It does not touch the model, GPU, or any training code.
"""
from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from pathlib import Path

DEFAULT_CSV = (
    Path(__file__).resolve().parents[2]
    / "GraPPI_data" / "exp_binder_data" / "final_dataset.csv"
)

# Decision thresholds (README.md section 3).
MAX_MIN_RATIO_MAX = 3.0
CV_MAX = 0.5
UNIFORM_P_MIN = 0.05


def load_target_counts(csv_path: Path) -> Counter:
    """Return per-target labeled-sample counts (rows with a non-empty ``binder``)."""
    counts: Counter = Counter()
    with csv_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if str(row["binder"]).strip() == "":
                continue
            counts[row["target_id"]] += 1
    if not counts:
        raise ValueError(f"No labeled rows found in {csv_path}")
    return counts


def chi2_uniform_pvalue(counts: list[int]) -> tuple[float, int, float | None]:
    """Chi-square goodness-of-fit against a uniform distribution.

    Returns (chi2, dof, p). ``p`` is ``None`` when SciPy is unavailable.
    """
    total = sum(counts)
    n = len(counts)
    expected = total / n
    chi2 = sum((c - expected) ** 2 / expected for c in counts)
    dof = n - 1
    try:
        from scipy.stats import chi2 as chi2_dist

        p = float(chi2_dist.sf(chi2, dof))
    except Exception:
        p = None
    return chi2, dof, p


def gini(counts: list[int]) -> float:
    """Gini coefficient of the count distribution (0 = equal, ->1 = concentrated)."""
    s = sorted(counts)
    n = len(s)
    total = sum(s)
    weighted = sum(i * x for i, x in enumerate(s, start=1))
    return (2 * weighted) / (n * total) - (n + 1) / n


def summarize(counts: Counter) -> dict:
    values = list(counts.values())
    n = len(values)
    total = sum(values)
    mean = total / n
    variance = sum((c - mean) ** 2 for c in values) / n
    std = math.sqrt(variance)
    cv = std / mean if mean else float("inf")
    mx, mn = max(values), min(values)
    chi2, dof, p = chi2_uniform_pvalue(values)
    similar = (
        (mx / mn) <= MAX_MIN_RATIO_MAX
        and cv <= CV_MAX
        and p is not None
        and p > UNIFORM_P_MIN
    )
    return {
        "n_targets": n,
        "total": total,
        "min": mn,
        "max": mx,
        "max_min_ratio": mx / mn,
        "mean": mean,
        "std": std,
        "cv": cv,
        "gini": gini(values),
        "chi2": chi2,
        "dof": dof,
        "p_uniform": p,
        "similar": similar,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    args = parser.parse_args()

    counts = load_target_counts(args.csv)
    stats = summarize(counts)

    print(f"CSV: {args.csv}")
    print(f"Labeled targets: {stats['n_targets']}   total samples: {stats['total']}\n")

    print(f"{'target_id':16s} {'count':>6s}  {'% of total':>10s}")
    for target, count in counts.most_common():
        print(f"{target:16s} {count:6d}  {100 * count / stats['total']:9.1f}%")

    p_str = "n/a (SciPy missing)" if stats["p_uniform"] is None else f"{stats['p_uniform']:.3g}"
    print("\n--- size-similarity statistics ---")
    print(f"min / max                 : {stats['min']} / {stats['max']}")
    print(f"max/min ratio             : {stats['max_min_ratio']:.1f}   (similar if <= {MAX_MIN_RATIO_MAX})")
    print(f"mean / std                : {stats['mean']:.1f} / {stats['std']:.1f}")
    print(f"coefficient of variation  : {stats['cv']:.3f}   (similar if <= {CV_MAX})")
    print(f"Gini                      : {stats['gini']:.3f}")
    print(f"chi2 vs uniform (dof {stats['dof']}) : {stats['chi2']:.1f}   p = {p_str}   (similar if p > {UNIFORM_P_MIN})")

    scheme = (
        "leave-2-targets-out 7-fold CV (grouped by target)"
        if stats["similar"]
        else "sample-level 5-fold random CV on the non-test targets"
    )
    verdict = "SIMILAR" if stats["similar"] else "NOT similar"
    print(f"\nDecision: targets are {verdict} in size -> use {scheme}.")


if __name__ == "__main__":
    main()
