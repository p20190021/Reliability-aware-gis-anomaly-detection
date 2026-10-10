"""
Final audit of five-seed stability masks.

Purpose
-------
Reconstruct the exact top-2% clean anomaly set for IF, LOF, MD, PCA, AE,
HBOS, Pure, and Full directly from each seed's saved *native/raw ranking
score*, using the same deterministic rule as the manuscript:

    higher score first;
    if the k-th boundary score is tied, lower original valid-pixel index first.

This script DOES NOT train or refit any model. It reads only existing saved
scores and writes all audit outputs to a NEW folder.

Why this audit is useful
------------------------
The baseline training notebooks saved exact-count masks using np.argpartition
on normalized scores. The main clean comparison later reselected from raw
scores with the manuscript's deterministic tie rule. This audit makes the
five-seed stability analysis use that same final ranking convention as well.

Run
---
Place this file directly in:
<PROJECT_ROOT>

Then:
    python FINAL_stability_exact_raw_audit.py
"""

from __future__ import annotations
import os

from itertools import combinations
from pathlib import Path
import json

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("GIS_ANOMALY_PROJECT_ROOT", Path.cwd())).resolve()

SEEDS = [42, 7, 123, 2024, 2025]
N = 4_290_958
K = 85_820

BASELINE_FOLDER = "07_Baseline_Results_Seed_{seed}"
PROPOSED_FOLDER = "07_Proposed_Results_Seed_{seed}"

OUTPUT = ROOT / "07_Stability_Analysis_ExactRaw_Audit"
MASK_DIR = OUTPUT / "Exact_Masks"
MATRIX_DIR = OUTPUT / "Pairwise_Jaccard_Matrices"

OUTPUT.mkdir(parents=True, exist_ok=True)
MASK_DIR.mkdir(parents=True, exist_ok=True)
MATRIX_DIR.mkdir(parents=True, exist_ok=True)

METHODS = {
    "IF": {
        "family": "baseline",
        "scores": ["Baseline_IF_scores_raw.npy", "IF_scores_raw.npy"],
        "old_masks": ["Baseline_IF_anomaly_labels.npy", "IF_anomaly_labels.npy"],
    },
    "LOF": {
        "family": "baseline",
        "scores": ["Baseline_LOF_scores_raw.npy", "LOF_scores_raw.npy"],
        "old_masks": ["Baseline_LOF_anomaly_labels.npy", "LOF_anomaly_labels.npy"],
    },
    "MD": {
        "family": "baseline",
        "scores": ["Baseline_MD_scores_raw.npy", "MD_scores_raw.npy"],
        "old_masks": ["Baseline_MD_anomaly_labels.npy", "MD_anomaly_labels.npy"],
    },
    "PCA": {
        "family": "baseline",
        "scores": ["Baseline_PCA_scores_raw.npy", "PCA_scores_raw.npy"],
        "old_masks": ["Baseline_PCA_anomaly_labels.npy", "PCA_anomaly_labels.npy"],
    },
    "AE": {
        "family": "baseline",
        "scores": ["Baseline_AE_scores_raw.npy", "AE_scores_raw.npy"],
        "old_masks": ["Baseline_AE_anomaly_labels.npy", "AE_anomaly_labels.npy"],
    },
    "HBOS": {
        "family": "baseline",
        "scores": ["Baseline_HBOS_scores_raw.npy", "HBOS_scores_raw.npy"],
        "old_masks": ["Baseline_HBOS_anomaly_labels.npy", "HBOS_anomaly_labels.npy"],
    },
    "Pure": {
        "family": "proposed",
        "scores": [
            "pure_final_fused_score_preclip.npy",
            "Pure_final_fused_score_preclip.npy",
            "pure_final_anomaly_score_raw.npy",
        ],
        "old_masks": [
            "Pure_top2pct_anomaly_mask.npy",
            "pure_final_anomaly_labels.npy",
        ],
    },
    "Full": {
        "family": "proposed",
        "scores": [
            "full_final_fused_score_preclip.npy",
            "Full_final_fused_score_preclip.npy",
            "full_final_anomaly_score_raw.npy",
        ],
        "old_masks": [
            "Full_top2pct_anomaly_mask.npy",
            "full_final_anomaly_labels.npy",
        ],
    },
}


def first_existing(folder: Path, names: list[str]) -> Path | None:
    for name in names:
        p = folder / name
        if p.is_file():
            return p
    return None


def exact_top_k(scores: np.ndarray, k: int) -> tuple[np.ndarray, dict]:
    """
    Return sorted selected indices.

    Higher score = more anomalous.
    Exact boundary ties: lower global index first.
    """
    s = np.asarray(scores).reshape(-1)

    if s.size != N:
        raise ValueError(f"Score length {s.size:,}, expected {N:,}.")
    if not np.isfinite(s).all():
        raise ValueError("Scores contain NaN/Inf.")

    boundary_pos = s.size - k
    threshold = np.partition(s, boundary_pos)[boundary_pos]

    above = np.flatnonzero(s > threshold)
    tied = np.flatnonzero(s == threshold)  # ascending global index

    need = k - above.size
    if need < 0 or need > tied.size:
        raise RuntimeError("Invalid tie accounting.")

    selected = np.concatenate([above, tied[:need]]).astype(np.int64, copy=False)
    selected.sort()

    if selected.size != k or np.unique(selected).size != k:
        raise RuntimeError("Exact top-k selection failed.")

    audit = {
        "threshold": float(threshold),
        "strictly_above": int(above.size),
        "boundary_tie_count": int(tied.size),
        "selected_from_boundary_tie": int(need),
        "boundary_tie_excess": int(tied.size - need),
    }
    return selected, audit


def old_mask_to_indices(path: Path) -> np.ndarray:
    arr = np.load(path, mmap_mode="r", allow_pickle=False)
    arr = np.asarray(arr).reshape(-1)
    if arr.size != N:
        raise ValueError(f"Old mask length mismatch: {path}")
    idx = np.flatnonzero(arr > 0).astype(np.int64, copy=False)
    return idx


def jaccard_sorted(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.intersect1d(a, b, assume_unique=True).size
    union = a.size + b.size - inter
    return float(inter / union)


all_selected: dict[str, dict[int, np.ndarray]] = {
    method: {} for method in METHODS
}
audit_rows = []

print("=" * 80)
print("FINAL STABILITY EXACT-RAW AUDIT")
print("=" * 80)
print("No training/refitting is performed.")
print(f"N={N:,}, exact top-2% K={K:,}")

for method, cfg in METHODS.items():
    for seed in SEEDS:
        folder = ROOT / (
            BASELINE_FOLDER.format(seed=seed)
            if cfg["family"] == "baseline"
            else PROPOSED_FOLDER.format(seed=seed)
        )

        if not folder.is_dir():
            raise FileNotFoundError(f"Missing folder:\n{folder}")

        score_path = first_existing(folder, cfg["scores"])
        if score_path is None:
            raise FileNotFoundError(
                f"No native/raw score file found for {method}, seed {seed} in:\n{folder}\n"
                f"Checked: {cfg['scores']}"
            )

        scores = np.load(score_path, mmap_mode="r", allow_pickle=False)
        selected, tie = exact_top_k(scores, K)
        all_selected[method][seed] = selected

        save_path = MASK_DIR / f"{method}_Seed_{seed}_exact_top2_indices.npy"
        np.save(save_path, selected)

        old_mask_path = first_existing(folder, cfg["old_masks"])
        old_count = None
        symmetric_difference = None
        old_vs_exact_jaccard = None

        if old_mask_path is not None:
            old_idx = old_mask_to_indices(old_mask_path)
            old_count = int(old_idx.size)
            inter = np.intersect1d(old_idx, selected, assume_unique=True).size
            symmetric_difference = int(old_idx.size + selected.size - 2 * inter)
            union = old_idx.size + selected.size - inter
            old_vs_exact_jaccard = float(inter / union) if union else 1.0

        row = {
            "Method": method,
            "Seed": seed,
            "Score_File": str(score_path),
            "Old_Mask_File": str(old_mask_path) if old_mask_path else "",
            "Exact_Selected": int(selected.size),
            "Boundary_Threshold": tie["threshold"],
            "Strictly_Above": tie["strictly_above"],
            "Boundary_Tie_Count": tie["boundary_tie_count"],
            "Selected_From_Boundary_Tie": tie["selected_from_boundary_tie"],
            "Boundary_Tie_Excess": tie["boundary_tie_excess"],
            "Old_Selected_Count": old_count,
            "Old_vs_Exact_Symmetric_Difference": symmetric_difference,
            "Old_vs_Exact_Jaccard": old_vs_exact_jaccard,
        }
        audit_rows.append(row)

        print(
            f"{method:5s} seed={seed:4d} | "
            f"tie={tie['boundary_tie_count']:7d} | "
            f"excess={tie['boundary_tie_excess']:7d} | "
            f"old-vs-exact J={old_vs_exact_jaccard if old_vs_exact_jaccard is not None else float('nan'):.6f}"
        )

audit_df = pd.DataFrame(audit_rows)
audit_csv = OUTPUT / "Exact_Raw_Mask_Audit.csv"
audit_df.to_csv(audit_csv, index=False)

summary_rows = []
pairwise_long_rows = []

for method in METHODS:
    mat = pd.DataFrame(
        np.eye(len(SEEDS), dtype=float),
        index=SEEDS,
        columns=SEEDS,
    )

    vals = []
    for s1, s2 in combinations(SEEDS, 2):
        j = jaccard_sorted(
            all_selected[method][s1],
            all_selected[method][s2],
        )
        mat.loc[s1, s2] = j
        mat.loc[s2, s1] = j
        vals.append(j)

        pairwise_long_rows.append(
            {
                "Method": method,
                "Seed_1": s1,
                "Seed_2": s2,
                "Jaccard": j,
            }
        )

    core = all_selected[method][SEEDS[0]]
    for seed in SEEDS[1:]:
        core = np.intersect1d(
            core,
            all_selected[method][seed],
            assume_unique=True,
        )

    summary_rows.append(
        {
            "Method": method,
            "Mean_Jaccard": float(np.mean(vals)),
            "Std_Jaccard": float(np.std(vals, ddof=0)),
            "Minimum": float(np.min(vals)),
            "Maximum": float(np.max(vals)),
            "All_Seed_Intersection": int(core.size),
            "Core_Retention_Percent": float(100.0 * core.size / K),
        }
    )

    mat.to_csv(
        MATRIX_DIR / f"{method}_Pairwise_Jaccard_ExactRaw.csv"
    )

summary_df = pd.DataFrame(summary_rows)
pairwise_df = pd.DataFrame(pairwise_long_rows)

summary_csv = OUTPUT / "Stability_Summary_ExactRaw.csv"
pairwise_csv = OUTPUT / "Pairwise_Jaccard_Long_ExactRaw.csv"

summary_df.to_csv(summary_csv, index=False)
pairwise_df.to_csv(pairwise_csv, index=False)

max_diff = audit_df["Old_vs_Exact_Symmetric_Difference"].dropna()
max_diff_value = int(max_diff.max()) if len(max_diff) else None

summary_json = {
    "status": "complete",
    "training_performed": False,
    "refitting_performed": False,
    "selection_rule": (
        "native/raw score descending; exact k=85,820; "
        "lower original valid-pixel index resolves the k-th-score tie"
    ),
    "seeds": SEEDS,
    "methods": list(METHODS),
    "max_old_vs_exact_symmetric_difference": max_diff_value,
    "outputs": {
        "mask_audit": str(audit_csv),
        "stability_summary": str(summary_csv),
        "pairwise_long": str(pairwise_csv),
        "matrix_dir": str(MATRIX_DIR),
    },
}

with open(
    OUTPUT / "FINAL_Stability_ExactRaw_Audit_Summary.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(summary_json, f, indent=2)

print("\n" + "=" * 80)
print("EXACT-RAW STABILITY SUMMARY")
print("=" * 80)
print(summary_df.to_string(index=False))

print("\nMaximum old-vs-exact symmetric difference:", max_diff_value)
print("Audit CSV :", audit_csv)
print("Summary   :", summary_csv)
print("\nNo existing result folder was modified.")
