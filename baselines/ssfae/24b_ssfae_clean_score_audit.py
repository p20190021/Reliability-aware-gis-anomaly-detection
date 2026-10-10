"""
SSF-AE CLEAN SCORE AUDIT - NUMPY/PANDAS ONLY

This script intentionally does NOT import PyTorch.

It consumes the clean score vectors created by:
    24a_ssfae_clean_gpu_scoring.py

and performs:
- score statistics
- coverage statistics
- exact-budget boundary/tie audit
- exact deterministic top-k selection
- clean top-2% selected-index export
- branch/fused score correlations

No model training, refitting, labels, context, or injected data are used.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_N = 4_290_958

BUDGETS = (
    ("0.1%", 0.001),
    ("0.25%", 0.0025),
    ("0.5%", 0.005),
    ("1%", 0.01),
    ("2%", 0.02),
)

EXPECTED_K = {
    "0.1%": 4_291,
    "0.25%": 10_728,
    "0.5%": 21_455,
    "1%": 42_910,
    "2%": 85_820,
}

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "results" / "clean_scoring"

GPU_SUMMARY = OUTPUT_DIR / "SSFAE_Clean_GPU_Scoring_Summary.json"
CNN_PATH = OUTPUT_DIR / "SSFAE_Clean_CNN_Score.npy"
VIT_PATH = OUTPUT_DIR / "SSFAE_Clean_ViT_Score.npy"
FUSED_PATH = OUTPUT_DIR / "SSFAE_Clean_Fused_Score.npy"
COVERAGE_PATH = OUTPUT_DIR / "SSFAE_Clean_Window_Coverage.npy"


def exact_topk_selection(scores: np.ndarray, k: int):
    n = scores.size

    boundary = np.partition(scores, n - k)[n - k]

    above = np.flatnonzero(scores > boundary)
    tied = np.flatnonzero(scores == boundary)

    slots = k - above.size

    if slots < 0 or slots > tied.size:
        raise RuntimeError("Top-k tie accounting failed.")

    # tied is already increasing by global pixel index.
    selected = np.concatenate(
        [above, tied[:slots]]
    ).astype(np.int64)

    if selected.size != k:
        raise RuntimeError(
            f"Selected {selected.size}; expected {k}."
        )

    order = np.lexsort(
        (
            selected,
            -scores[selected],
        )
    )

    return {
        "boundary": float(boundary),
        "n_above": int(above.size),
        "n_tied": int(tied.size),
        "slots": int(slots),
        "selected_ranked": selected[order],
    }


def score_stats(name: str, x: np.ndarray):
    q = np.quantile(
        x,
        [
            0.0,
            0.001,
            0.01,
            0.05,
            0.25,
            0.50,
            0.75,
            0.95,
            0.99,
            0.999,
            1.0,
        ],
    )

    return {
        "score": name,
        "n": int(x.size),
        "finite": bool(np.isfinite(x).all()),
        "min": float(q[0]),
        "q0.1%": float(q[1]),
        "q1%": float(q[2]),
        "q5%": float(q[3]),
        "q25%": float(q[4]),
        "median": float(q[5]),
        "q75%": float(q[6]),
        "q95%": float(q[7]),
        "q99%": float(q[8]),
        "q99.9%": float(q[9]),
        "max": float(q[10]),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "zero_count": int(np.count_nonzero(x == 0.0)),
        "negative_count": int(np.count_nonzero(x < 0.0)),
    }


def main():
    required = [
        GPU_SUMMARY,
        CNN_PATH,
        VIT_PATH,
        FUSED_PATH,
        COVERAGE_PATH,
    ]

    for path in required:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing required clean-scoring artifact: {path}\n"
                "Run 24a_ssfae_clean_gpu_scoring.py first."
            )

    with open(GPU_SUMMARY, "r", encoding="utf-8") as f:
        gpu_summary = json.load(f)

    if (
        gpu_summary.get("status")
        != "clean_gpu_scoring_complete_postaudit_not_run"
    ):
        raise RuntimeError("GPU scoring summary status is not complete.")

    cnn = np.load(CNN_PATH, mmap_mode="r")
    vit = np.load(VIT_PATH, mmap_mode="r")
    fused = np.load(FUSED_PATH, mmap_mode="r")
    coverage = np.load(COVERAGE_PATH, mmap_mode="r")

    for name, arr in [
        ("CNN", cnn),
        ("ViT", vit),
        ("Fused", fused),
        ("Coverage", coverage),
    ]:
        if arr.shape != (EXPECTED_N,):
            raise RuntimeError(
                f"{name} shape {arr.shape}; expected {(EXPECTED_N,)}"
            )

    if not np.isfinite(cnn).all():
        raise RuntimeError("CNN score contains non-finite values.")
    if not np.isfinite(vit).all():
        raise RuntimeError("ViT score contains non-finite values.")
    if not np.isfinite(fused).all():
        raise RuntimeError("Fused score contains non-finite values.")

    if np.any(cnn < 0.0) or np.any(vit < 0.0) or np.any(fused < 0.0):
        raise RuntimeError("MSE score contains negative values.")

    if np.any(coverage == 0):
        raise RuntimeError("Coverage contains uncovered valid pixels.")

    w_cnn = float(gpu_summary["fusion"]["w_cnn"])
    w_vit = float(gpu_summary["fusion"]["w_vit"])

    reproduced = w_cnn * cnn + w_vit * vit
    fusion_max_abs = float(
        np.max(np.abs(fused - reproduced))
    )

    stats_df = pd.DataFrame(
        [
            score_stats("CNN", cnn),
            score_stats("ViT", vit),
            score_stats("Fused", fused),
        ]
    )

    stats_path = OUTPUT_DIR / "SSFAE_Clean_Score_Statistics.csv"
    stats_df.to_csv(stats_path, index=False)

    audit_rows = []
    top2 = None

    for label, budget in BUDGETS:
        k = int(math.ceil(budget * EXPECTED_N))

        if k != EXPECTED_K[label]:
            raise RuntimeError(
                f"Unexpected k for {label}: {k}"
            )

        sel = exact_topk_selection(fused, k)
        selected = sel["selected_ranked"]

        audit_rows.append(
            {
                "budget": label,
                "budget_fraction": budget,
                "k": k,
                "boundary_score": sel["boundary"],
                "strictly_above_boundary": sel["n_above"],
                "exact_boundary_tie_count": sel["n_tied"],
                "slots_taken_from_boundary_tie": sel["slots"],
                "boundary_tie_excess": (
                    sel["n_tied"] - sel["slots"]
                ),
                "selected_count": int(selected.size),
                "selected_score_min": float(fused[selected].min()),
                "selected_score_mean": float(fused[selected].mean()),
                "selected_score_max": float(fused[selected].max()),
            }
        )

        if label == "2%":
            top2 = selected

    audit_df = pd.DataFrame(audit_rows)

    audit_path = OUTPUT_DIR / "SSFAE_Clean_Budget_Tie_Audit.csv"
    audit_df.to_csv(audit_path, index=False)

    if top2 is None:
        raise RuntimeError("Top-2% set was not generated.")

    top2_path = OUTPUT_DIR / "SSFAE_Clean_Top2pct_Selected_Indices.npy"
    np.save(top2_path, top2.astype(np.int64, copy=False))

    corr = np.corrcoef(
        np.vstack(
            [
                np.asarray(cnn),
                np.asarray(vit),
                np.asarray(fused),
            ]
        )
    )

    final_summary = {
        "status": (
            "clean_fullscene_scoring_and_audit_complete_"
            "injected_data_not_used"
        ),
        "scientific_protocol": {
            "training_performed": False,
            "refitting_performed": False,
            "labels_loaded": False,
            "flood_context_loaded": False,
            "injected_data_loaded": False,
            "score_orientation": "higher reconstruction MSE = more anomalous",
            "exact_budget_selection": (
                "higher score first; exact ties resolved by lower global pixel index"
            ),
        },
        "gpu_scoring": gpu_summary,
        "coverage": {
            "min": int(np.min(coverage)),
            "mean": float(np.mean(coverage)),
            "median": float(np.median(coverage)),
            "max": int(np.max(coverage)),
            "uncovered": int(np.count_nonzero(coverage == 0)),
        },
        "fusion_reproduction_max_abs_error": fusion_max_abs,
        "score_correlations": {
            "CNN_vs_ViT": float(corr[0, 1]),
            "CNN_vs_Fused": float(corr[0, 2]),
            "ViT_vs_Fused": float(corr[1, 2]),
        },
        "score_stats": {
            row["score"]: {
                k: v
                for k, v in row.items()
                if k != "score"
            }
            for row in stats_df.to_dict(orient="records")
        },
        "budget_tie_audit": audit_df.to_dict(orient="records"),
        "outputs": {
            "cnn_score": str(CNN_PATH),
            "vit_score": str(VIT_PATH),
            "fused_score": str(FUSED_PATH),
            "coverage": str(COVERAGE_PATH),
            "statistics_csv": str(stats_path),
            "tie_audit_csv": str(audit_path),
            "top2_indices": str(top2_path),
        },
        "next_step_rule": (
            "If the clean top-2% boundary is not pathologically tied, "
            "freeze this score and proceed to clean top-2% characterization. "
            "Only after that should the 15 injected cases be scored."
        ),
    }

    summary_path = OUTPUT_DIR / "SSFAE_Clean_Scoring_Summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(final_summary, f, indent=2)

    print("=" * 76)
    print("SSF-AE CLEAN SCORE AUDIT COMPLETE")
    print("=" * 76)
    print(
        "Coverage min/mean/max:",
        int(np.min(coverage)),
        "/",
        float(np.mean(coverage)),
        "/",
        int(np.max(coverage)),
    )
    print("Fusion max |delta|:", fusion_max_abs)

    fused_row = stats_df[stats_df["score"] == "Fused"].iloc[0]
    print(
        "Fused min/mean/std/max:",
        fused_row["min"],
        "/",
        fused_row["mean"],
        "/",
        fused_row["std"],
        "/",
        fused_row["max"],
    )

    print("\nExact budget boundary audit:")
    print(
        audit_df[
            [
                "budget",
                "k",
                "boundary_score",
                "exact_boundary_tie_count",
                "slots_taken_from_boundary_tie",
                "boundary_tie_excess",
            ]
        ].to_string(index=False)
    )

    print("\nTop-2% indices:", top2_path)
    print("Summary:", summary_path)
    print("\nSTOP HERE. No injected dataset has been evaluated.")


if __name__ == "__main__":
    main()
