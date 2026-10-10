"""
SSF-AE FINAL ALL-15 EXACT-BUDGET EVALUATION - NUMPY/PANDAS ONLY

Consumes the 15 full-scene native SSF-AE score vectors produced by script 26b
and computes the frozen manuscript evaluation.

Primary metric
--------------
Injected-anomaly detection rate (%) within exact top-k candidate sets.

Budgets
-------
0.1%, 0.25%, 0.5%, 1%, 2%

Exact ranking rule
------------------
Higher native fused SSF-AE score = more anomalous.
Exact kth-score ties are resolved by lower original valid-vector index.

This script:
- loads each ratio's authoritative inject_indices.npy;
- validates injection_ground_truth_labels.npy when present;
- performs NO training, refitting, rescoring, or final-score normalization;
- does NOT use flood context;
- saves exact ranking/tie diagnostics and aggregate result tables;
- imports NO PyTorch.

Run
---
    python 26c_ssfae_final_all15_budget_evaluation.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_N = 4_290_958

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent

INJECTED_ROOT = ROOT_DIR / "03_Injected_Proposed_Results_Seed_42"
PREPARED_DIR = SCRIPT_DIR / "results" / "final_injected_preprocessing"
SCORING_DIR = SCRIPT_DIR / "results" / "final_injected_scoring"
OUTPUT_DIR = SCRIPT_DIR / "results" / "final_injected_evaluation"

BUDGETS = [
    0.0010,
    0.0025,
    0.0050,
    0.0100,
    0.0200,
]

RATIO_CONFIG = {
    "0.05%": {
        "ratio": 0.0005,
        "folder": "Injected_Datasets_with_0.05%_Injection_Rate",
        "tag": "ratio_0p0005",
    },
    "0.1%": {
        "ratio": 0.0010,
        "folder": "Injected_Datasets_with_0.1%_Injection_Rate",
        "tag": "ratio_0p0010",
    },
    "0.5%": {
        "ratio": 0.0050,
        "folder": "Injected_Datasets_with_0.5%_Injection_Rate",
        "tag": "ratio_0p0050",
    },
}

SCENARIOS = [
    "extreme",
    "noise_strong",
    "noise_moderate",
    "noise_mild",
    "inconsistent",
]

SCENARIO_DISPLAY = {
    "extreme": "Extreme",
    "noise_strong": "Strong noise",
    "noise_moderate": "Moderate noise",
    "noise_mild": "Mild noise",
    "inconsistent": "Cross-feature inconsistency",
}


def require_file(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Required file was not found:\n{path}")
    return path


def load_injection_definition(ratio_dir: Path):
    idx_path = require_file(ratio_dir / "inject_indices.npy")
    inject_idx = np.load(idx_path, allow_pickle=False).astype(np.int64, copy=False)

    if inject_idx.ndim != 1:
        raise ValueError(f"inject_indices.npy must be 1-D: {idx_path}")
    if np.any(inject_idx < 0) or np.any(inject_idx >= EXPECTED_N):
        raise IndexError(f"Out-of-range injection indices: {idx_path}")
    if np.unique(inject_idx).size != inject_idx.size:
        raise ValueError(f"Duplicate injection indices: {idx_path}")

    labels_path = ratio_dir / "injection_ground_truth_labels.npy"
    labels_validated = False

    if labels_path.is_file():
        labels = np.load(labels_path, mmap_mode="r", allow_pickle=False)
        labels = np.asarray(labels).reshape(-1)

        if labels.size != EXPECTED_N:
            raise ValueError(f"Ground-truth label length mismatch: {labels_path}")

        gt_idx = np.flatnonzero(labels > 0)

        if not np.array_equal(np.sort(inject_idx), gt_idx):
            raise ValueError(
                f"inject_indices.npy does not match injection_ground_truth_labels.npy:\n"
                f"{ratio_dir}"
            )

        labels_validated = True

    return inject_idx, labels_validated


def exact_top_k_order(scores, k):
    scores = np.asarray(scores).reshape(-1)

    if not 1 <= k <= scores.size:
        raise ValueError("Invalid top-k size.")

    boundary_position = scores.size - k

    threshold = np.partition(
        scores,
        boundary_position,
    )[boundary_position]

    above = np.flatnonzero(
        scores > threshold
    )

    tied = np.flatnonzero(
        scores == threshold
    )

    need = k - above.size

    if need < 0 or need > tied.size:
        raise RuntimeError("Unexpected exact top-k tie accounting.")

    selected = np.concatenate(
        [
            above,
            tied[:need],
        ]
    ).astype(
        np.int64,
        copy=False,
    )

    if selected.size != k:
        raise RuntimeError(
            f"Exact top-k returned {selected.size}; expected {k}."
        )

    order = np.lexsort(
        (
            selected,
            -scores[selected],
        )
    )

    return {
        "selected": selected[order],
        "threshold": float(threshold),
        "strictly_above": int(above.size),
        "tie_count": int(tied.size),
        "selected_from_tie": int(need),
        "tie_excess": int(tied.size - need),
    }


def main():
    gpu_summary_path = require_file(
        SCORING_DIR / "SSFAE_All15_GPU_Scoring_Summary.json"
    )

    with open(gpu_summary_path, "r", encoding="utf-8") as f:
        gpu_summary = json.load(f)

    if gpu_summary.get("status") != "COMPLETE":
        raise RuntimeError("All-15 GPU scoring summary is not complete.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    budget_rows = []
    scenario_rows = []

    print("=" * 78)
    print("SSF-AE FINAL ALL-15 EXACT-BUDGET EVALUATION")
    print("=" * 78)
    print("Training/refitting       : NO")
    print("Final score normalization: NO")
    print("Higher score anomalous   : YES")
    print("Exact lower-index tie rule: YES")

    for ratio_label, cfg in RATIO_CONFIG.items():
        ratio_dir = INJECTED_ROOT / cfg["folder"]
        inject_idx, labels_validated = load_injection_definition(ratio_dir)

        ratio_tag = cfg["tag"]
        ratio_out = OUTPUT_DIR / ratio_tag
        ratio_out.mkdir(parents=True, exist_ok=True)

        np.save(
            ratio_out / "inject_indices.npy",
            inject_idx,
        )

        print("\n" + "=" * 78)
        print(
            f"{ratio_label}: {inject_idx.size:,}/{EXPECTED_N:,} injected "
            f"({100.0 * inject_idx.size / EXPECTED_N:.6f}%)"
        )
        print("=" * 78)

        for scenario in SCENARIOS:
            score_path = require_file(
                SCORING_DIR
                / ratio_tag
                / scenario
                / "SSFAE_Injected_Fused_Score.npy"
            )

            scores = np.load(
                score_path,
                mmap_mode="r",
                allow_pickle=False,
            )

            if scores.shape != (EXPECTED_N,):
                raise ValueError(
                    f"Bad score shape for {ratio_label}/{scenario}: {scores.shape}"
                )

            if not np.isfinite(scores).all():
                raise ValueError(
                    f"Non-finite score in {ratio_label}/{scenario}."
                )

            scenario_out = ratio_out / scenario
            scenario_out.mkdir(parents=True, exist_ok=True)

            injected_scores = np.asarray(scores[inject_idx], dtype=np.float64)

            total_score_sum = float(
                np.sum(
                    scores,
                    dtype=np.float64,
                )
            )

            injected_score_sum = float(
                np.sum(
                    injected_scores,
                    dtype=np.float64,
                )
            )

            noninjected_n = EXPECTED_N - inject_idx.size
            noninjected_mean = (
                total_score_sum - injected_score_sum
            ) / noninjected_n

            scenario_row = {
                "Injection_Ratio": ratio_label,
                "Injection_Ratio_Fraction_Target": float(cfg["ratio"]),
                "Injection_Ratio_Fraction_Actual": float(
                    inject_idx.size / EXPECTED_N
                ),
                "Scenario": SCENARIO_DISPLAY[scenario],
                "Scenario_Key": scenario,
                "Total_Injected": int(inject_idx.size),
                "Ground_Truth_Labels_Validated": bool(labels_validated),
                "Score_Min": float(np.min(scores)),
                "Score_Max": float(np.max(scores)),
                "Score_Mean": float(total_score_sum / EXPECTED_N),
                "Injected_Score_Mean": float(injected_score_sum / inject_idx.size),
                "Injected_Score_Median": float(np.median(injected_scores)),
                "NonInjected_Score_Mean": float(noninjected_mean),
                "Injected_Minus_NonInjected_Mean": float(
                    injected_score_sum / inject_idx.size - noninjected_mean
                ),
                "Score_Source": str(score_path),
            }

            scenario_rows.append(scenario_row)

            per_case_budget_rows = []

            for budget in BUDGETS:
                k = int(math.ceil(budget * EXPECTED_N))

                selection = exact_top_k_order(
                    scores,
                    k,
                )

                selected = selection["selected"]

                # Membership test without allocating a full N-length label vector.
                detected = int(
                    np.count_nonzero(
                        np.isin(
                            inject_idx,
                            selected,
                            assume_unique=True,
                        )
                    )
                )

                total_injected = int(inject_idx.size)
                missed = total_injected - detected

                detection_rate = (
                    100.0
                    * detected
                    / total_injected
                )

                theoretical_max = min(
                    100.0,
                    100.0 * k / total_injected,
                )

                row = {
                    "Injection_Ratio": ratio_label,
                    "Injection_Ratio_Fraction_Target": float(cfg["ratio"]),
                    "Injection_Ratio_Fraction_Actual": float(
                        total_injected / EXPECTED_N
                    ),
                    "Scenario": SCENARIO_DISPLAY[scenario],
                    "Scenario_Key": scenario,
                    "Budget_Fraction": float(budget),
                    "Budget_Percent": float(100.0 * budget),
                    "Selected_Count": int(k),
                    "Total_Injected": total_injected,
                    "Detected_Injected": detected,
                    "Missed_Injected": missed,
                    "Detection_Rate_Percent": float(detection_rate),
                    "Theoretical_Max_Detection_Rate_Percent": float(
                        theoretical_max
                    ),
                    "Detection_As_Percent_Of_Theoretical_Max": float(
                        100.0 * detection_rate / theoretical_max
                        if theoretical_max > 0
                        else np.nan
                    ),
                    "Raw_Final_Score_Threshold": selection["threshold"],
                    "Scores_Strictly_Greater_Than_Threshold": (
                        selection["strictly_above"]
                    ),
                    "Scores_Equal_To_Threshold": selection["tie_count"],
                    "Selected_From_Boundary_Tie": selection["selected_from_tie"],
                    "Boundary_Tie_Excess": selection["tie_excess"],
                }

                budget_rows.append(row)
                per_case_budget_rows.append(row)

                token = (
                    f"{100.0 * budget:.2f}"
                    .replace(".", "p")
                    + "pct"
                )

                np.save(
                    scenario_out
                    / f"SSFAE_top_{token}_indices.npy",
                    selected,
                )

            pd.DataFrame(
                per_case_budget_rows
            ).to_csv(
                scenario_out / "SSFAE_Budget_Detection_Rates.csv",
                index=False,
            )

            top2 = per_case_budget_rows[-1]

            print(
                f"{SCENARIO_DISPLAY[scenario]:28s} | "
                f"top-2% DR={top2['Detection_Rate_Percent']:.3f}% | "
                f"injected mean={scenario_row['Injected_Score_Mean']:.6g} | "
                f"non-injected mean={scenario_row['NonInjected_Score_Mean']:.6g}"
            )

    budget_df = pd.DataFrame(budget_rows)
    scenario_df = pd.DataFrame(scenario_rows)

    budget_csv = OUTPUT_DIR / "SSFAE_All15_Budget_Detection_Rates.csv"
    scenario_csv = OUTPUT_DIR / "SSFAE_All15_Scenario_Diagnostics.csv"

    budget_df.to_csv(budget_csv, index=False)
    scenario_df.to_csv(scenario_csv, index=False)

    mean_df = (
        budget_df
        .groupby(
            [
                "Injection_Ratio",
                "Budget_Fraction",
                "Budget_Percent",
                "Selected_Count",
            ]
        )["Detection_Rate_Percent"]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
        .rename(
            columns={
                "mean": "Mean_Detection_Rate_Percent",
                "std": "Std_Detection_Rate_Percent",
                "min": "Min_Detection_Rate_Percent",
                "max": "Max_Detection_Rate_Percent",
            }
        )
    )

    mean_csv = OUTPUT_DIR / "SSFAE_Mean_Detection_Rates_By_Ratio.csv"
    mean_df.to_csv(mean_csv, index=False)

    reference_01 = budget_df[
        budget_df["Injection_Ratio"] == "0.1%"
    ].copy()

    reference_csv = (
        OUTPUT_DIR
        / "SSFAE_Reference_0p1_Scenario_Detection_Table.csv"
    )
    reference_01.to_csv(reference_csv, index=False)

    tie_problem_count = int(
        np.count_nonzero(
            budget_df["Boundary_Tie_Excess"].to_numpy() > 0
        )
    )

    top2_summary = (
        budget_df[
            budget_df["Budget_Fraction"] == 0.0200
        ][
            [
                "Injection_Ratio",
                "Scenario",
                "Detection_Rate_Percent",
                "Raw_Final_Score_Threshold",
                "Scores_Equal_To_Threshold",
                "Boundary_Tie_Excess",
            ]
        ]
        .copy()
    )

    top2_csv = OUTPUT_DIR / "SSFAE_Top2pct_All15_Summary.csv"
    top2_summary.to_csv(top2_csv, index=False)

    summary = {
        "stage": "SSFAE_final_all15_exact_budget_evaluation",
        "status": "COMPLETE",
        "cases": 15,
        "budget_evaluations": int(len(budget_df)),
        "ranking_score": (
            "frozen native SSF-AE loss-weighted fused per-pixel reconstruction "
            "MSE after overlap averaging; higher is more anomalous"
        ),
        "ranking_normalization": "none",
        "exact_ranking_rule": (
            "higher score first; lower original valid-vector index for exact ties"
        ),
        "final_budgets": BUDGETS,
        "ratios": RATIO_CONFIG,
        "scenarios": SCENARIOS,
        "spatial_scoring": (
            "Every injected case was full-rescored over all frozen inference "
            "windows; clean non-injected scores were not simply patched because "
            "spatial context allows an injected pixel to affect neighboring "
            "pixel reconstruction errors."
        ),
        "injected_data_used_for_training_or_tuning": False,
        "training_or_refitting_during_evaluation": False,
        "flood_context_used": False,
        "final_score_normalization_before_ranking": False,
        "boundary_budget_cases_with_tie_excess": tie_problem_count,
        "outputs": {
            "all_budget_detection_rates": str(budget_csv),
            "scenario_diagnostics": str(scenario_csv),
            "mean_detection_rates_by_ratio": str(mean_csv),
            "reference_0p1_table": str(reference_csv),
            "top2_all15_summary": str(top2_csv),
        },
        "mean_detection_rates_by_ratio": mean_df.to_dict(orient="records"),
        "top2_all15": top2_summary.to_dict(orient="records"),
        "next_step": (
            "Integrate SSF-AE with SCoNE, INA-CR, conventional baselines, Pure, "
            "and Full in the manuscript comparison. Do not tune SSF-AE based on "
            "these final injected results."
        ),
    }

    summary_path = OUTPUT_DIR / "SSFAE_Final_All15_Summary.json"

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 78)
    print("SSF-AE FINAL ALL-15 EVALUATION COMPLETE")
    print("=" * 78)
    print("Budget rows         :", len(budget_df))
    print("Tie-problem cases   :", tie_problem_count)

    print("\nMean detection rate across five scenarios:")
    print(
        mean_df[
            [
                "Injection_Ratio",
                "Budget_Percent",
                "Mean_Detection_Rate_Percent",
            ]
        ].to_string(index=False)
    )

    print("\n0.1% injection scenario table:")
    print(
        reference_01[
            [
                "Scenario",
                "Budget_Percent",
                "Detection_Rate_Percent",
            ]
        ].to_string(index=False)
    )

    print("\nSummary:")
    print(" ", summary_path)
    print(
        "\nFINAL RULE: do not alter the frozen SSF-AE architecture, "
        "preprocessing, training, or fusion based on these results."
    )


if __name__ == "__main__":
    main()
