"""
SSF-AE POST-HOC RANGE-MECHANISM AUDIT
=====================================

Purpose
-------
The final SSF-AE injected evaluation is already frozen and complete.
This script DOES NOT change, retrain, refit, clip, normalize, or rescore SSF-AE.

It only asks a reviewer-relevant diagnostic question:

    Are the very high injected-anomaly recovery rates explained only by
    injected PCA values falling outside the clean [0,1] training range?

For every injection ratio and scenario, the script separates injected pixels into:
    A) fully in-range: all 21 frozen normalized PCA values lie in [0,1]
    B) out-of-range: at least one of the 21 values lies outside [0,1]

It then reports, using the ALREADY SAVED final exact top-k selections:
    - number / percentage of in-range and out-of-range injected pixels;
    - number of out-of-range PCA dimensions per injected pixel;
    - exceedance magnitude outside [0,1];
    - exact-budget detection rate for all injected pixels;
    - conditional detection rate for fully in-range injected pixels;
    - conditional detection rate for out-of-range injected pixels.

No alternative SSF-AE ranking is generated.
No clipped experiment is run.
No result is used to modify the frozen baseline.

Expected inputs
---------------
Code/SSFAE/results/final_injected_preprocessing/
    ratio_0p0005|ratio_0p0010|ratio_0p0050/
        inject_indices.npy
        <scenario>/SSFAE_Injected_Rows_PCA21.npy

Code/SSFAE/results/final_injected_evaluation/
    ratio_.../<scenario>/
        SSFAE_top_0p10pct_indices.npy
        SSFAE_top_0p25pct_indices.npy
        SSFAE_top_0p50pct_indices.npy
        SSFAE_top_1p00pct_indices.npy
        SSFAE_top_2p00pct_indices.npy

Optional full native score vectors from script 26b are used only for
descriptive correlation with out-of-range magnitude:
Code/SSFAE/results/final_injected_scoring/
    ratio_.../<scenario>/SSFAE_Injected_Fused_Score.npy

Outputs
-------
Code/SSFAE/results/range_mechanism_audit/
    SSFAE_Range_Audit_Case_Summary.csv
    SSFAE_Range_Audit_Conditional_Detection.csv
    SSFAE_Range_Audit_0p1pct_Reference.csv
    SSFAE_Range_Mechanism_Audit_Summary.json

Run
---
    python 27_ssfae_range_mechanism_audit.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_N = 4_290_958
REAL_DEPTH = 21

SCRIPT_DIR = Path(__file__).resolve().parent

PREP_DIR = SCRIPT_DIR / "results" / "final_injected_preprocessing"
EVAL_DIR = SCRIPT_DIR / "results" / "final_injected_evaluation"
SCORING_DIR = SCRIPT_DIR / "results" / "final_injected_scoring"
OUTPUT_DIR = SCRIPT_DIR / "results" / "range_mechanism_audit"

PREP_SUMMARY = PREP_DIR / "SSFAE_Injected_Preprocessing_Summary.json"
FINAL_SUMMARY = EVAL_DIR / "SSFAE_Final_All15_Summary.json"

RATIO_CONFIG = {
    "0.05%": "ratio_0p0005",
    "0.1%": "ratio_0p0010",
    "0.5%": "ratio_0p0050",
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

BUDGET_FILES = {
    0.10: "SSFAE_top_0p10pct_indices.npy",
    0.25: "SSFAE_top_0p25pct_indices.npy",
    0.50: "SSFAE_top_0p50pct_indices.npy",
    1.00: "SSFAE_top_1p00pct_indices.npy",
    2.00: "SSFAE_top_2p00pct_indices.npy",
}


def require_file(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found:\n{path}")
    return path


def safe_rate(detected: int, total: int):
    if total == 0:
        return np.nan
    return 100.0 * detected / total


def main():
    require_file(PREP_SUMMARY)
    require_file(FINAL_SUMMARY)

    with open(PREP_SUMMARY, "r", encoding="utf-8") as f:
        prep_summary = json.load(f)

    with open(FINAL_SUMMARY, "r", encoding="utf-8") as f:
        final_summary = json.load(f)

    if prep_summary.get("status") != "COMPLETE":
        raise RuntimeError("Injected preprocessing summary is not COMPLETE.")

    if final_summary.get("status") != "COMPLETE":
        raise RuntimeError("Final SSF-AE evaluation summary is not COMPLETE.")

    protocol = prep_summary.get("scientific_protocol", {})
    if protocol.get("inference_clipping") is not False:
        raise RuntimeError(
            "Expected the frozen SSF-AE protocol to have inference clipping OFF."
        )

    if final_summary.get("injected_data_used_for_training_or_tuning") is not False:
        raise RuntimeError("Final summary indicates injected-data tuning; aborting audit.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    case_rows = []
    conditional_rows = []

    print("=" * 78)
    print("SSF-AE POST-HOC RANGE-MECHANISM AUDIT")
    print("=" * 78)
    print("Model changes       : NONE")
    print("Rescoring           : NO")
    print("Clipping            : NO")
    print("Retuning/refitting  : NO")
    print("Purpose             : explain frozen final results only")

    for ratio_label, tag in RATIO_CONFIG.items():
        ratio_prep = PREP_DIR / tag
        inject_idx = np.load(
            require_file(ratio_prep / "inject_indices.npy"),
            allow_pickle=False,
        ).astype(np.int64, copy=False)

        if inject_idx.ndim != 1:
            raise ValueError(f"Bad inject_idx shape for {ratio_label}.")

        if np.unique(inject_idx).size != inject_idx.size:
            raise ValueError(f"Duplicate inject_idx for {ratio_label}.")

        print("\n" + "-" * 78)
        print(f"RATIO {ratio_label} | injected={inject_idx.size:,}")
        print("-" * 78)

        for scenario in SCENARIOS:
            transformed_path = require_file(
                ratio_prep / scenario / "SSFAE_Injected_Rows_PCA21.npy"
            )

            Xp = np.load(
                transformed_path,
                mmap_mode="r",
                allow_pickle=False,
            )

            if Xp.shape != (inject_idx.size, REAL_DEPTH):
                raise ValueError(
                    f"Unexpected transformed shape for {ratio_label}/{scenario}: "
                    f"{Xp.shape}"
                )

            arr = np.asarray(Xp, dtype=np.float64)

            below = arr < 0.0
            above = arr > 1.0
            oob_dim_mask = below | above

            oob_dim_count = oob_dim_mask.sum(axis=1)
            any_oob = oob_dim_count > 0
            fully_in_range = ~any_oob

            below_excess = np.maximum(-arr, 0.0)
            above_excess = np.maximum(arr - 1.0, 0.0)
            total_exceedance = (below_excess + above_excess).sum(axis=1)
            max_exceedance = (below_excess + above_excess).max(axis=1)

            n_oob = int(any_oob.sum())
            n_in = int(fully_in_range.sum())

            score_path = (
                SCORING_DIR
                / tag
                / scenario
                / "SSFAE_Injected_Fused_Score.npy"
            )

            score_corr = np.nan
            injected_score_mean_in = np.nan
            injected_score_mean_oob = np.nan

            if score_path.is_file():
                full_scores = np.load(
                    score_path,
                    mmap_mode="r",
                    allow_pickle=False,
                )

                if full_scores.shape != (EXPECTED_N,):
                    raise ValueError(
                        f"Bad full score shape for {ratio_label}/{scenario}."
                    )

                inj_scores = np.asarray(
                    full_scores[inject_idx],
                    dtype=np.float64,
                )

                # Spearman-style rank correlation via pandas, descriptive only.
                if (
                    np.std(total_exceedance) > 0
                    and np.std(inj_scores) > 0
                ):
                    score_corr = float(
                        pd.Series(total_exceedance)
                        .corr(
                            pd.Series(inj_scores),
                            method="spearman",
                        )
                    )

                if n_in > 0:
                    injected_score_mean_in = float(
                        inj_scores[fully_in_range].mean()
                    )

                if n_oob > 0:
                    injected_score_mean_oob = float(
                        inj_scores[any_oob].mean()
                    )

            case_row = {
                "Injection_Ratio": ratio_label,
                "Scenario": SCENARIO_DISPLAY[scenario],
                "Scenario_Key": scenario,
                "Total_Injected": int(inject_idx.size),
                "Fully_In_Range_Injected": n_in,
                "Fully_In_Range_Percent": safe_rate(n_in, inject_idx.size),
                "Any_Out_Of_Range_Injected": n_oob,
                "Any_Out_Of_Range_Percent": safe_rate(n_oob, inject_idx.size),
                "Mean_OOB_Dimensions_Per_Injected_Pixel": float(
                    oob_dim_count.mean()
                ),
                "Median_OOB_Dimensions_Per_Injected_Pixel": float(
                    np.median(oob_dim_count)
                ),
                "Max_OOB_Dimensions_Per_Injected_Pixel": int(
                    oob_dim_count.max()
                ),
                "Mean_Total_Exceedance": float(total_exceedance.mean()),
                "Median_Total_Exceedance": float(np.median(total_exceedance)),
                "Mean_Max_Exceedance": float(max_exceedance.mean()),
                "Spearman_Exceedance_vs_Injected_Score": score_corr,
                "Mean_Injected_Score_Fully_In_Range": injected_score_mean_in,
                "Mean_Injected_Score_Any_OOB": injected_score_mean_oob,
            }

            case_rows.append(case_row)

            for budget_percent, filename in BUDGET_FILES.items():
                selected = np.load(
                    require_file(
                        EVAL_DIR
                        / tag
                        / scenario
                        / filename
                    ),
                    allow_pickle=False,
                ).astype(np.int64, copy=False)

                selected_set = np.zeros(EXPECTED_N, dtype=bool)
                selected_set[selected] = True

                detected_mask = selected_set[inject_idx]

                detected_all = int(detected_mask.sum())
                detected_in = int(
                    np.count_nonzero(
                        detected_mask & fully_in_range
                    )
                )
                detected_oob = int(
                    np.count_nonzero(
                        detected_mask & any_oob
                    )
                )

                conditional_rows.append(
                    {
                        "Injection_Ratio": ratio_label,
                        "Scenario": SCENARIO_DISPLAY[scenario],
                        "Scenario_Key": scenario,
                        "Budget_Percent": budget_percent,
                        "Total_Injected": int(inject_idx.size),
                        "Detected_All": detected_all,
                        "Detection_Rate_All_Percent": safe_rate(
                            detected_all,
                            inject_idx.size,
                        ),
                        "Fully_In_Range_Total": n_in,
                        "Detected_Fully_In_Range": detected_in,
                        "Detection_Rate_Fully_In_Range_Percent": safe_rate(
                            detected_in,
                            n_in,
                        ),
                        "Any_OOB_Total": n_oob,
                        "Detected_Any_OOB": detected_oob,
                        "Detection_Rate_Any_OOB_Percent": safe_rate(
                            detected_oob,
                            n_oob,
                        ),
                    }
                )

            top2_row = conditional_rows[-1]

            print(
                f"{SCENARIO_DISPLAY[scenario]:28s} | "
                f"in-range={n_in:6d} ({case_row['Fully_In_Range_Percent']:6.2f}%) | "
                f"OOB={n_oob:6d} ({case_row['Any_Out_Of_Range_Percent']:6.2f}%) | "
                f"top2 all={top2_row['Detection_Rate_All_Percent']:6.2f}% | "
                f"in-range={top2_row['Detection_Rate_Fully_In_Range_Percent']:6.2f}% | "
                f"OOB={top2_row['Detection_Rate_Any_OOB_Percent']:6.2f}%"
            )

    case_df = pd.DataFrame(case_rows)
    cond_df = pd.DataFrame(conditional_rows)

    case_csv = OUTPUT_DIR / "SSFAE_Range_Audit_Case_Summary.csv"
    cond_csv = OUTPUT_DIR / "SSFAE_Range_Audit_Conditional_Detection.csv"

    case_df.to_csv(case_csv, index=False)
    cond_df.to_csv(cond_csv, index=False)

    ref = cond_df[
        cond_df["Injection_Ratio"] == "0.1%"
    ].copy()

    ref_csv = OUTPUT_DIR / "SSFAE_Range_Audit_0p1pct_Reference.csv"
    ref.to_csv(ref_csv, index=False)

    # A concise table at principal top-2% budget.
    top2 = cond_df[
        cond_df["Budget_Percent"] == 2.00
    ].copy()

    top2_csv = OUTPUT_DIR / "SSFAE_Range_Audit_Top2pct.csv"
    top2.to_csv(top2_csv, index=False)

    summary = {
        "stage": "SSFAE_posthoc_range_mechanism_audit",
        "status": "COMPLETE",
        "scientific_status": {
            "changes_frozen_model": False,
            "changes_frozen_preprocessing": False,
            "clips_inputs": False,
            "rescoring_performed": False,
            "reranking_performed": False,
            "retuning_performed": False,
            "uses_existing_exact_topk_selections_only": True,
        },
        "question": (
            "Are the frozen SSF-AE final injected detection rates explained "
            "only by normalized PCA values leaving the clean [0,1] range?"
        ),
        "interpretation_rule": (
            "If fully in-range injected pixels also retain high detection rates, "
            "then the final SSF-AE recovery cannot be attributed solely to the "
            "Sigmoid/output-range mismatch. If recovery is high almost exclusively "
            "for out-of-range injected pixels, this mechanism must be discussed "
            "prominently as an adaptation-specific sensitivity."
        ),
        "outputs": {
            "case_summary_csv": str(case_csv),
            "conditional_detection_csv": str(cond_csv),
            "reference_0p1_csv": str(ref_csv),
            "top2_csv": str(top2_csv),
        },
        "top2_results": top2.to_dict(orient="records"),
        "next_step": (
            "Review this audit without changing SSF-AE. Then integrate the frozen "
            "SSF-AE results with SCoNE, INA-CR, conventional baselines, Pure, and Full."
        ),
    }

    summary_path = OUTPUT_DIR / "SSFAE_Range_Mechanism_Audit_Summary.json"

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 78)
    print("SSF-AE RANGE-MECHANISM AUDIT COMPLETE")
    print("=" * 78)

    print("\n0.1% injection, top-2% conditional detection:")
    print(
        top2[
            top2["Injection_Ratio"] == "0.1%"
        ][
            [
                "Scenario",
                "Detection_Rate_All_Percent",
                "Fully_In_Range_Total",
                "Detection_Rate_Fully_In_Range_Percent",
                "Any_OOB_Total",
                "Detection_Rate_Any_OOB_Percent",
            ]
        ].to_string(index=False)
    )

    print("\nSummary:")
    print(" ", summary_path)
    print(
        "\nDo NOT use this diagnostic to retune or replace the frozen final SSF-AE result."
    )


if __name__ == "__main__":
    main()
