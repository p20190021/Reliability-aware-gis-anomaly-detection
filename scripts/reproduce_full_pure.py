#!/usr/bin/env python3
"""Reproduce the reported Full/Pure post-training results from saved scores.

This is the publication-facing replacement for the historical chain of
notebooks/correction scripts. It imports the single canonical implementation in
``src/proposed_method.py`` and never modifies the research source folders.

It intentionally does not retrain MVAE, context AE, IF, LOF or Mahalanobis
models. Those upstream stages are consolidated separately. This script answers
a narrower reproducibility question: given the saved normalized upstream
component scores, does the cleaned final scoring code reproduce the reported
Full/Pure results?

Expected legacy research layout
-------------------------------
<project-root>/
  valid_mask_1.npy
  03_Proposed_Results_Seed_42/
    ae_score_normalized.npy
    if_feature_score_normalized.npy
    lof_feature_score_normalized.npy
    md_feature_score_normalized.npy
    if_embedding_score_normalized.npy
    lof_embedding_score_normalized.npy
    md_embedding_score_normalized.npy
    context_score_normalized.npy
    pure_spatial_score_normalized.npy        # optional parity check
    pure_final_anomaly_score_raw.npy         # optional parity check
  03_Injected_Proposed_Results_Seed_42/
    Injected_Datasets_with_0.05%_Injection_Rate/
    Injected_Datasets_with_0.1%_Injection_Rate/
    Injected_Datasets_with_0.5%_Injection_Rate/

Example
-------
python scripts/reproduce_full_pure.py ^
  --project-root "C:\\...\\Outlier Paper\\Code"

Outputs are written only to ``--output`` (default:
``<project-root>/PUBLIC_REPRODUCTION_FULL_PURE``).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# Allow direct execution from the repository without installation.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from proposed_method import (  # noqa: E402
    BETA,
    ALPHA,
    FULL_COMPONENT_ORDER,
    LAMBDA_AGREEMENT,
    LAMBDA_CERTAINTY,
    SPATIAL_WINDOW,
    TOP_FRACTION,
    exact_top_k,
    normalize_robust,
    robust_bounds,
    score_clean,
    score_injected,
)


EXPECTED_MASK_SHAPE = (2897, 2696)
EXPECTED_N_SAMPLES = 4_290_958

BUDGETS = (0.001, 0.0025, 0.005, 0.01, 0.02)
SCENARIOS = (
    "extreme",
    "noise_strong",
    "noise_moderate",
    "noise_mild",
    "inconsistent",
)
RATIO_FOLDERS = {
    "0.05%": "Injected_Datasets_with_0.05%_Injection_Rate",
    "0.1%": "Injected_Datasets_with_0.1%_Injection_Rate",
    "0.5%": "Injected_Datasets_with_0.5%_Injection_Rate",
}

COMPONENT_FILES = {
    "AE": "ae_score_normalized.npy",
    "IF_feature": "if_feature_score_normalized.npy",
    "LOF_feature": "lof_feature_score_normalized.npy",
    "MD_feature": "md_feature_score_normalized.npy",
    "IF_embedding": "if_embedding_score_normalized.npy",
    "LOF_embedding": "lof_embedding_score_normalized.npy",
    "MD_embedding": "md_embedding_score_normalized.npy",
}
CONTEXT_FILE = "context_score_normalized.npy"

# Locked manuscript values retained only as additional sanity checks.
HEADLINE_EXPECTED = {
    ("0.1%", "Full", 0.001): 82.11655011655012,
    ("0.1%", "Pure", 0.001): 86.53146853146853,
    ("0.1%", "Full", 0.020): 99.03962703962704,
    ("0.1%", "Pure", 0.020): 98.42424242424242,
}


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Required file is missing: {path}")
    return path


def load_vector(path: Path) -> np.ndarray:
    arr = np.load(require_file(path), mmap_mode="r", allow_pickle=False)
    arr = np.asarray(arr).reshape(-1)
    if not np.isfinite(arr).all():
        raise ValueError(f"Non-finite values found in {path}")
    return arr.astype(np.float32, copy=False)


def load_component_scores(folder: Path) -> tuple[dict[str, np.ndarray], np.ndarray]:
    comps = {
        name: load_vector(folder / COMPONENT_FILES[name])
        for name in FULL_COMPONENT_ORDER
    }
    context = load_vector(folder / CONTEXT_FILE)
    lengths = {arr.size for arr in [*comps.values(), context]}
    if len(lengths) != 1:
        raise ValueError(f"Component length mismatch in {folder}: {sorted(lengths)}")
    return comps, context


def resolve_valid_mask(project_root: Path, explicit: Path | None) -> tuple[np.ndarray, Path]:
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit.expanduser())
    candidates.extend(
        [
            project_root / "valid_mask_1.npy",
            project_root / "03_Proposed_Results_Seed_42" / "valid_mask_1.npy",
            project_root / "valid_mask.npy",
            project_root / "03_Proposed_Results_Seed_42" / "valid_mask.npy",
        ]
    )

    path = next((p.resolve() for p in candidates if p.is_file()), None)
    if path is None:
        raise FileNotFoundError(
            "Could not locate valid_mask_1.npy (preferred) or valid_mask.npy. "
            "Supply --valid-mask explicitly."
        )

    mask = np.load(path, allow_pickle=False).astype(bool, copy=False)
    if mask.ndim != 2:
        raise ValueError(f"Valid mask must be 2-D: {path}; found {mask.shape}")
    if tuple(mask.shape) != EXPECTED_MASK_SHAPE:
        raise ValueError(
            f"Unexpected mask shape {mask.shape}; expected {EXPECTED_MASK_SHAPE}"
        )
    n_valid = int(mask.sum())
    if n_valid != EXPECTED_N_SAMPLES:
        raise ValueError(
            f"Mask has {n_valid:,} valid cells; expected {EXPECTED_N_SAMPLES:,}"
        )
    return mask, path


def evaluate_budgets(
    scores: np.ndarray,
    inject_idx: np.ndarray,
    *,
    method: str,
    ratio: str,
    scenario: str,
) -> list[dict]:
    inject_idx = np.asarray(inject_idx, dtype=np.int64).reshape(-1)
    if inject_idx.size == 0:
        raise ValueError("No injection indices")
    if inject_idx.min() < 0 or inject_idx.max() >= scores.size:
        raise IndexError("Injection index outside score-vector range")

    rows = []
    for budget in BUDGETS:
        mask, idx, threshold = exact_top_k(scores, fraction=budget)
        detected = int(mask[inject_idx].sum())
        rows.append(
            {
                "Injection_Ratio": ratio,
                "Scenario": scenario,
                "Method": method,
                "Budget_Fraction": budget,
                "Budget_Percent": budget * 100.0,
                "Selected_Count": int(idx.size),
                "Total_Injected": int(inject_idx.size),
                "Detected_Injected": detected,
                "Detection_Rate_Percent": 100.0 * detected / inject_idx.size,
                "Raw_Final_Score_Threshold": threshold,
            }
        )
    return rows


def compare_clean_reference_arrays(
    clean_dir: Path,
    clean_outputs: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Optional checks against stored clean artifacts from the final lineage."""
    checks: list[dict] = []

    specs = (
        (
            "Pure normalized spatial score",
            clean_dir / "pure_spatial_score_normalized.npy",
            clean_outputs["pure_spatial"],
        ),
        (
            "Pure pre-final-clipping score",
            clean_dir / "pure_final_anomaly_score_raw.npy",
            clean_outputs["pure_final_raw"],
        ),
        (
            "Pure exact top-2% mask",
            clean_dir / "pure_final_anomaly_labels.npy",
            clean_outputs["pure_mask"],
        ),
        (
            "Full normalized spatial score",
            clean_dir / "spatial_score_normalized.npy",
            clean_outputs["full_spatial"],
        ),
    )

    for label, path, recomputed in specs:
        if not path.is_file():
            checks.append(
                {
                    "Check": label,
                    "Status": "REFERENCE_NOT_PRESENT",
                    "MaxAbsDiff": np.nan,
                    "ExactEqual": False,
                    "Reference_File": path.name,
                }
            )
            continue

        stored = np.load(path, mmap_mode="r", allow_pickle=False)
        stored = np.asarray(stored).reshape(-1)
        new = np.asarray(recomputed).reshape(-1)
        if stored.size != new.size:
            checks.append(
                {
                    "Check": label,
                    "Status": "LENGTH_MISMATCH",
                    "MaxAbsDiff": np.nan,
                    "ExactEqual": False,
                    "Reference_File": path.name,
                }
            )
            continue

        if stored.dtype == np.bool_ or np.issubdtype(stored.dtype, np.integer):
            exact = bool(np.array_equal(stored, new))
            maxdiff = float(np.max(np.abs(stored.astype(np.int64) - new.astype(np.int64))))
        else:
            exact = bool(np.array_equal(stored, new))
            maxdiff = float(
                np.max(
                    np.abs(
                        stored.astype(np.float64, copy=False)
                        - new.astype(np.float64, copy=False)
                    )
                )
            )
        checks.append(
            {
                "Check": label,
                "Status": "MATCH" if maxdiff <= 1e-6 else "MISMATCH",
                "MaxAbsDiff": maxdiff,
                "ExactEqual": exact,
                "Reference_File": path.name,
            }
        )

    return pd.DataFrame(checks)


def compare_existing_injected_summaries(
    result: pd.DataFrame,
    injected_root: Path,
) -> pd.DataFrame:
    parity_rows: list[dict] = []

    for ratio, folder_name in RATIO_FOLDERS.items():
        reference = (
            injected_root
            / folder_name
            / "Injected_Proposed_Detection_Rates_Raw_Final_Scores.csv"
        )
        if not reference.is_file():
            for method in ("Full", "Pure"):
                for budget in BUDGETS:
                    parity_rows.append(
                        {
                            "Injection_Ratio": ratio,
                            "Method": method,
                            "Budget_Fraction": budget,
                            "Reference_Mean_DR": np.nan,
                            "Recomputed_Mean_DR": np.nan,
                            "AbsDiff": np.nan,
                            "Status": "REFERENCE_NOT_PRESENT",
                        }
                    )
            continue

        ref = pd.read_csv(reference)
        ref_mean = (
            ref.groupby(["Method", "Budget_Fraction"], as_index=False)[
                "Detection_Rate_Percent"
            ]
            .mean()
        )
        new_mean = (
            result[result["Injection_Ratio"] == ratio]
            .groupby(["Method", "Budget_Fraction"], as_index=False)[
                "Detection_Rate_Percent"
            ]
            .mean()
        )
        merged = ref_mean.merge(
            new_mean,
            on=["Method", "Budget_Fraction"],
            suffixes=("_Reference", "_Recomputed"),
        )
        for _, row in merged.iterrows():
            ref_value = float(row["Detection_Rate_Percent_Reference"])
            new_value = float(row["Detection_Rate_Percent_Recomputed"])
            diff = abs(ref_value - new_value)
            parity_rows.append(
                {
                    "Injection_Ratio": ratio,
                    "Method": row["Method"],
                    "Budget_Fraction": float(row["Budget_Fraction"]),
                    "Reference_Mean_DR": ref_value,
                    "Recomputed_Mean_DR": new_value,
                    "AbsDiff": diff,
                    "Status": "MATCH" if diff < 5e-6 else "MISMATCH",
                }
            )

    return pd.DataFrame(parity_rows)


def headline_checks(mean_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (ratio, method, budget), expected in HEADLINE_EXPECTED.items():
        q = mean_df[
            (mean_df["Injection_Ratio"] == ratio)
            & (mean_df["Method"] == method)
            & np.isclose(mean_df["Budget_Fraction"], budget)
        ]
        observed = float(q.iloc[0]["Detection_Rate_Percent"]) if len(q) == 1 else np.nan
        diff = abs(observed - expected) if np.isfinite(observed) else np.nan
        rows.append(
            {
                "Injection_Ratio": ratio,
                "Method": method,
                "Budget_Fraction": budget,
                "Expected": expected,
                "Observed": observed,
                "AbsDiff": diff,
                "Status": "MATCH" if np.isfinite(diff) and diff < 5e-6 else "MISMATCH",
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clean reproduction of the reported Full/Pure post-training results"
    )
    parser.add_argument(
        "--project-root",
        required=True,
        type=Path,
        help="Original research Code directory containing saved score arrays",
    )
    parser.add_argument(
        "--valid-mask",
        type=Path,
        default=None,
        help="Optional explicit valid-mask path; valid_mask_1.npy is preferred",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Default: <project-root>/PUBLIC_REPRODUCTION_FULL_PURE",
    )
    parser.add_argument(
        "--save-arrays",
        action="store_true",
        help="Also save large clean/injected score arrays (off by default)",
    )
    args = parser.parse_args()

    project_root = args.project_root.expanduser().resolve()
    clean_dir = project_root / "03_Proposed_Results_Seed_42"
    injected_root = project_root / "03_Injected_Proposed_Results_Seed_42"
    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else project_root / "PUBLIC_REPRODUCTION_FULL_PURE"
    )

    if not clean_dir.is_dir():
        raise FileNotFoundError(clean_dir)
    if not injected_root.is_dir():
        raise FileNotFoundError(injected_root)
    output.mkdir(parents=True, exist_ok=True)

    print(f"Project root : {project_root}")
    print(f"Output       : {output}")

    valid_mask, valid_mask_path = resolve_valid_mask(project_root, args.valid_mask)
    print(f"Valid mask   : {valid_mask_path.name}")

    print("\n[1/4] Fit clean calibration and recompute clean Full/Pure scores...")
    components, context = load_component_scores(clean_dir)
    calibration, clean_outputs = score_clean(components, context, valid_mask)

    # Keep the saved metadata portable: no absolute user path is written.
    calibration_payload = {
        "valid_mask_file": valid_mask_path.name,
        **calibration.to_dict(),
    }
    (output / "calibration.json").write_text(
        json.dumps(calibration_payload, indent=2), encoding="utf-8"
    )

    clean_summary = {
        "n_samples": calibration.n_samples,
        "exact_top2_count": int(math.ceil(TOP_FRACTION * calibration.n_samples)),
        "full_raw_threshold": float(clean_outputs["full_threshold"]),
        "pure_raw_threshold": float(clean_outputs["pure_threshold"]),
        "alpha": ALPHA,
        "beta": BETA,
        "lambda_agreement": LAMBDA_AGREEMENT,
        "lambda_certainty": LAMBDA_CERTAINTY,
        "spatial_window": SPATIAL_WINDOW,
        "ranking_score": "pre-final-clipping refined_score_x_reliability",
        "pure_spatial_definition": (
            "RMS of locally centered AE residuals within each fixed neighborhood"
        ),
    }
    (output / "clean_summary.json").write_text(
        json.dumps(clean_summary, indent=2), encoding="utf-8"
    )

    clean_checks = compare_clean_reference_arrays(clean_dir, clean_outputs)
    clean_checks.to_csv(output / "clean_implementation_parity.csv", index=False)

    if args.save_arrays:
        arr_dir = output / "clean_arrays"
        arr_dir.mkdir(exist_ok=True)
        for key in (
            "full_spatial",
            "full_refined",
            "full_reliability",
            "full_final_raw",
            "full_final_normalized",
            "full_mask",
            "full_indices",
            "pure_spatial",
            "pure_refined",
            "pure_reliability",
            "pure_final_raw",
            "pure_final_normalized",
            "pure_mask",
            "pure_indices",
        ):
            np.save(arr_dir / f"{key}.npy", clean_outputs[key])

    print("[2/4] Recompute all 15 injected Full/Pure cases...")
    rows: list[dict] = []
    for ratio, folder_name in RATIO_FOLDERS.items():
        ratio_dir = injected_root / folder_name
        inject_idx = np.load(
            require_file(ratio_dir / "inject_indices.npy"), allow_pickle=False
        ).astype(np.int64, copy=False).reshape(-1)

        for scenario in SCENARIOS:
            scenario_dir = ratio_dir / scenario
            if not scenario_dir.is_dir():
                raise FileNotFoundError(scenario_dir)

            injected_components, injected_context = load_component_scores(scenario_dir)
            scored = score_injected(
                injected_components,
                injected_context,
                valid_mask,
                calibration,
            )
            rows.extend(
                evaluate_budgets(
                    scored["full_final_raw"],
                    inject_idx,
                    method="Full",
                    ratio=ratio,
                    scenario=scenario,
                )
            )
            rows.extend(
                evaluate_budgets(
                    scored["pure_final_raw"],
                    inject_idx,
                    method="Pure",
                    ratio=ratio,
                    scenario=scenario,
                )
            )

            if args.save_arrays:
                arr_dir = output / "injected_arrays" / folder_name / scenario
                arr_dir.mkdir(parents=True, exist_ok=True)
                for key in (
                    "full_final_raw",
                    "full_final_normalized",
                    "pure_final_raw",
                    "pure_final_normalized",
                ):
                    np.save(arr_dir / f"{key}.npy", scored[key])

            print(f"  done: {ratio} / {scenario}")

    result = pd.DataFrame(rows)
    result.to_csv(output / "full_pure_detection_results.csv", index=False)

    mean_df = (
        result.groupby(
            ["Injection_Ratio", "Method", "Budget_Fraction", "Budget_Percent"],
            as_index=False,
        )["Detection_Rate_Percent"]
        .mean()
        .sort_values(["Injection_Ratio", "Method", "Budget_Fraction"])
    )
    mean_df.to_csv(output / "full_pure_mean_detection_rates.csv", index=False)

    print("[3/4] Compare against the saved final result summaries...")
    parity = compare_existing_injected_summaries(result, injected_root)
    parity.to_csv(output / "parity_against_existing.csv", index=False)

    headlines = headline_checks(mean_df)
    headlines.to_csv(output / "headline_checks.csv", index=False)

    print("[4/4] Write validation report...")
    parity_bad = parity[parity["Status"] == "MISMATCH"]
    parity_matched = int((parity["Status"] == "MATCH").sum())
    parity_total = int((parity["Status"].isin(["MATCH", "MISMATCH"])).sum())

    clean_bad = clean_checks[clean_checks["Status"] == "MISMATCH"]
    headline_bad = headlines[headlines["Status"] == "MISMATCH"]

    report = {
        "status": (
            "PASS"
            if len(parity_bad) == 0 and len(clean_bad) == 0 and len(headline_bad) == 0
            else "CHECK_REQUIRED"
        ),
        "parity_matches": parity_matched,
        "parity_comparable_checks": parity_total,
        "parity_max_abs_diff": (
            float(parity.loc[parity["Status"] == "MATCH", "AbsDiff"].max())
            if parity_matched
            else None
        ),
        "clean_reference_mismatches": int(len(clean_bad)),
        "headline_mismatches": int(len(headline_bad)),
        "scope": (
            "post-training Full/Pure scoring, spatial refinement, reliability, "
            "deterministic ranking, and injected evaluation"
        ),
        "not_in_scope": (
            "MVAE/context-AE/detector retraining and raw-data feature engineering"
        ),
    }
    (output / "validation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    print("\nValidation report:")
    print(json.dumps(report, indent=2))
    print(f"\nOutputs written to: {output}")

    if report["status"] != "PASS":
        raise SystemExit(
            "Validation did not fully pass. Inspect parity_against_existing.csv, "
            "clean_implementation_parity.csv and headline_checks.csv."
        )


if __name__ == "__main__":
    main()
