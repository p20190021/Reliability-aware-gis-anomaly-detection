#!/usr/bin/env python3
"""
Canonical reproduction (verified final) of the FINAL Full/Pure post-training pipeline used in:

  Reliability-Aware Multi-View Anomaly Detection for Quality Assurance of
  Heterogeneous GIS Data Pipelines

Verification status
-------------------
Verified on the final experiment outputs: all 30 Full/Pure injection-ratio x
inspection-budget parity checks matched the reference results, with maximum
absolute numerical difference 1.4210854715202004e-14.

Purpose
-------
This script consolidates the final, corrected score-fusion logic that produced
reported Full/Pure results. It deliberately does NOT retrain the multi-view AE,
context AE, IF, LOF, or Mahalanobis models. Those upstream stages produce the
normalized component score arrays consumed here.

The script has two useful modes:

1) --verify-existing
   Lightweight check of the already-generated CSV summaries against the
   manuscript's locked headline values. This does not need large .npy arrays.

2) --recompute
   Recomputes clean calibration and all 15 injected Full/Pure experiments from
   the saved normalized component score arrays, using a separate output folder.
   Existing research outputs are never overwritten.

Important provenance note
-------------------------
The final manuscript results used a corrected Pure branch in which the Pure
spatial signal is derived from the AE reconstruction field rather than reusing
Full's spatial signal. The exact Pure spatial implementation below is preserved
from the final correction run so that the reported numerical results are
reproducible. Full preserves the spatial implementation used by the original
Full pipeline.

Expected project layout
-----------------------
<root>/
  valid_mask_1.npy  # canonical mask; valid_mask.npy is legacy fallback
  03_Proposed_Results_Seed_42/
    ae_score_normalized.npy
    if_feature_score_normalized.npy
    lof_feature_score_normalized.npy
    md_feature_score_normalized.npy
    if_embedding_score_normalized.npy
    lof_embedding_score_normalized.npy
    md_embedding_score_normalized.npy
    context_score_normalized.npy
  03_Injected_Proposed_Results_Seed_42/
    Injected_Datasets_with_0.05%_Injection_Rate/
      inject_indices.npy
      extreme/ ... component score arrays ...
      noise_strong/
      noise_moderate/
      noise_mild/
      inconsistent/
    Injected_Datasets_with_0.1%_Injection_Rate/
    Injected_Datasets_with_0.5%_Injection_Rate/

Example
-------
python canonical_full_pure_reproduction.py --root "C:/.../Outlier Paper/Code" --verify-existing
python canonical_full_pure_reproduction.py --root "C:/.../Outlier Paper/Code" --recompute

The recompute mode writes only to:
  <root>/REPRODUCIBILITY_CANONICAL/
unless --output is supplied.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter


# -----------------------------------------------------------------------------
# Locked manuscript configuration
# -----------------------------------------------------------------------------
TOP_FRACTION = 0.02
BUDGETS = (0.001, 0.0025, 0.005, 0.01, 0.02)
ALPHA = 0.75
BETA = 0.25
LAMBDA_AGREEMENT = 0.60
LAMBDA_CERTAINTY = 0.40
SPATIAL_WINDOW = 5
EPS = 1e-6
EXPECTED_MASK_SHAPE = (2897, 2696)
EXPECTED_N_SAMPLES = 4_290_958

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

FULL_COMPONENT_FILES = {
    "AE": "ae_score_normalized.npy",
    "IF_feature": "if_feature_score_normalized.npy",
    "LOF_feature": "lof_feature_score_normalized.npy",
    "MD_feature": "md_feature_score_normalized.npy",
    "IF_embedding": "if_embedding_score_normalized.npy",
    "LOF_embedding": "lof_embedding_score_normalized.npy",
    "MD_embedding": "md_embedding_score_normalized.npy",
}
CONTEXT_FILE = "context_score_normalized.npy"

# Headline values locked in the submitted manuscript. These are only used for
# lightweight verification of existing summary CSVs.
MANUSCRIPT_HEADLINES = {
    ("0.1%", "Full", 0.001): 82.11655011655012,
    ("0.1%", "Pure", 0.001): 86.53146853146853,
    ("0.1%", "Full", 0.020): 99.03962703962704,
    ("0.1%", "Pure", 0.020): 98.42424242424242,
}


# -----------------------------------------------------------------------------
# Data structures
# -----------------------------------------------------------------------------
@dataclass
class VariantCalibration:
    spatial_p1: float
    spatial_p99: float
    refined_p1: float
    refined_p99: float
    uncertainty_min: float
    uncertainty_max: float
    final_p1: float
    final_p99: float


@dataclass
class CalibrationBundle:
    full: VariantCalibration
    pure: VariantCalibration
    n_samples: int
    top_fraction: float = TOP_FRACTION
    alpha: float = ALPHA
    beta: float = BETA
    lambda_agreement: float = LAMBDA_AGREEMENT
    lambda_certainty: float = LAMBDA_CERTAINTY
    spatial_window: int = SPATIAL_WINDOW


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------
def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Required file is missing: {path}")
    return path


def load_vector(path: Path, *, mmap: bool = True) -> np.ndarray:
    arr = np.load(
        require_file(path),
        mmap_mode="r" if mmap else None,
        allow_pickle=False,
    )
    arr = np.asarray(arr).reshape(-1)
    if not np.isfinite(arr).all():
        raise ValueError(f"Non-finite values found in {path}")
    return arr.astype(np.float32, copy=False)


def robust_bounds(x: np.ndarray) -> tuple[float, float]:
    p1, p99 = np.percentile(x, [1.0, 99.0])
    p1, p99 = float(p1), float(p99)
    if not np.isfinite(p1) or not np.isfinite(p99) or p99 <= p1:
        raise ValueError(f"Invalid robust bounds: p1={p1}, p99={p99}")
    return p1, p99


def normalize_robust(x: np.ndarray, p1: float, p99: float) -> np.ndarray:
    if p99 <= p1:
        raise ValueError("p99 must exceed p1")
    y = (np.clip(x, p1, p99) - p1) / (p99 - p1)
    return np.asarray(y, dtype=np.float32)


def normalize_minmax(x: np.ndarray, xmin: float, xmax: float) -> np.ndarray:
    if xmax <= xmin:
        return np.zeros(np.asarray(x).shape, dtype=np.float32)
    y = (x - xmin) / (xmax - xmin)
    return np.clip(y, 0.0, 1.0).astype(np.float32, copy=False)


def exact_top_k(
    scores: np.ndarray,
    *,
    fraction: float | None = None,
    k: int | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Deterministic exact top-k; lower original index breaks boundary ties."""
    s = np.asarray(scores).reshape(-1)
    if not np.isfinite(s).all():
        raise ValueError("Scores contain NaN/Inf")
    n = s.size
    if k is None:
        if fraction is None or not (0.0 < fraction <= 1.0):
            raise ValueError("Supply k or a fraction in (0, 1]")
        k = int(math.ceil(fraction * n))
    if not (1 <= k <= n):
        raise ValueError(f"Invalid k={k} for n={n}")

    threshold = float(np.partition(s, n - k)[n - k])
    selected = s > threshold
    need = k - int(selected.sum())
    if need > 0:
        ties = np.flatnonzero(s == threshold)  # ascending original index
        if ties.size < need:
            raise RuntimeError("Insufficient boundary ties")
        selected[ties[:need]] = True
    idx = np.flatnonzero(selected).astype(np.int64, copy=False)
    if idx.size != k:
        raise RuntimeError(f"Exact top-k failed: expected {k}, got {idx.size}")
    return selected.astype(np.uint8, copy=False), idx, threshold


def find_valid_mask(root: Path, clean_dir: Path) -> tuple[np.ndarray, Path]:
    """Load the canonical raster-validity mask used by the final pipeline.

    Provenance audit shows that ``01_feature_engineering.ipynb`` writes
    ``valid_mask_1.npy`` and the final SSF-AE / Full-without-IF pipelines use
    that file explicitly.  A legacy ``valid_mask.npy`` may coexist in the
    project root; its presence must therefore not make mask discovery
    ambiguous.  ``valid_mask_1.npy`` is preferred deterministically, with the
    legacy name used only as a fallback for older copies of the project.
    """
    preferred = [
        root / "valid_mask_1.npy",
        clean_dir / "valid_mask_1.npy",
    ]
    fallback = [
        root / "valid_mask.npy",
        clean_dir / "valid_mask.npy",
    ]

    mask_path = next((p for p in preferred if p.is_file()), None)
    if mask_path is None:
        mask_path = next((p for p in fallback if p.is_file()), None)

    if mask_path is None:
        # Last-resort recursive lookup.  Prefer the canonical filename.
        preferred_found = sorted(root.glob("**/valid_mask_1.npy"))
        fallback_found = sorted(root.glob("**/valid_mask.npy"))
        found = preferred_found if preferred_found else fallback_found
        if len(found) != 1:
            raise FileNotFoundError(
                "Could not identify the canonical valid mask. Candidates: "
                + ", ".join(str(p) for p in found)
            )
        mask_path = found[0]

    mask = np.load(mask_path, allow_pickle=False).astype(bool, copy=False)
    if mask.ndim != 2:
        raise ValueError(f"Valid mask must be 2-D: {mask_path}; found {mask.shape}")
    if tuple(mask.shape) != EXPECTED_MASK_SHAPE:
        raise ValueError(
            f"Unexpected valid-mask shape {mask.shape}; expected {EXPECTED_MASK_SHAPE}. "
            f"File: {mask_path}"
        )
    n_valid = int(mask.sum())
    if n_valid != EXPECTED_N_SAMPLES:
        raise ValueError(
            f"{mask_path.name} has {n_valid:,} valid cells; "
            f"expected {EXPECTED_N_SAMPLES:,}."
        )

    # If the legacy root-level alias also exists, report whether it is identical
    # without allowing it to override the audited canonical file.
    legacy = root / "valid_mask.npy"
    canonical = root / "valid_mask_1.npy"
    if mask_path.resolve() == canonical.resolve() and legacy.is_file():
        try:
            legacy_mask = np.load(legacy, allow_pickle=False).astype(bool, copy=False)
            relation = "identical" if (legacy_mask.shape == mask.shape and np.array_equal(legacy_mask, mask)) else "DIFFERENT"
            print(
                f"Valid mask: using canonical {canonical.name}; "
                f"legacy {legacy.name} also exists and is {relation}."
            )
        except Exception as exc:
            print(
                f"Valid mask: using canonical {canonical.name}; "
                f"could not compare legacy {legacy.name}: {exc}"
            )
    else:
        print(f"Valid mask: {mask_path}")

    return mask, mask_path


def load_component_scores(folder: Path) -> tuple[dict[str, np.ndarray], np.ndarray]:
    comps = {name: load_vector(folder / filename) for name, filename in FULL_COMPONENT_FILES.items()}
    context = load_vector(folder / CONTEXT_FILE)
    lengths = {arr.size for arr in [*comps.values(), context]}
    if len(lengths) != 1:
        raise ValueError(f"Component length mismatch in {folder}: {sorted(lengths)}")
    return comps, context


def average_signals(signals: Iterable[np.ndarray]) -> np.ndarray:
    seq = list(signals)
    if not seq:
        raise ValueError("No signals supplied")
    out = np.zeros(seq[0].shape, dtype=np.float32)
    for s in seq:
        out += np.asarray(s, dtype=np.float32)
    out /= np.float32(len(seq))
    return out


# -----------------------------------------------------------------------------
# Spatial implementations preserved from final result lineage
# -----------------------------------------------------------------------------
def spatial_full(base: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Full spatial inconsistency used by the original final Full pipeline."""
    n_valid = int(valid_mask.sum())
    if base.size != n_valid:
        raise ValueError(f"Full base has {base.size:,} rows, mask has {n_valid:,}")

    raster = np.full(valid_mask.shape, np.nan, dtype=np.float32)
    raster[valid_mask] = np.asarray(base, dtype=np.float32)
    valid = np.isfinite(raster)
    kernel_area = float(SPATIAL_WINDOW * SPATIAL_WINDOW)

    def nan_local_mean(arr: np.ndarray) -> np.ndarray:
        ok = np.isfinite(arr)
        filled = np.where(ok, arr, 0.0).astype(np.float32)
        local_sum = uniform_filter(filled, size=SPATIAL_WINDOW, mode="nearest") * kernel_area
        local_count = uniform_filter(ok.astype(np.float32), size=SPATIAL_WINDOW, mode="nearest") * kernel_area
        return local_sum / np.maximum(local_count, 1.0)

    local_mean = nan_local_mean(raster)
    second = np.where(valid, raster ** 2, np.nan)
    local_second = nan_local_mean(second)
    local_var = np.maximum(local_second - local_mean ** 2, 0.0)
    local_std = np.sqrt(local_var)
    score = np.abs(raster - local_mean) / (local_std + EPS)
    out = score[valid_mask].astype(np.float32, copy=False)
    if not np.isfinite(out).all():
        raise ValueError("Full spatial score contains NaN/Inf")
    return out


def spatial_pure_final_reference(base: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """
    Pure-specific spatial implementation used by the final correction run that
    generated the reported Pure results. It is intentionally preserved exactly
    for numerical reproducibility.
    """
    n_valid = int(valid_mask.sum())
    if base.size != n_valid:
        raise ValueError(f"Pure base has {base.size:,} rows, mask has {n_valid:,}")

    raster = np.full(valid_mask.shape, np.nan, dtype=np.float32)
    raster[valid_mask] = np.asarray(base, dtype=np.float32)
    valid_float = valid_mask.astype(np.float32, copy=False)
    filled = np.where(valid_mask, raster, 0.0).astype(np.float32, copy=False)
    kernel_area = float(SPATIAL_WINDOW * SPATIAL_WINDOW)

    local_sum = uniform_filter(filled, size=SPATIAL_WINDOW, mode="nearest") * kernel_area
    local_count = uniform_filter(valid_float, size=SPATIAL_WINDOW, mode="nearest") * kernel_area
    local_mean = local_sum / np.maximum(local_count, 1.0)

    diff_sq = (raster - local_mean) ** 2
    diff_sq = np.where(np.isnan(diff_sq), 0.0, diff_sq).astype(np.float32, copy=False)
    local_var_sum = uniform_filter(diff_sq, size=SPATIAL_WINDOW, mode="nearest") * kernel_area
    local_var = local_var_sum / np.maximum(local_count, 1.0)
    local_std = np.sqrt(np.maximum(local_var, 0.0)).astype(np.float32, copy=False)

    score = np.abs(raster - local_mean) / (local_std + EPS)
    out = score[valid_mask].astype(np.float32, copy=False)
    if not np.isfinite(out).all():
        raise ValueError("Pure spatial score contains NaN/Inf")
    return out


# -----------------------------------------------------------------------------
# Reliability and variant scoring
# -----------------------------------------------------------------------------
def reliability_clean(signals: list[np.ndarray]) -> tuple[np.ndarray, float, float, np.ndarray, np.ndarray]:
    if not signals:
        raise ValueError("No reliability signals")
    n = signals[0].size
    agreement = np.zeros(n, dtype=np.uint8)
    for signal in signals:
        flag, _, _ = exact_top_k(signal, fraction=TOP_FRACTION)
        agreement += flag
    agreement_fraction = agreement.astype(np.float32) / np.float32(len(signals))

    stack = np.stack(signals, axis=0).astype(np.float32, copy=False)
    uncertainty_raw = np.std(stack, axis=0, dtype=np.float32).astype(np.float32, copy=False)
    umin, umax = float(uncertainty_raw.min()), float(uncertainty_raw.max())
    uncertainty_norm = normalize_minmax(uncertainty_raw, umin, umax)
    reliability = (
        np.float32(LAMBDA_AGREEMENT) * agreement_fraction
        + np.float32(LAMBDA_CERTAINTY) * (np.float32(1.0) - uncertainty_norm)
    )
    reliability = np.clip(reliability, 0.0, 1.0).astype(np.float32, copy=False)
    return reliability, umin, umax, agreement, uncertainty_norm


def reliability_apply(signals: list[np.ndarray], umin: float, umax: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = signals[0].size
    agreement = np.zeros(n, dtype=np.uint8)
    for signal in signals:
        flag, _, _ = exact_top_k(signal, fraction=TOP_FRACTION)
        agreement += flag
    agreement_fraction = agreement.astype(np.float32) / np.float32(len(signals))

    stack = np.stack(signals, axis=0).astype(np.float32, copy=False)
    uncertainty_raw = np.std(stack, axis=0, dtype=np.float32).astype(np.float32, copy=False)
    uncertainty_norm = normalize_minmax(uncertainty_raw, umin, umax)
    reliability = (
        np.float32(LAMBDA_AGREEMENT) * agreement_fraction
        + np.float32(LAMBDA_CERTAINTY) * (np.float32(1.0) - uncertainty_norm)
    )
    reliability = np.clip(reliability, 0.0, 1.0).astype(np.float32, copy=False)
    return reliability, agreement, uncertainty_norm


def score_clean(
    comps: dict[str, np.ndarray],
    context: np.ndarray,
    valid_mask: np.ndarray,
) -> tuple[CalibrationBundle, dict[str, np.ndarray]]:
    ae = comps["AE"]
    n = ae.size
    if int(valid_mask.sum()) != n:
        raise ValueError(f"Mask count {int(valid_mask.sum()):,} != score count {n:,}")

    # ----- Full -----
    full_base = average_signals(comps.values())
    full_spatial_raw = spatial_full(full_base, valid_mask)
    fs1, fs99 = robust_bounds(full_spatial_raw)
    full_spatial = normalize_robust(full_spatial_raw, fs1, fs99)

    full_ref_raw = (np.float32(ALPHA) * full_base + np.float32(BETA) * full_spatial).astype(np.float32)
    fr1, fr99 = robust_bounds(full_ref_raw)
    full_ref = normalize_robust(full_ref_raw, fr1, fr99)

    full_rel_signals = [*comps.values(), full_spatial, context]
    full_rel, fu_min, fu_max, full_agreement, full_uncertainty = reliability_clean(full_rel_signals)
    full_final_raw = (full_ref * full_rel).astype(np.float32, copy=False)
    ff1, ff99 = robust_bounds(full_final_raw)
    full_final_norm = normalize_robust(full_final_raw, ff1, ff99)
    full_mask, full_idx, full_threshold = exact_top_k(full_final_raw, fraction=TOP_FRACTION)

    # ----- Pure (corrected branch) -----
    pure_base = ae
    pure_spatial_raw = spatial_pure_final_reference(pure_base, valid_mask)
    ps1, ps99 = robust_bounds(pure_spatial_raw)
    pure_spatial = normalize_robust(pure_spatial_raw, ps1, ps99)

    pure_ref_raw = (np.float32(ALPHA) * pure_base + np.float32(BETA) * pure_spatial).astype(np.float32)
    pr1, pr99 = robust_bounds(pure_ref_raw)
    pure_ref = normalize_robust(pure_ref_raw, pr1, pr99)

    pure_rel_signals = [ae, pure_spatial, context]
    pure_rel, pu_min, pu_max, pure_agreement, pure_uncertainty = reliability_clean(pure_rel_signals)
    pure_final_raw = (pure_ref * pure_rel).astype(np.float32, copy=False)
    pf1, pf99 = robust_bounds(pure_final_raw)
    pure_final_norm = normalize_robust(pure_final_raw, pf1, pf99)
    pure_mask, pure_idx, pure_threshold = exact_top_k(pure_final_raw, fraction=TOP_FRACTION)

    cal = CalibrationBundle(
        full=VariantCalibration(fs1, fs99, fr1, fr99, fu_min, fu_max, ff1, ff99),
        pure=VariantCalibration(ps1, ps99, pr1, pr99, pu_min, pu_max, pf1, pf99),
        n_samples=n,
    )
    out = {
        "full_base": full_base,
        "full_spatial": full_spatial,
        "full_refined": full_ref,
        "full_reliability": full_rel,
        "full_agreement": full_agreement,
        "full_uncertainty": full_uncertainty,
        "full_final_raw": full_final_raw,
        "full_final_normalized": full_final_norm,
        "full_mask": full_mask,
        "full_indices": full_idx,
        "full_threshold": np.asarray(full_threshold),
        "pure_base": pure_base,
        "pure_spatial": pure_spatial,
        "pure_refined": pure_ref,
        "pure_reliability": pure_rel,
        "pure_agreement": pure_agreement,
        "pure_uncertainty": pure_uncertainty,
        "pure_final_raw": pure_final_raw,
        "pure_final_normalized": pure_final_norm,
        "pure_mask": pure_mask,
        "pure_indices": pure_idx,
        "pure_threshold": np.asarray(pure_threshold),
    }
    return cal, out


def score_injected(
    comps: dict[str, np.ndarray],
    context: np.ndarray,
    valid_mask: np.ndarray,
    cal: CalibrationBundle,
) -> dict[str, np.ndarray]:
    ae = comps["AE"]
    if ae.size != cal.n_samples:
        raise ValueError(f"Injected score length {ae.size:,} != clean {cal.n_samples:,}")

    # Full: clean calibration is frozen.
    full_base = average_signals(comps.values())
    full_spatial_raw = spatial_full(full_base, valid_mask)
    full_spatial = normalize_robust(full_spatial_raw, cal.full.spatial_p1, cal.full.spatial_p99)
    full_ref_raw = (np.float32(ALPHA) * full_base + np.float32(BETA) * full_spatial).astype(np.float32)
    full_ref = normalize_robust(full_ref_raw, cal.full.refined_p1, cal.full.refined_p99)
    full_rel, full_agreement, full_uncertainty = reliability_apply(
        [*comps.values(), full_spatial, context], cal.full.uncertainty_min, cal.full.uncertainty_max
    )
    full_final_raw = (full_ref * full_rel).astype(np.float32, copy=False)
    full_final_norm = normalize_robust(full_final_raw, cal.full.final_p1, cal.full.final_p99)

    # Pure: corrected Pure-specific spatial field; clean calibration frozen.
    pure_spatial_raw = spatial_pure_final_reference(ae, valid_mask)
    pure_spatial = normalize_robust(pure_spatial_raw, cal.pure.spatial_p1, cal.pure.spatial_p99)
    pure_ref_raw = (np.float32(ALPHA) * ae + np.float32(BETA) * pure_spatial).astype(np.float32)
    pure_ref = normalize_robust(pure_ref_raw, cal.pure.refined_p1, cal.pure.refined_p99)
    pure_rel, pure_agreement, pure_uncertainty = reliability_apply(
        [ae, pure_spatial, context], cal.pure.uncertainty_min, cal.pure.uncertainty_max
    )
    pure_final_raw = (pure_ref * pure_rel).astype(np.float32, copy=False)
    pure_final_norm = normalize_robust(pure_final_raw, cal.pure.final_p1, cal.pure.final_p99)

    return {
        "full_final_raw": full_final_raw,
        "full_final_normalized": full_final_norm,
        "full_spatial": full_spatial,
        "full_reliability": full_rel,
        "full_agreement": full_agreement,
        "full_uncertainty": full_uncertainty,
        "pure_final_raw": pure_final_raw,
        "pure_final_normalized": pure_final_norm,
        "pure_spatial": pure_spatial,
        "pure_reliability": pure_rel,
        "pure_agreement": pure_agreement,
        "pure_uncertainty": pure_uncertainty,
    }


def evaluate_budgets(scores: np.ndarray, inject_idx: np.ndarray, method: str, ratio: str, scenario: str) -> list[dict]:
    rows = []
    inject_idx = np.asarray(inject_idx, dtype=np.int64).reshape(-1)
    for budget in BUDGETS:
        mask, idx, threshold = exact_top_k(scores, fraction=budget)
        detected = int(mask[inject_idx].sum())
        total = int(inject_idx.size)
        rows.append({
            "Injection_Ratio": ratio,
            "Scenario": scenario,
            "Method": method,
            "Budget_Fraction": budget,
            "Budget_Percent": budget * 100.0,
            "Selected_Count": int(idx.size),
            "Total_Injected": total,
            "Detected_Injected": detected,
            "Detection_Rate_Percent": 100.0 * detected / total,
            "Raw_Final_Score_Threshold": threshold,
        })
    return rows


# -----------------------------------------------------------------------------
# Existing-result verification (lightweight; no large arrays required)
# -----------------------------------------------------------------------------
def verify_existing(root: Path) -> pd.DataFrame:
    rows = []
    proposed_root = root / "03_Injected_Proposed_Results_Seed_42"

    # Verify all saved proposed mean curves and the manuscript headline values.
    for ratio, folder_name in RATIO_FOLDERS.items():
        csv_path = proposed_root / folder_name / "Injected_Proposed_Detection_Rates_Raw_Final_Scores.csv"
        if not csv_path.is_file():
            rows.append({"Check": f"Proposed summary {ratio}", "Status": "MISSING", "Observed": np.nan, "Expected": np.nan, "AbsDiff": np.nan})
            continue
        df = pd.read_csv(csv_path)
        for method in ("Full", "Pure"):
            for budget in BUDGETS:
                sel = df[(df["Method"] == method) & np.isclose(df["Budget_Fraction"], budget)]
                if len(sel) != len(SCENARIOS):
                    status = f"BAD_ROWS({len(sel)})"
                    obs = np.nan
                else:
                    obs = float(sel["Detection_Rate_Percent"].mean())
                    status = "OK"
                exp = MANUSCRIPT_HEADLINES.get((ratio, method, budget), np.nan)
                diff = abs(obs - exp) if np.isfinite(obs) and np.isfinite(exp) else np.nan
                if np.isfinite(diff) and diff > 5e-6:
                    status = "MISMATCH"
                rows.append({
                    "Check": f"{ratio} {method} mean DR @ {budget*100:g}%",
                    "Status": status,
                    "Observed": obs,
                    "Expected": exp,
                    "AbsDiff": diff,
                })

    # Verify the main ablation top-2 values if available.
    ablation = root / "06_Ablation_Study_Results_Seed_42" / "module_ablation_top2.csv"
    if ablation.is_file():
        adf = pd.read_csv(ablation)
        expected_top2 = {
            "Full": 99.03962703962704,
            "Full base": 93.52447552447553,
            "Full without spatial": 97.23543123543122,
            "Full without context": 98.45221445221445,
            "Full without reliability": 98.82051282051282,
            "Pure": 98.42424242424242,
            "Pure base": 94.02331002331002,
            "Pure without spatial": 95.31468531468530,
            "Pure without context": 97.52447552447552,
            "Pure without reliability": 96.01864801864801,
        }
        for variant, exp in expected_top2.items():
            q = adf[adf["Variant"] == variant]
            obs = float(q.iloc[0]["Mean"]) if len(q) == 1 else np.nan
            diff = abs(obs - exp) if np.isfinite(obs) else np.nan
            rows.append({
                "Check": f"Ablation top-2 mean: {variant}",
                "Status": "OK" if np.isfinite(diff) and diff < 5e-6 else "MISMATCH",
                "Observed": obs,
                "Expected": exp,
                "AbsDiff": diff,
            })

    # Verify exact-raw five-seed stability values used by the manuscript.
    stability = root / "07_Stability_Analysis_ExactRaw_Audit" / "Stability_Summary_ExactRaw.csv"
    if stability.is_file():
        sdf = pd.read_csv(stability)
        expected_j = {"Full": 0.481710, "Pure": 0.421058}
        for method, exp in expected_j.items():
            q = sdf[sdf["Method"] == method]
            obs = float(q.iloc[0]["Mean_Jaccard"]) if len(q) == 1 else np.nan
            diff = abs(obs - exp) if np.isfinite(obs) else np.nan
            rows.append({
                "Check": f"Five-seed exact-raw mean Jaccard: {method}",
                "Status": "OK" if np.isfinite(diff) and diff < 5e-6 else "MISMATCH",
                "Observed": obs,
                "Expected": exp,
                "AbsDiff": diff,
            })

    # Verify SSF-AE headline baseline values if present.
    ssf = root / "SSFAE" / "results" / "final_injected_evaluation" / "SSFAE_Mean_Detection_Rates_By_Ratio.csv"
    if ssf.is_file():
        ssf_df = pd.read_csv(ssf)
        for budget, exp in ((0.001, 82.0), (0.02, 99.8881118881119)):
            q = ssf_df[(ssf_df["Injection_Ratio"] == "0.1%") & np.isclose(ssf_df["Budget_Fraction"], budget)]
            obs = float(q.iloc[0]["Mean_Detection_Rate_Percent"]) if len(q) == 1 else np.nan
            diff = abs(obs - exp) if np.isfinite(obs) else np.nan
            rows.append({
                "Check": f"SSF-AE 0.1% mean DR @ {budget*100:g}%",
                "Status": "OK" if np.isfinite(diff) and diff < 5e-6 else "MISMATCH",
                "Observed": obs,
                "Expected": exp,
                "AbsDiff": diff,
            })

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# End-to-end post-training recomputation from saved normalized component scores
# -----------------------------------------------------------------------------
def recompute(root: Path, output: Path, save_arrays: bool = False) -> tuple[pd.DataFrame, CalibrationBundle]:
    clean_dir = root / "03_Proposed_Results_Seed_42"
    injected_root = root / "03_Injected_Proposed_Results_Seed_42"
    if not clean_dir.is_dir():
        raise FileNotFoundError(clean_dir)
    if not injected_root.is_dir():
        raise FileNotFoundError(injected_root)

    output.mkdir(parents=True, exist_ok=True)
    valid_mask, mask_path = find_valid_mask(root, clean_dir)
    comps, context = load_component_scores(clean_dir)
    cal, clean_out = score_clean(comps, context, valid_mask)

    with open(output / "canonical_calibration.json", "w", encoding="utf-8") as f:
        json.dump({
            "valid_mask": str(mask_path),
            **asdict(cal),
        }, f, indent=2)

    clean_summary = {
        "N": cal.n_samples,
        "Top2_Count": int(clean_out["full_indices"].size),
        "Full_Final_Raw_Threshold": float(clean_out["full_threshold"]),
        "Pure_Final_Raw_Threshold": float(clean_out["pure_threshold"]),
    }
    with open(output / "clean_summary.json", "w", encoding="utf-8") as f:
        json.dump(clean_summary, f, indent=2)

    if save_arrays:
        clean_arrays = output / "clean_arrays"
        clean_arrays.mkdir(exist_ok=True)
        for key in (
            "full_spatial", "full_refined", "full_reliability", "full_final_raw", "full_final_normalized", "full_mask", "full_indices",
            "pure_spatial", "pure_refined", "pure_reliability", "pure_final_raw", "pure_final_normalized", "pure_mask", "pure_indices",
        ):
            np.save(clean_arrays / f"{key}.npy", clean_out[key])

    rows: list[dict] = []
    for ratio, folder_name in RATIO_FOLDERS.items():
        ratio_dir = injected_root / folder_name
        inject_idx = np.load(require_file(ratio_dir / "inject_indices.npy"), allow_pickle=False).astype(np.int64, copy=False).reshape(-1)
        for scenario in SCENARIOS:
            scenario_dir = ratio_dir / scenario
            if not scenario_dir.is_dir():
                raise FileNotFoundError(scenario_dir)
            icomps, icontext = load_component_scores(scenario_dir)
            scored = score_injected(icomps, icontext, valid_mask, cal)
            rows.extend(evaluate_budgets(scored["full_final_raw"], inject_idx, "Full", ratio, scenario))
            rows.extend(evaluate_budgets(scored["pure_final_raw"], inject_idx, "Pure", ratio, scenario))

            if save_arrays:
                arr_dir = output / "injected_arrays" / folder_name / scenario
                arr_dir.mkdir(parents=True, exist_ok=True)
                for key in ("full_final_raw", "full_final_normalized", "pure_final_raw", "pure_final_normalized"):
                    np.save(arr_dir / f"{key}.npy", scored[key])

    result = pd.DataFrame(rows)
    result.to_csv(output / "canonical_injected_detection_results.csv", index=False)
    mean_df = (
        result.groupby(["Injection_Ratio", "Method", "Budget_Fraction", "Budget_Percent"], as_index=False)["Detection_Rate_Percent"]
        .mean()
        .sort_values(["Injection_Ratio", "Method", "Budget_Fraction"])
    )
    mean_df.to_csv(output / "canonical_mean_detection_rates.csv", index=False)

    # Compare recomputed values against existing saved final summaries.
    parity_rows = []
    for ratio, folder_name in RATIO_FOLDERS.items():
        reference = injected_root / folder_name / "Injected_Proposed_Detection_Rates_Raw_Final_Scores.csv"
        if not reference.is_file():
            continue
        ref = pd.read_csv(reference)
        ref_mean = ref.groupby(["Method", "Budget_Fraction"], as_index=False)["Detection_Rate_Percent"].mean()
        new_mean = result[result["Injection_Ratio"] == ratio].groupby(["Method", "Budget_Fraction"], as_index=False)["Detection_Rate_Percent"].mean()
        merged = ref_mean.merge(new_mean, on=["Method", "Budget_Fraction"], suffixes=("_Reference", "_Recomputed"))
        for _, r in merged.iterrows():
            diff = abs(float(r["Detection_Rate_Percent_Reference"]) - float(r["Detection_Rate_Percent_Recomputed"]))
            parity_rows.append({
                "Injection_Ratio": ratio,
                "Method": r["Method"],
                "Budget_Fraction": float(r["Budget_Fraction"]),
                "Reference_Mean_DR": float(r["Detection_Rate_Percent_Reference"]),
                "Recomputed_Mean_DR": float(r["Detection_Rate_Percent_Recomputed"]),
                "AbsDiff": diff,
                "Status": "MATCH" if diff < 5e-6 else "MISMATCH",
            })
    parity = pd.DataFrame(parity_rows)
    parity.to_csv(output / "canonical_parity_against_existing.csv", index=False)
    return result, cal


def main() -> None:
    parser = argparse.ArgumentParser(description="Canonical Full/Pure reproduction and verification")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Project Code directory")
    parser.add_argument("--output", type=Path, default=None, help="Output directory (default: <root>/REPRODUCIBILITY_CANONICAL)")
    parser.add_argument("--verify-existing", action="store_true", help="Verify lightweight saved summaries")
    parser.add_argument("--recompute", action="store_true", help="Recompute Full/Pure from saved normalized component scores")
    parser.add_argument("--save-arrays", action="store_true", help="Also save large recomputed score arrays")
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    output = (args.output.expanduser().resolve() if args.output else root / "REPRODUCIBILITY_CANONICAL")

    if not args.verify_existing and not args.recompute:
        args.verify_existing = True

    print(f"Project root: {root}")

    if args.verify_existing:
        report = verify_existing(root)
        output.mkdir(parents=True, exist_ok=True)
        report_path = output / "existing_results_verification.csv"
        report.to_csv(report_path, index=False)
        print("\nExisting-result verification:")
        print(report.to_string(index=False))
        print(f"\nSaved: {report_path}")
        bad = report[report["Status"].isin(["MISMATCH", "MISSING"])]
        if len(bad):
            print(f"WARNING: {len(bad)} verification checks require attention.")
        else:
            print("All available locked checks passed.")

    if args.recompute:
        _, cal = recompute(root, output, save_arrays=args.save_arrays)
        print("\nRecomputation complete.")
        print(f"N={cal.n_samples:,}; exact top-2% k={math.ceil(TOP_FRACTION * cal.n_samples):,}")
        print(f"Outputs: {output}")


if __name__ == "__main__":
    main()
