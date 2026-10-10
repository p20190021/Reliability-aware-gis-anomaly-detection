"""
08_recompute_corrected_pure_outputs.py
=====================================

Purpose
-------
Recompute ONLY the Pure branch after correcting the architectural leakage in
which Pure reused the Full-derived spatial inconsistency score.

Corrected Pure definition
-------------------------
    AE normalized reconstruction score
        -> Pure-specific spatial inconsistency computed from AE itself
        -> 0.75 * AE + 0.25 * Pure spatial inconsistency
        -> Pure reliability from [AE, Pure spatial inconsistency, Context]
        -> final Pure score
        -> deterministic exact top-2% mask

Locked weights
--------------
    Full: 0.75 base + 0.25 spatial   (NOT changed by this script)
    Pure: 0.75 AE   + 0.25 spatial   (aligned with Full to avoid an unnecessary variant-specific hyperparameter)
    Reliability: 0.60 agreement + 0.40 certainty

What is NOT recomputed
----------------------
- Multi-view AE training
- Context AE training
- IF / LOF / MD training or scoring
- Any baseline
- Any Full score or Full mask

What IS recomputed
------------------
1. Main clean Pure outputs in 03_Proposed_Results_Seed_42.
2. Main clean comparison Pure score/mask/index files in
   04_Baseline_Comparison_Results_Seed_42.
3. All discovered injected Pure scenario folders under
   03_Injected_Proposed_Results_Seed_42 (all three configured injection ratios (0.05%, 0.1%, 0.5%)).
4. Pure outputs for stability seeds 42, 7, 123, 2024, 2025 in the existing
   07_Proposed_Results_Seed_* folders.
5. Corrected Pure module-ablation source scores/results for the reference
   0.1% injected datasets, saved into the existing ablation results folder.

Existing result filenames are preserved wherever they already exist.
One NEW internal helper file is intentionally added:
    pure_spatial_score_normalized.npy
This is necessary because the old implementation had no separate Pure-specific
spatial signal; overwriting spatial_score_normalized.npy would corrupt Full.

After this script finishes, rerun the downstream ANALYSIS notebooks/scripts
(04, 05, 06, 07) to regenerate their Excel tables/figures from the corrected
Pure source artifacts. No model training is required.
"""

from __future__ import annotations
import os

from dataclasses import dataclass
from pathlib import Path
import csv
import gc
import json
import math
import sys
import warnings

import numpy as np
from scipy.ndimage import uniform_filter


# ============================================================================
# 1. USER CONFIGURATION
# ============================================================================

ROOT_DIR = Path(os.environ.get("GIS_ANOMALY_PROJECT_ROOT", Path.cwd())).resolve()

# Main folders already used in the project.
MAIN_CLEAN_DIR = ROOT_DIR / "03_Proposed_Results_Seed_42"
MAIN_COMPARISON_DIR = ROOT_DIR / "04_Baseline_Comparison_Results_Seed_42"
INJECTED_ROOT = ROOT_DIR / "03_Injected_Proposed_Results_Seed_42"
ABLATION_RESULTS_DIR = ROOT_DIR / "06_Ablation_Study_Results_Seed_42"

STABILITY_SEEDS = [42, 7, 123, 2024, 2025]
STABILITY_PROPOSED_PATTERN = "07_Proposed_Results_Seed_{seed}"

# Locked architecture/settings.
TOP_FRACTION = 0.02
PURE_BASE_WEIGHT = 0.75
PURE_SPATIAL_WEIGHT = 0.25
AGREEMENT_WEIGHT = 0.60
CERTAINTY_WEIGHT = 0.40
SPATIAL_WINDOW_SIZE = 5
EPS = 1e-6

# Reference injection ratio used by the existing ablation study.
ABLATION_REFERENCE_RATIO_TOKEN = "0.1%"
ABLATION_BUDGETS = [0.001, 0.0025, 0.005, 0.01, 0.02]

VALID_SCENARIOS = {
    "extreme",
    "noise_strong",
    "noise_moderate",
    "noise_mild",
    "inconsistent",
}

SCENARIO_DISPLAY = {
    "extreme": "Extreme",
    "noise_strong": "Strong noise",
    "noise_moderate": "Moderate noise",
    "noise_mild": "Mild noise",
    "inconsistent": "Cross-feature inconsistency",
}

# Explicit injection-ratio folders. Each ratio has its OWN inject_indices.npy.
# This prevents accidental reuse of the indices from a different ratio.
INJECTION_RATIO_CONFIG = {
    "0.05%": {
        "ratio": 0.0005,
        "folder": "Injected_Datasets_with_0.05%_Injection_Rate",
    },
    "0.1%": {
        "ratio": 0.0010,
        "folder": "Injected_Datasets_with_0.1%_Injection_Rate",
    },
    "0.5%": {
        "ratio": 0.0050,
        "folder": "Injected_Datasets_with_0.5%_Injection_Rate",
    },
}

# Set True only if you intentionally want the script to stop on the first
# missing optional folder. Required main-clean inputs always stop execution.
STRICT_OPTIONAL_FOLDERS = False


# ============================================================================
# 2. SMALL UTILITIES
# ============================================================================


def log(message: str = "") -> None:
    print(message, flush=True)


def require_file(path: Path, description: str | None = None) -> Path:
    if not path.exists():
        label = description or path.name
        raise FileNotFoundError(f"Missing {label}: {path}")
    return path


def load_array(path: Path, dtype=np.float32) -> np.ndarray:
    arr = np.load(require_file(path), allow_pickle=False)
    if dtype is not None:
        arr = np.asarray(arr, dtype=dtype)
    return arr


def load_npz_dict(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        return {}
    with np.load(path, allow_pickle=False) as data:
        return {key: np.array(data[key], copy=True) for key in data.files}


def save_npz_preserving(path: Path, updates: dict[str, np.ndarray | float | int]) -> None:
    """Preserve all existing keys, replace/add only supplied keys."""
    payload = load_npz_dict(path)
    for key, value in updates.items():
        payload[key] = np.asarray(value)
    np.savez_compressed(path, **payload)


def robust_bounds(values: np.ndarray, low: float = 1.0, high: float = 99.0) -> tuple[float, float]:
    p_low, p_high = np.percentile(values, [low, high])
    p_low = float(p_low)
    p_high = float(p_high)
    if not np.isfinite(p_low) or not np.isfinite(p_high):
        raise ValueError("Non-finite percentile bound.")
    if p_high <= p_low:
        raise ValueError(f"Invalid normalization bounds: p{low}={p_low}, p{high}={p_high}")
    return p_low, p_high


def normalize_with_bounds(values: np.ndarray, p_low: float, p_high: float) -> np.ndarray:
    if p_high <= p_low:
        raise ValueError(f"Invalid bounds: {p_low}, {p_high}")
    clipped = np.clip(values, p_low, p_high)
    normalized = (clipped - p_low) / (p_high - p_low)
    return normalized.astype(np.float32, copy=False)


def normalize_minmax(values: np.ndarray, minimum: float, maximum: float) -> np.ndarray:
    if maximum <= minimum:
        # All uncertainty values are identical. In that degenerate case there
        # is no dispersion difference, so normalized uncertainty is zero.
        return np.zeros(values.shape, dtype=np.float32)
    normalized = (values - minimum) / (maximum - minimum)
    normalized = np.clip(normalized, 0.0, 1.0)
    return normalized.astype(np.float32, copy=False)


def deterministic_exact_top_fraction(
    scores: np.ndarray,
    fraction: float = TOP_FRACTION,
) -> tuple[np.ndarray, float, np.ndarray]:
    """
    Deterministic exact top-k selection without a full sort.

    All values strictly above the boundary are selected. If a tie occurs at
    the boundary, the remaining tied observations with the smallest original
    indices are selected. This guarantees exactly ceil(fraction * n) pixels.
    """
    scores = np.asarray(scores)
    if scores.ndim != 1:
        raise ValueError("Scores must be one-dimensional.")
    if not np.isfinite(scores).all():
        raise ValueError("Scores contain NaN or Inf.")
    if not (0.0 < fraction <= 1.0):
        raise ValueError("fraction must be in (0, 1].")

    n = scores.size
    k = int(math.ceil(fraction * n))
    if k <= 0 or k > n:
        raise ValueError(f"Invalid selected count k={k} for n={n}")

    boundary_position = n - k
    boundary = float(np.partition(scores, boundary_position)[boundary_position])

    mask = scores > boundary
    current = int(mask.sum())
    remaining = k - current

    if remaining < 0:
        raise RuntimeError("Internal top-k error: selected more than k above boundary.")

    if remaining > 0:
        tied_indices = np.flatnonzero(scores == boundary)
        if tied_indices.size < remaining:
            raise RuntimeError("Internal top-k error: insufficient boundary ties.")
        mask[tied_indices[:remaining]] = True

    selected_indices = np.flatnonzero(mask).astype(np.int64, copy=False)

    if selected_indices.size != k:
        raise RuntimeError(
            f"Exact top-k failed: expected {k}, got {selected_indices.size}."
        )

    return mask.astype(np.uint8, copy=False), boundary, selected_indices


def jaccard(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = np.asarray(mask_a).astype(bool, copy=False)
    b = np.asarray(mask_b).astype(bool, copy=False)
    if a.shape != b.shape:
        raise ValueError("Masks have different shapes.")
    intersection = int(np.logical_and(a, b).sum())
    union = int(np.logical_or(a, b).sum())
    return 1.0 if union == 0 else intersection / union


def score_summary(score: np.ndarray) -> dict[str, float]:
    return {
        "Minimum score": float(np.min(score)),
        "Maximum score": float(np.max(score)),
        "Mean score": float(np.mean(score, dtype=np.float64)),
        "Median score": float(np.median(score)),
        "98th percentile": float(np.percentile(score, 98)),
    }


# ============================================================================
# 3. FIND THE ORIGINAL VALID RASTER MASK
# ============================================================================


def resolve_valid_mask() -> tuple[np.ndarray, Path]:
    candidates = [
        ROOT_DIR / "valid_mask_1.npy",
        MAIN_CLEAN_DIR / "valid_mask_1.npy",
        ROOT_DIR / "valid_mask.npy",
        MAIN_CLEAN_DIR / "valid_mask.npy",
    ]

    for path in candidates:
        if path.exists():
            mask = np.load(path, allow_pickle=False).astype(bool, copy=False)
            if mask.ndim != 2:
                raise ValueError(f"Valid mask must be 2-D: {path}, shape={mask.shape}")
            return mask, path

    # Conservative fallback: search a small set of likely result/data folders,
    # not the whole OneDrive tree.
    search_roots = [
        ROOT_DIR,
        ROOT_DIR / "01_Preprocessing_Results",
        ROOT_DIR / "01_Preprocessed_Data",
        ROOT_DIR / "Data",
        ROOT_DIR / "data",
    ]
    names = {"valid_mask_1.npy", "valid_mask.npy"}
    found: list[Path] = []
    for base in search_roots:
        if not base.exists():
            continue
        for name in names:
            found.extend(base.glob(f"*/{name}"))
            found.extend(base.glob(f"*/*/{name}"))

    found = [p for p in found if p.is_file()]
    if len(found) == 1:
        path = found[0]
        mask = np.load(path, allow_pickle=False).astype(bool, copy=False)
        if mask.ndim != 2:
            raise ValueError(f"Valid mask must be 2-D: {path}, shape={mask.shape}")
        return mask, path

    raise FileNotFoundError(
        "Could not uniquely locate valid_mask_1.npy / valid_mask.npy. "
        "Put the correct path in resolve_valid_mask() candidates. "
        f"Found candidates: {[str(p) for p in found]}"
    )


# ============================================================================
# 4. PURE-SPECIFIC SPATIAL INCONSISTENCY
# ============================================================================


def compute_pure_spatial_inconsistency(
    ae_score: np.ndarray,
    valid_mask: np.ndarray,
    window_size: int = SPATIAL_WINDOW_SIZE,
) -> np.ndarray:
    """
    Reproduce the original spatial-inconsistency construction, but use the
    Pure base (AE reconstruction score) rather than the Full seven-signal base.

    Important: the valid raster geometry and the 5x5 neighborhood are kept
    unchanged. Only the input anomaly surface is corrected.
    """
    ae_score = np.asarray(ae_score, dtype=np.float32)
    n_valid = int(valid_mask.sum())
    if ae_score.size != n_valid:
        raise ValueError(
            f"AE score length ({ae_score.size:,}) does not match valid-mask "
            f"count ({n_valid:,})."
        )

    base_raster = np.full(valid_mask.shape, np.nan, dtype=np.float32)
    base_raster[valid_mask] = ae_score

    valid_float = valid_mask.astype(np.float32, copy=False)
    filled = np.where(valid_mask, base_raster, 0.0).astype(np.float32, copy=False)
    kernel_area = float(window_size * window_size)

    local_sum = uniform_filter(
        filled,
        size=window_size,
        mode="nearest",
    ) * kernel_area

    local_count = uniform_filter(
        valid_float,
        size=window_size,
        mode="nearest",
    ) * kernel_area

    local_mean = local_sum / np.maximum(local_count, 1.0)

    diff_sq = (base_raster - local_mean) ** 2
    diff_sq = np.where(np.isnan(diff_sq), 0.0, diff_sq).astype(np.float32, copy=False)

    local_var_sum = uniform_filter(
        diff_sq,
        size=window_size,
        mode="nearest",
    ) * kernel_area

    local_var = local_var_sum / np.maximum(local_count, 1.0)
    local_std = np.sqrt(np.maximum(local_var, 0.0)).astype(np.float32, copy=False)

    spatial_raster = np.abs(base_raster - local_mean) / (local_std + EPS)
    spatial = spatial_raster[valid_mask].astype(np.float32, copy=False)

    if spatial.shape != ae_score.shape:
        raise RuntimeError("Pure spatial score shape mismatch.")
    if not np.isfinite(spatial).all():
        raise ValueError("Pure spatial score contains NaN or Inf.")

    return spatial


# ============================================================================
# 5. RELIABILITY
# ============================================================================


def three_signal_reliability_clean(
    ae: np.ndarray,
    pure_spatial: np.ndarray,
    context: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Fit Pure reliability calibration on a clean dataset."""
    ae_labels, _, _ = deterministic_exact_top_fraction(ae)
    spatial_labels, _, _ = deterministic_exact_top_fraction(pure_spatial)
    context_labels, _, _ = deterministic_exact_top_fraction(context)

    agreement_count = (
        ae_labels.astype(np.uint8)
        + spatial_labels.astype(np.uint8)
        + context_labels.astype(np.uint8)
    )
    agreement_fraction = agreement_count.astype(np.float32) / 3.0

    stack = np.stack([ae, pure_spatial, context], axis=0).astype(np.float32, copy=False)
    uncertainty_raw = np.std(stack, axis=0).astype(np.float32, copy=False)
    uncertainty_min = float(np.min(uncertainty_raw))
    uncertainty_max = float(np.max(uncertainty_raw))
    uncertainty_norm = normalize_minmax(
        uncertainty_raw,
        uncertainty_min,
        uncertainty_max,
    )

    reliability = (
        AGREEMENT_WEIGHT * agreement_fraction
        + CERTAINTY_WEIGHT * (1.0 - uncertainty_norm)
    ).astype(np.float32, copy=False)
    reliability = np.clip(reliability, 0.0, 1.0).astype(np.float32, copy=False)

    return (
        agreement_count,
        uncertainty_raw,
        uncertainty_norm,
        reliability,
        uncertainty_min,
        uncertainty_max,
    )


def three_signal_reliability_apply(
    ae: np.ndarray,
    pure_spatial: np.ndarray,
    context: np.ndarray,
    uncertainty_min: float,
    uncertainty_max: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ae_labels, _, _ = deterministic_exact_top_fraction(ae)
    spatial_labels, _, _ = deterministic_exact_top_fraction(pure_spatial)
    context_labels, _, _ = deterministic_exact_top_fraction(context)

    agreement_count = (
        ae_labels.astype(np.uint8)
        + spatial_labels.astype(np.uint8)
        + context_labels.astype(np.uint8)
    )
    agreement_fraction = agreement_count.astype(np.float32) / 3.0

    stack = np.stack([ae, pure_spatial, context], axis=0).astype(np.float32, copy=False)
    uncertainty_raw = np.std(stack, axis=0).astype(np.float32, copy=False)
    uncertainty_norm = normalize_minmax(
        uncertainty_raw,
        uncertainty_min,
        uncertainty_max,
    )

    reliability = (
        AGREEMENT_WEIGHT * agreement_fraction
        + CERTAINTY_WEIGHT * (1.0 - uncertainty_norm)
    ).astype(np.float32, copy=False)
    reliability = np.clip(reliability, 0.0, 1.0).astype(np.float32, copy=False)

    return agreement_count, uncertainty_raw, uncertainty_norm, reliability


def two_signal_reliability_fit(
    signal_a: np.ndarray,
    signal_b: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    """Clean calibration for a two-signal leave-one-module-out reliability."""
    labels_a, _, _ = deterministic_exact_top_fraction(signal_a)
    labels_b, _, _ = deterministic_exact_top_fraction(signal_b)
    agreement_fraction = (
        labels_a.astype(np.float32) + labels_b.astype(np.float32)
    ) / 2.0

    stack = np.stack([signal_a, signal_b], axis=0).astype(np.float32, copy=False)
    uncertainty_raw = np.std(stack, axis=0).astype(np.float32, copy=False)
    u_min = float(np.min(uncertainty_raw))
    u_max = float(np.max(uncertainty_raw))
    uncertainty_norm = normalize_minmax(uncertainty_raw, u_min, u_max)

    reliability = (
        AGREEMENT_WEIGHT * agreement_fraction
        + CERTAINTY_WEIGHT * (1.0 - uncertainty_norm)
    ).astype(np.float32, copy=False)
    reliability = np.clip(reliability, 0.0, 1.0).astype(np.float32, copy=False)
    return reliability, u_min, u_max


def two_signal_reliability_apply(
    signal_a: np.ndarray,
    signal_b: np.ndarray,
    u_min: float,
    u_max: float,
) -> np.ndarray:
    labels_a, _, _ = deterministic_exact_top_fraction(signal_a)
    labels_b, _, _ = deterministic_exact_top_fraction(signal_b)
    agreement_fraction = (
        labels_a.astype(np.float32) + labels_b.astype(np.float32)
    ) / 2.0

    stack = np.stack([signal_a, signal_b], axis=0).astype(np.float32, copy=False)
    uncertainty_raw = np.std(stack, axis=0).astype(np.float32, copy=False)
    uncertainty_norm = normalize_minmax(uncertainty_raw, u_min, u_max)

    reliability = (
        AGREEMENT_WEIGHT * agreement_fraction
        + CERTAINTY_WEIGHT * (1.0 - uncertainty_norm)
    ).astype(np.float32, copy=False)
    return np.clip(reliability, 0.0, 1.0).astype(np.float32, copy=False)


# ============================================================================
# 6. PURE CALIBRATION / OUTPUT DATA STRUCTURES
# ============================================================================


@dataclass
class PureCalibration:
    spatial_inconsistency_p1: float
    spatial_inconsistency_p99: float
    fused_spatial_p1: float
    fused_spatial_p99: float
    uncertainty_min: float
    uncertainty_max: float
    final_p1: float
    final_p99: float

    # For Pure module ablation.
    no_spatial_uncertainty_min: float
    no_spatial_uncertainty_max: float
    no_context_uncertainty_min: float
    no_context_uncertainty_max: float


@dataclass
class PureOutputs:
    pure_spatial_norm: np.ndarray
    fused_spatial_raw: np.ndarray
    fused_spatial_norm: np.ndarray
    agreement_count: np.ndarray
    uncertainty_raw: np.ndarray
    uncertainty_norm: np.ndarray
    reliability: np.ndarray
    final_raw: np.ndarray
    final_norm: np.ndarray
    final_labels: np.ndarray
    selected_indices: np.ndarray
    final_threshold: float


# ============================================================================
# 7. CORE CLEAN PURE RECOMPUTATION
# ============================================================================


def validate_clean_inputs(folder: Path, valid_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ae = load_array(folder / "ae_score_normalized.npy")
    context = load_array(folder / "context_score_normalized.npy")
    expected = int(valid_mask.sum())
    if ae.size != expected or context.size != expected:
        raise ValueError(
            f"Input length mismatch in {folder}: AE={ae.size:,}, Context={context.size:,}, "
            f"valid={expected:,}"
        )
    if not np.isfinite(ae).all() or not np.isfinite(context).all():
        raise ValueError(f"Non-finite AE/context values in {folder}")
    return ae, context


def recompute_clean_pure(
    folder: Path,
    valid_mask: np.ndarray,
    *,
    save_outputs: bool = True,
    update_main_npz: bool = True,
) -> tuple[PureCalibration, PureOutputs, dict[str, np.ndarray]]:
    log(f"\n[Clean Pure] {folder}")
    ae, context = validate_clean_inputs(folder, valid_mask)

    # 1) Pure-specific spatial inconsistency from AE only.
    spatial_raw = compute_pure_spatial_inconsistency(ae, valid_mask)
    s_p1, s_p99 = robust_bounds(spatial_raw)
    spatial_norm = normalize_with_bounds(spatial_raw, s_p1, s_p99)

    # 2) Spatial fusion using the same base/spatial weights as Full (0.75 / 0.25).
    fused_spatial_raw = (
        PURE_BASE_WEIGHT * ae
        + PURE_SPATIAL_WEIGHT * spatial_norm
    ).astype(np.float32, copy=False)
    f_p1, f_p99 = robust_bounds(fused_spatial_raw)
    fused_spatial_norm = normalize_with_bounds(fused_spatial_raw, f_p1, f_p99)

    # 3) Corrected Pure reliability = [AE, Pure spatial, Context].
    (
        agreement_count,
        uncertainty_raw,
        uncertainty_norm,
        reliability,
        u_min,
        u_max,
    ) = three_signal_reliability_clean(ae, spatial_norm, context)

    # 4) Final raw (pre-final-clipping) Pure ranking score.
    final_raw = (fused_spatial_norm * reliability).astype(np.float32, copy=False)
    final_p1, final_p99 = robust_bounds(final_raw)
    final_norm = normalize_with_bounds(final_raw, final_p1, final_p99)

    # 5) Exact top-2% selection from PRE-FINAL-CLIPPING score.
    final_labels, final_threshold, selected_indices = deterministic_exact_top_fraction(
        final_raw,
        TOP_FRACTION,
    )

    # 6) Calibrate two-signal Pure ablation reliability variants on clean data.
    _, no_spatial_u_min, no_spatial_u_max = two_signal_reliability_fit(ae, context)
    _, no_context_u_min, no_context_u_max = two_signal_reliability_fit(ae, spatial_norm)

    calibration = PureCalibration(
        spatial_inconsistency_p1=s_p1,
        spatial_inconsistency_p99=s_p99,
        fused_spatial_p1=f_p1,
        fused_spatial_p99=f_p99,
        uncertainty_min=u_min,
        uncertainty_max=u_max,
        final_p1=final_p1,
        final_p99=final_p99,
        no_spatial_uncertainty_min=no_spatial_u_min,
        no_spatial_uncertainty_max=no_spatial_u_max,
        no_context_uncertainty_min=no_context_u_min,
        no_context_uncertainty_max=no_context_u_max,
    )

    outputs = PureOutputs(
        pure_spatial_norm=spatial_norm,
        fused_spatial_raw=fused_spatial_raw,
        fused_spatial_norm=fused_spatial_norm,
        agreement_count=agreement_count,
        uncertainty_raw=uncertainty_raw,
        uncertainty_norm=uncertainty_norm,
        reliability=reliability,
        final_raw=final_raw,
        final_norm=final_norm,
        final_labels=final_labels,
        selected_indices=selected_indices,
        final_threshold=final_threshold,
    )

    # Pure module-ablation clean source scores.
    no_spatial_reliability = two_signal_reliability_apply(
        ae,
        context,
        no_spatial_u_min,
        no_spatial_u_max,
    )
    no_context_reliability = two_signal_reliability_apply(
        ae,
        spatial_norm,
        no_context_u_min,
        no_context_u_max,
    )

    ablation_scores = {
        "Pure base": ae.astype(np.float32, copy=False),
        "Pure without spatial": (ae * no_spatial_reliability).astype(np.float32, copy=False),
        "Pure without context": (fused_spatial_norm * no_context_reliability).astype(np.float32, copy=False),
        "Pure without reliability": fused_spatial_norm.astype(np.float32, copy=False),
        "Pure": final_raw.astype(np.float32, copy=False),
    }

    if save_outputs:
        save_clean_pure_outputs(
            folder=folder,
            outputs=outputs,
            calibration=calibration,
            update_main_npz=update_main_npz,
        )

    log(f"  Pure spatial raw p1/p99: {s_p1:.9g} / {s_p99:.9g}")
    log(f"  Pure fused spatial p1/p99: {f_p1:.9g} / {f_p99:.9g}")
    log(f"  Pure uncertainty min/max: {u_min:.9g} / {u_max:.9g}")
    log(f"  Pure final raw p1/p99: {final_p1:.9g} / {final_p99:.9g}")
    log(f"  Exact selected pixels: {int(final_labels.sum()):,}")
    log(f"  Pure final raw top-2% boundary: {final_threshold:.9g}")

    return calibration, outputs, ablation_scores


def save_clean_pure_outputs(
    folder: Path,
    outputs: PureOutputs,
    calibration: PureCalibration,
    update_main_npz: bool,
) -> None:
    folder.mkdir(parents=True, exist_ok=True)

    # Existing output names are deliberately retained.
    np.save(folder / "pure_final_anomaly_score.npy", outputs.final_norm)
    np.save(folder / "pure_final_anomaly_labels.npy", outputs.final_labels.astype(np.uint8, copy=False))

    # Raw/pre-final-clipping ranking score. Keep both common historical names.
    np.save(folder / "pure_final_anomaly_score_raw.npy", outputs.final_raw)
    np.save(folder / "pure_final_fused_score_preclip.npy", outputs.final_raw)

    # Existing capitalized comparison/stability naming convention.
    np.save(folder / "Pure_top2pct_anomaly_mask.npy", outputs.final_labels.astype(np.uint8, copy=False))
    np.save(folder / "Pure_top2pct_selected_indices.npy", outputs.selected_indices)

    # NEW helper that is required to distinguish Pure spatial from Full spatial.
    np.save(folder / "pure_spatial_score_normalized.npy", outputs.pure_spatial_norm)

    if update_main_npz:
        diagnostics_path = folder / "full_pure_reliability_diagnostics.npz"
        save_npz_preserving(
            diagnostics_path,
            {
                "pure_agreement": outputs.agreement_count.astype(np.uint8, copy=False),
                "pure_uncertainty": outputs.uncertainty_norm,
                "pure_uncertainty_raw": outputs.uncertainty_raw,
                "pure_reliability": outputs.reliability,
            },
        )

        params_path = folder / "full_pure_final_normalization_params.npz"
        save_npz_preserving(
            params_path,
            {
                # Existing Pure keys keep their established meaning.
                "pure_spatial_p1": np.float32(calibration.fused_spatial_p1),
                "pure_spatial_p99": np.float32(calibration.fused_spatial_p99),
                "pure_uncertainty_min": np.float32(calibration.uncertainty_min),
                "pure_uncertainty_max": np.float32(calibration.uncertainty_max),
                "pure_final_p1": np.float32(calibration.final_p1),
                "pure_final_p99": np.float32(calibration.final_p99),
                "pure_base_weight": np.float32(PURE_BASE_WEIGHT),
                "pure_spatial_weight": np.float32(PURE_SPATIAL_WEIGHT),
                "consensus_weight": np.float32(AGREEMENT_WEIGHT),
                "certainty_weight": np.float32(CERTAINTY_WEIGHT),

                # New explicit calibration for the corrected Pure-specific
                # spatial inconsistency signal.
                "pure_spatial_inconsistency_p1": np.float32(
                    calibration.spatial_inconsistency_p1
                ),
                "pure_spatial_inconsistency_p99": np.float32(
                    calibration.spatial_inconsistency_p99
                ),
                "pure_spatial_window_size": np.int32(SPATIAL_WINDOW_SIZE),

                # New explicit clean calibrations for Pure module ablations.
                "pure_no_spatial_uncertainty_min": np.float32(
                    calibration.no_spatial_uncertainty_min
                ),
                "pure_no_spatial_uncertainty_max": np.float32(
                    calibration.no_spatial_uncertainty_max
                ),
                "pure_no_context_uncertainty_min": np.float32(
                    calibration.no_context_uncertainty_min
                ),
                "pure_no_context_uncertainty_max": np.float32(
                    calibration.no_context_uncertainty_max
                ),
            },
        )


# ============================================================================
# 8. APPLY CLEAN CALIBRATION TO AN INJECTED SCENARIO
# ============================================================================


def recompute_injected_pure(
    folder: Path,
    valid_mask: np.ndarray,
    calibration: PureCalibration,
) -> tuple[PureOutputs, dict[str, np.ndarray]]:
    log(f"\n[Injected Pure] {folder}")
    ae, context = validate_clean_inputs(folder, valid_mask)

    spatial_raw = compute_pure_spatial_inconsistency(ae, valid_mask)
    spatial_norm = normalize_with_bounds(
        spatial_raw,
        calibration.spatial_inconsistency_p1,
        calibration.spatial_inconsistency_p99,
    )

    fused_spatial_raw = (
        PURE_BASE_WEIGHT * ae
        + PURE_SPATIAL_WEIGHT * spatial_norm
    ).astype(np.float32, copy=False)

    fused_spatial_norm = normalize_with_bounds(
        fused_spatial_raw,
        calibration.fused_spatial_p1,
        calibration.fused_spatial_p99,
    )

    agreement_count, uncertainty_raw, uncertainty_norm, reliability = (
        three_signal_reliability_apply(
            ae,
            spatial_norm,
            context,
            calibration.uncertainty_min,
            calibration.uncertainty_max,
        )
    )

    final_raw = (fused_spatial_norm * reliability).astype(np.float32, copy=False)
    final_norm = normalize_with_bounds(
        final_raw,
        calibration.final_p1,
        calibration.final_p99,
    )

    final_labels, final_threshold, selected_indices = deterministic_exact_top_fraction(
        final_raw,
        TOP_FRACTION,
    )

    outputs = PureOutputs(
        pure_spatial_norm=spatial_norm,
        fused_spatial_raw=fused_spatial_raw,
        fused_spatial_norm=fused_spatial_norm,
        agreement_count=agreement_count,
        uncertainty_raw=uncertainty_raw,
        uncertainty_norm=uncertainty_norm,
        reliability=reliability,
        final_raw=final_raw,
        final_norm=final_norm,
        final_labels=final_labels,
        selected_indices=selected_indices,
        final_threshold=final_threshold,
    )

    # Corrected Pure module-ablation scores using clean calibrations.
    no_spatial_reliability = two_signal_reliability_apply(
        ae,
        context,
        calibration.no_spatial_uncertainty_min,
        calibration.no_spatial_uncertainty_max,
    )
    no_context_reliability = two_signal_reliability_apply(
        ae,
        spatial_norm,
        calibration.no_context_uncertainty_min,
        calibration.no_context_uncertainty_max,
    )

    ablation_scores = {
        "Pure base": ae.astype(np.float32, copy=False),
        "Pure without spatial": (ae * no_spatial_reliability).astype(np.float32, copy=False),
        "Pure without context": (fused_spatial_norm * no_context_reliability).astype(np.float32, copy=False),
        "Pure without reliability": fused_spatial_norm.astype(np.float32, copy=False),
        "Pure": final_raw.astype(np.float32, copy=False),
    }

    # Save established injected output filenames.
    np.save(folder / "pure_spatial_score_normalized.npy", spatial_norm)
    np.save(folder / "pure_final_anomaly_score_raw.npy", final_raw)
    np.save(folder / "pure_final_fused_score_preclip.npy", final_raw)
    np.save(folder / "pure_final_anomaly_score.npy", final_norm)
    np.save(folder / "pure_final_anomaly_labels.npy", final_labels.astype(np.uint8, copy=False))
    np.save(folder / "Pure_top2pct_anomaly_mask.npy", final_labels.astype(np.uint8, copy=False))
    np.save(folder / "Pure_top2pct_selected_indices.npy", selected_indices)

    # Existing injected diagnostics file uses unprefixed key names.
    np.savez_compressed(
        folder / "pure_reliability_diagnostics.npz",
        agreement=agreement_count.astype(np.uint8, copy=False),
        uncertainty_raw=uncertainty_raw,
        uncertainty_normalized=uncertainty_norm,
        reliability=reliability,
    )

    log(f"  Exact selected pixels: {int(final_labels.sum()):,}")
    log(f"  Pure final raw top-2% boundary: {final_threshold:.9g}")

    return outputs, ablation_scores


# ============================================================================
# 9. DISCOVER ALL INJECTED SCENARIO RESULT FOLDERS
# ============================================================================


def discover_injected_scenario_dirs() -> list[Path]:
    """
    Discover exactly the five scenario-result folders inside EACH of the
    three configured injection-ratio folders.

    Required structure (scenario folders may be direct children or nested
    below the ratio folder):

      03_Injected_Proposed_Results_Seed_42/
        Injected_Datasets_with_0.05%_Injection_Rate/
          inject_indices.npy
          .../extreme/
          .../noise_strong/
          .../noise_moderate/
          .../noise_mild/
          .../inconsistent/

        Injected_Datasets_with_0.1%_Injection_Rate/
          inject_indices.npy
          ... five scenarios ...

        Injected_Datasets_with_0.5%_Injection_Rate/
          inject_indices.npy
          ... five scenarios ...

    A usable scenario folder must contain both:
      ae_score_normalized.npy
      context_score_normalized.npy

    Ambiguous immediate children of INJECTED_ROOT are intentionally ignored,
    because they cannot be safely associated with one injection ratio.
    """
    if not INJECTED_ROOT.exists():
        raise FileNotFoundError(
            f"Injected-results root does not exist: {INJECTED_ROOT}"
        )

    found: list[Path] = []
    problems: list[str] = []

    for ratio_label, config in INJECTION_RATIO_CONFIG.items():
        ratio_dir = INJECTED_ROOT / str(config["folder"])

        if not ratio_dir.exists():
            problems.append(
                f"{ratio_label}: missing ratio folder: {ratio_dir}"
            )
            continue

        index_path = ratio_dir / "inject_indices.npy"
        if not index_path.exists():
            problems.append(
                f"{ratio_label}: missing ratio-specific inject_indices.npy: "
                f"{index_path}"
            )

        for scenario_name in sorted(VALID_SCENARIOS):
            direct = ratio_dir / scenario_name

            candidates: list[Path] = []
            if direct.is_dir():
                candidates.append(direct)

            # Support one or more intermediate result folders while remaining
            # strictly inside this injection-ratio directory.
            candidates.extend(
                p
                for p in ratio_dir.rglob(scenario_name)
                if p.is_dir() and p != direct
            )

            usable = [
                p for p in candidates
                if (p / "ae_score_normalized.npy").exists()
                and (p / "context_score_normalized.npy").exists()
            ]

            # De-duplicate paths.
            usable = sorted(set(usable), key=lambda p: str(p).lower())

            if len(usable) == 0:
                problems.append(
                    f"{ratio_label}/{scenario_name}: no scenario-result folder "
                    "containing ae_score_normalized.npy and "
                    "context_score_normalized.npy was found inside "
                    f"{ratio_dir}"
                )
                continue

            if len(usable) > 1:
                problems.append(
                    f"{ratio_label}/{scenario_name}: multiple usable scenario "
                    f"folders found: {[str(p) for p in usable]}"
                )
                continue

            found.append(usable[0])

    if problems:
        raise FileNotFoundError(
            "Injected-ratio preflight failed. The script will not continue "
            "because silently skipping a ratio could mix injection indices.\n\n"
            + "\n".join(problems)
        )

    expected = len(INJECTION_RATIO_CONFIG) * len(VALID_SCENARIOS)
    if len(found) != expected:
        raise RuntimeError(
            f"Expected {expected} injected scenario folders "
            f"(3 ratios x 5 scenarios), but found {len(found)}."
        )

    return sorted(found, key=lambda p: str(p).lower())


def get_injection_ratio_info(scenario_dir: Path) -> tuple[str, float, Path]:
    """Return (ratio label, ratio value, ratio folder) for a scenario folder."""
    scenario_resolved = scenario_dir.resolve()

    for ratio_label, config in INJECTION_RATIO_CONFIG.items():
        ratio_dir = (INJECTED_ROOT / str(config["folder"])).resolve()
        try:
            scenario_resolved.relative_to(ratio_dir)
            return ratio_label, float(config["ratio"]), ratio_dir
        except ValueError:
            continue

    raise ValueError(
        f"Scenario folder is not inside a configured injection-ratio folder: "
        f"{scenario_dir}"
    )


def validate_ratio_specific_injected_indices(n_samples: int) -> None:
    """Validate each ratio's own injection-index file before any overwrite."""
    for ratio_label, config in INJECTION_RATIO_CONFIG.items():
        ratio_value = float(config["ratio"])
        ratio_dir = INJECTED_ROOT / str(config["folder"])
        index_path = require_file(
            ratio_dir / "inject_indices.npy",
            f"{ratio_label} injection indices",
        )

        indices = np.load(
            index_path,
            allow_pickle=False,
        ).astype(
            np.int64,
            copy=False,
        ).reshape(-1)

        expected_count = int(
            np.floor(
                ratio_value * n_samples
            )
        )

        if indices.size != expected_count:
            raise ValueError(
                f"{ratio_label}: inject_indices.npy contains "
                f"{indices.size:,} indices; expected {expected_count:,}."
            )

        if np.unique(indices).size != indices.size:
            raise ValueError(
                f"{ratio_label}: inject_indices.npy contains duplicates."
            )

        if indices.size > 0 and (
            indices.min() < 0
            or indices.max() >= n_samples
        ):
            raise ValueError(
                f"{ratio_label}: inject_indices.npy contains out-of-range "
                "indices."
            )

        log(
            f"  {ratio_label}: {indices.size:,} ratio-specific injected "
            f"indices validated -> {index_path}"
        )


# ============================================================================
# 10. UPDATE THE CLEAN COMPARISON FOLDER USING THE SAME HISTORICAL FILENAMES
# ============================================================================


def update_main_comparison_folder(outputs: PureOutputs) -> None:
    if not MAIN_COMPARISON_DIR.exists():
        message = f"Comparison folder not found: {MAIN_COMPARISON_DIR}"
        if STRICT_OPTIONAL_FOLDERS:
            raise FileNotFoundError(message)
        warnings.warn(message)
        return

    old_mask_path = MAIN_COMPARISON_DIR / "Pure_top2pct_anomaly_mask.npy"
    old_mask = None
    if old_mask_path.exists():
        old_mask = np.load(old_mask_path, allow_pickle=False).astype(np.uint8, copy=False)

    np.save(
        MAIN_COMPARISON_DIR / "Pure_final_fused_score_preclip.npy",
        outputs.final_raw,
    )
    np.save(
        MAIN_COMPARISON_DIR / "Pure_top2pct_anomaly_mask.npy",
        outputs.final_labels.astype(np.uint8, copy=False),
    )
    np.save(
        MAIN_COMPARISON_DIR / "Pure_top2pct_selected_indices.npy",
        outputs.selected_indices,
    )

    if old_mask is not None and old_mask.shape == outputs.final_labels.shape:
        similarity = jaccard(old_mask, outputs.final_labels)
        log(f"\n[Comparison folder] Old-vs-corrected Pure top-2% Jaccard: {similarity:.6f}")


# ============================================================================
# 11. FIND INJECTED INDICES FOR A SCENARIO
# ============================================================================


def find_injected_indices(scenario_dir: Path) -> Path | None:
    candidate_names = [
        # Current injected notebooks save this exact filename.
        "inject_indices.npy",
        # Historical/alternate names retained for compatibility.
        "injected_indices.npy",
        "injection_indices.npy",
        "inject_idx.npy",
        "Inject_idx.npy",
        "injected_idx.npy",
    ]

    # Check scenario folder and ancestors up to the injected root.
    current = scenario_dir
    while True:
        for name in candidate_names:
            candidate = current / name
            if candidate.exists():
                return candidate

        # Scenario-specific variants.
        for pattern in [
            f"*{scenario_dir.name}*indices*.npy",
            f"*{scenario_dir.name}*idx*.npy",
        ]:
            matches = list(current.glob(pattern))
            if len(matches) == 1:
                return matches[0]

        if current == INJECTED_ROOT or current.parent == current:
            break
        current = current.parent

    # Last, shallow search inside the ratio folder that contains this scenario.
    ratio_parent = scenario_dir.parent
    matches: list[Path] = []
    for pattern in ["*indices*.npy", "*idx*.npy"]:
        matches.extend(ratio_parent.glob(pattern))
    matches = [p for p in matches if p.is_file()]

    # Avoid returning detector selected-index files.
    matches = [
        p for p in matches
        if "selected" not in p.name.lower()
        and "top" not in p.name.lower()
    ]

    if len(matches) == 1:
        return matches[0]
    return None


# ============================================================================
# 12. REFRESH THE EXISTING PER-SCENARIO TOP-2% DETECTION SUMMARY WHEN POSSIBLE
# ============================================================================


def update_injection_detection_summary(
    scenario_dir: Path,
    pure_labels: np.ndarray,
) -> None:
    index_path = find_injected_indices(scenario_dir)
    if index_path is None:
        return

    injected_indices = np.load(index_path, allow_pickle=False).astype(np.int64, copy=False).ravel()
    if injected_indices.size == 0:
        return
    if injected_indices.min() < 0 or injected_indices.max() >= pure_labels.size:
        warnings.warn(
            f"Injected index file does not match score length; skipping summary: {index_path}"
        )
        return

    detected = int(pure_labels[injected_indices].sum())
    total = int(injected_indices.size)
    missed = total - detected
    detection_rate = detected / total
    miss_rate = missed / total

    summary_path = scenario_dir / "injection_detection_summary.csv"
    existing_rows: list[dict[str, str]] = []
    fieldnames = [
        "scenario",
        "method",
        "total_injected",
        "detected_injected",
        "missed_injected",
        "detection_rate",
        "miss_rate",
    ]

    if summary_path.exists():
        try:
            with summary_path.open("r", newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                existing_rows = [dict(row) for row in reader]
                if reader.fieldnames:
                    fieldnames = list(reader.fieldnames)
        except Exception as exc:
            warnings.warn(f"Could not read {summary_path}: {exc}")
            existing_rows = []

    # Preserve every non-Pure row (especially Full), replace only Pure.
    existing_rows = [
        row for row in existing_rows
        if str(row.get("method", "")).strip().lower() != "pure"
    ]

    pure_row = {
        "scenario": scenario_dir.name,
        "method": "Pure",
        "total_injected": str(total),
        "detected_injected": str(detected),
        "missed_injected": str(missed),
        "detection_rate": repr(float(detection_rate)),
        "miss_rate": repr(float(miss_rate)),
    }

    # Ensure required fields exist even when historical CSV field order differs.
    for key in pure_row:
        if key not in fieldnames:
            fieldnames.append(key)

    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in existing_rows:
            writer.writerow(row)
        writer.writerow(pure_row)


# ============================================================================
# 13. CORRECTED PURE MODULE ABLATION RESULTS (REFERENCE 0.1% INJECTION)
# ============================================================================


def is_reference_ablation_scenario_dir(path: Path) -> bool:
    text = str(path.parent).lower()
    return (
        path.name in VALID_SCENARIOS
        and "0.1%" in text
        and "0.05%" not in text
        and "0.5%" not in text
    )


def evaluate_budget_detection(
    score: np.ndarray,
    injected_indices: np.ndarray,
    budget: float,
) -> tuple[int, int, float]:
    mask, _, _ = deterministic_exact_top_fraction(score, budget)
    detected = int(mask[injected_indices].sum())
    total = int(injected_indices.size)
    rate = 100.0 * detected / total
    return int(mask.sum()), detected, rate


def write_corrected_pure_ablation_outputs(
    clean_ablation_scores: dict[str, np.ndarray],
    injected_ablation_scores: dict[Path, dict[str, np.ndarray]],
) -> None:
    """
    Save corrected Pure-only module-ablation tables in the existing ablation
    results folder. These are intentionally Pure-only replacement tables; the
    Full and representation/detector ablation results remain valid and do not
    need recomputation.
    """
    if not injected_ablation_scores:
        warnings.warn(
            "No reference 0.1% injected folders with usable injected-index files were found; "
            "Pure module-ablation replacement tables were not generated."
        )
        return

    ABLATION_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    variants = [
        "Pure base",
        "Pure without spatial",
        "Pure without context",
        "Pure without reliability",
        "Pure",
    ]

    # Clean exact-top-2% masks and overlap with corrected complete Pure.
    clean_masks: dict[str, np.ndarray] = {}
    for variant in variants:
        mask, _, _ = deterministic_exact_top_fraction(clean_ablation_scores[variant], TOP_FRACTION)
        clean_masks[variant] = mask

    reference_mask = clean_masks["Pure"]
    overlap_rows: list[dict[str, object]] = []
    for variant in variants:
        mask = clean_masks[variant]
        inter = int(np.logical_and(mask.astype(bool), reference_mask.astype(bool)).sum())
        union = int(np.logical_or(mask.astype(bool), reference_mask.astype(bool)).sum())
        jac = 1.0 if union == 0 else inter / union
        replacement = 100.0 * (1.0 - inter / int(reference_mask.sum()))
        overlap_rows.append(
            {
                "Section": "Spatial, context, and reliability",
                "Variant": variant,
                "Reference": "Pure",
                "Intersection": inter,
                "Union": union,
                "Jaccard similarity": jac,
                "Replacement rate (%)": replacement,
            }
        )

    overlap_path = ABLATION_RESULTS_DIR / "Pure_Module_Corrected_Clean_Overlap.csv"
    with overlap_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(overlap_rows[0].keys()))
        writer.writeheader()
        writer.writerows(overlap_rows)

    # Clean score summary.
    clean_summary_rows: list[dict[str, object]] = []
    for variant in variants:
        row = {"Variant": variant}
        row.update(score_summary(clean_ablation_scores[variant]))
        clean_summary_rows.append(row)

    clean_summary_path = ABLATION_RESULTS_DIR / "Pure_Module_Corrected_Clean_Score_Summary.csv"
    with clean_summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(clean_summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(clean_summary_rows)

    # Long injected results across five scenarios and five budgets.
    long_rows: list[dict[str, object]] = []

    for scenario_dir in sorted(injected_ablation_scores, key=lambda p: p.name):
        index_path = find_injected_indices(scenario_dir)
        if index_path is None:
            continue
        injected_indices = np.load(index_path, allow_pickle=False).astype(np.int64, copy=False).ravel()
        scenario_scores = injected_ablation_scores[scenario_dir]

        for variant in variants:
            score = scenario_scores[variant]
            for budget in ABLATION_BUDGETS:
                selected, detected, rate = evaluate_budget_detection(
                    score,
                    injected_indices,
                    budget,
                )
                long_rows.append(
                    {
                        "Section": "Spatial, contextual, and reliability modules",
                        "Scenario key": scenario_dir.name,
                        "Scenario": SCENARIO_DISPLAY[scenario_dir.name],
                        "Variant": variant,
                        "Budget": f"{100 * budget:.2f}%",
                        "Budget ratio": budget,
                        "Selected pixels": selected,
                        "Injected pixels": int(injected_indices.size),
                        "Detected injected pixels": detected,
                        "Detection rate (%)": rate,
                    }
                )

    if not long_rows:
        warnings.warn("No corrected Pure ablation long-results rows were generated.")
        return

    long_path = ABLATION_RESULTS_DIR / "Pure_Module_Corrected_Long_Results.csv"
    with long_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(long_rows[0].keys()))
        writer.writeheader()
        writer.writerows(long_rows)

    # Paper-style compact summary matching the existing Table 14 logic.
    def rows_for(variant: str, budget: float | None = None) -> list[dict[str, object]]:
        selected = [row for row in long_rows if row["Variant"] == variant]
        if budget is not None:
            selected = [
                row for row in selected
                if math.isclose(float(row["Budget ratio"]), budget)
            ]
        return selected

    compact_rows: list[dict[str, object]] = []
    reference_top2_mean: float | None = None

    # Compute complete Pure reference first.
    pure_01 = rows_for("Pure", 0.001)
    pure_02 = rows_for("Pure", 0.02)
    if pure_01 and pure_02:
        reference_top2_mean = float(np.mean([float(r["Detection rate (%)"]) for r in pure_02]))

    for variant in variants:
        r01 = rows_for(variant, 0.001)
        r02 = rows_for(variant, 0.02)
        if not r01 or not r02:
            continue

        mean_01 = float(np.mean([float(r["Detection rate (%)"]) for r in r01]))
        mean_02 = float(np.mean([float(r["Detection rate (%)"]) for r in r02]))

        mild = next(
            (float(r["Detection rate (%)"]) for r in r02 if r["Scenario key"] == "noise_mild"),
            float("nan"),
        )
        inconsistent = next(
            (float(r["Detection rate (%)"]) for r in r02 if r["Scenario key"] == "inconsistent"),
            float("nan"),
        )

        compact_rows.append(
            {
                "Configuration": variant,
                "Mean DR at 0.1%": mean_01,
                "Mild DR at 2%": mild,
                "Cross-feature DR at 2%": inconsistent,
                "Mean DR at 2%": mean_02,
                "Delta vs Pure reference": (
                    mean_02 - reference_top2_mean
                    if reference_top2_mean is not None
                    else float("nan")
                ),
            }
        )

    compact_path = ABLATION_RESULTS_DIR / "Pure_Module_Corrected_Table14_Replacement.csv"
    with compact_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(compact_rows[0].keys()))
        writer.writeheader()
        writer.writerows(compact_rows)

    # Calibration audit specific to corrected Pure module variants.
    calibration_audit = ABLATION_RESULTS_DIR / "Pure_Module_Corrected_Calibration.json"
    with calibration_audit.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "pure_base_weight": PURE_BASE_WEIGHT,
                "pure_spatial_weight": PURE_SPATIAL_WEIGHT,
                "agreement_weight": AGREEMENT_WEIGHT,
                "certainty_weight": CERTAINTY_WEIGHT,
                "spatial_window_size": SPATIAL_WINDOW_SIZE,
                "note": (
                    "Pure-specific spatial inconsistency is computed from the AE reconstruction "
                    "score; Full spatial files are not reused."
                ),
            },
            handle,
            indent=2,
        )

    log("\n[Pure module ablation replacement files]")
    log(f"  {long_path}")
    log(f"  {compact_path}")
    log(f"  {overlap_path}")
    log(f"  {clean_summary_path}")


# ============================================================================
# 14. STABILITY SEEDS
# ============================================================================


def recompute_stability_pure(valid_mask: np.ndarray) -> list[dict[str, object]]:
    audit: list[dict[str, object]] = []

    for seed in STABILITY_SEEDS:
        folder = ROOT_DIR / STABILITY_PROPOSED_PATTERN.format(seed=seed)
        if not folder.exists():
            message = f"Stability proposed folder missing for seed {seed}: {folder}"
            if STRICT_OPTIONAL_FOLDERS:
                raise FileNotFoundError(message)
            warnings.warn(message)
            continue

        ae_path = folder / "ae_score_normalized.npy"
        context_path = folder / "context_score_normalized.npy"
        if not ae_path.exists() or not context_path.exists():
            message = (
                f"Seed {seed} folder lacks AE/context normalized scores; cannot recompute Pure: {folder}"
            )
            if STRICT_OPTIONAL_FOLDERS:
                raise FileNotFoundError(message)
            warnings.warn(message)
            continue

        old_mask_path = folder / "Pure_top2pct_anomaly_mask.npy"
        old_mask = None
        if old_mask_path.exists():
            old_mask = np.load(old_mask_path, allow_pickle=False).astype(np.uint8, copy=False)

        calibration, outputs, _ = recompute_clean_pure(
            folder,
            valid_mask,
            save_outputs=True,
            update_main_npz=True,
        )

        similarity = None
        if old_mask is not None and old_mask.shape == outputs.final_labels.shape:
            similarity = jaccard(old_mask, outputs.final_labels)

        audit.append(
            {
                "seed": seed,
                "selected_pixels": int(outputs.final_labels.sum()),
                "old_vs_corrected_jaccard": similarity,
                "pure_spatial_inconsistency_p1": calibration.spatial_inconsistency_p1,
                "pure_spatial_inconsistency_p99": calibration.spatial_inconsistency_p99,
                "pure_final_p1": calibration.final_p1,
                "pure_final_p99": calibration.final_p99,
            }
        )

        # Free the 4.29M-element arrays before the next seed.
        del outputs
        gc.collect()

    return audit


# ============================================================================
# 15. AUDIT OUTPUT
# ============================================================================


def write_audit_csv(rows: list[dict[str, object]], filename: str) -> None:
    if not rows:
        return
    path = ROOT_DIR / filename
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    log(f"Saved audit: {path}")


# ============================================================================
# 16. MAIN
# ============================================================================


def main() -> None:
    log("=" * 78)
    log("CORRECTED PURE RECOMPUTATION — FINAL LOCKED 0.75/0.25")
    log("=" * 78)
    log(f"ROOT_DIR: {ROOT_DIR}")
    log("Locked weights:")
    log(f"  Pure AE/base weight     = {PURE_BASE_WEIGHT}")
    log(f"  Pure spatial weight     = {PURE_SPATIAL_WEIGHT}")
    log(f"  Reliability agreement  = {AGREEMENT_WEIGHT}")
    log(f"  Reliability certainty  = {CERTAINTY_WEIGHT}")
    log(f"  Spatial window          = {SPATIAL_WINDOW_SIZE} x {SPATIAL_WINDOW_SIZE}")
    log(f"  Final selection         = exact top {100 * TOP_FRACTION:.1f}%")

    require_file(MAIN_CLEAN_DIR / "ae_score_normalized.npy", "main clean AE score")
    require_file(MAIN_CLEAN_DIR / "context_score_normalized.npy", "main clean context score")

    valid_mask, valid_mask_path = resolve_valid_mask()
    n_valid = int(valid_mask.sum())
    log(f"\nValid mask: {valid_mask_path}")
    log(f"Raster shape: {valid_mask.shape}")
    log(f"Valid pixels: {n_valid:,}")

    expected_top2 = int(math.ceil(TOP_FRACTION * n_valid))
    log(f"Expected exact top-2% count: {expected_top2:,}")

    log("\nValidating ratio-specific injection indices:")
    validate_ratio_specific_injected_indices(n_valid)

    # ----------------------------------------------------------------------
    # A. Main clean Pure
    # ----------------------------------------------------------------------
    main_calibration, main_outputs, clean_ablation_scores = recompute_clean_pure(
        MAIN_CLEAN_DIR,
        valid_mask,
        save_outputs=True,
        update_main_npz=True,
    )

    if int(main_outputs.final_labels.sum()) != expected_top2:
        raise RuntimeError("Main Pure exact top-2% count is wrong.")

    update_main_comparison_folder(main_outputs)

    # ----------------------------------------------------------------------
    # B. All injected folders, all ratios/scenarios
    # ----------------------------------------------------------------------
    injected_dirs = discover_injected_scenario_dirs()
    log(f"\nDiscovered injected scenario result folders: {len(injected_dirs)}")
    log("  Expected coverage: 3 injection ratios x 5 scenarios = 15 folders")
    for scenario_dir in injected_dirs:
        ratio_label, _, _ = get_injection_ratio_info(scenario_dir)
        log(f"  {ratio_label:>5s} | {scenario_dir.name:16s} | {scenario_dir}")

    injected_audit: list[dict[str, object]] = []
    ablation_injected_scores: dict[Path, dict[str, np.ndarray]] = {}

    for scenario_dir in injected_dirs:
        old_mask_path = scenario_dir / "pure_final_anomaly_labels.npy"
        old_mask = None
        if old_mask_path.exists():
            old_mask = np.load(old_mask_path, allow_pickle=False).astype(np.uint8, copy=False)

        outputs, module_scores = recompute_injected_pure(
            scenario_dir,
            valid_mask,
            main_calibration,
        )

        update_injection_detection_summary(scenario_dir, outputs.final_labels)

        old_new_jaccard = None
        if old_mask is not None and old_mask.shape == outputs.final_labels.shape:
            old_new_jaccard = jaccard(old_mask, outputs.final_labels)

        ratio_label, ratio_value, ratio_dir = get_injection_ratio_info(
            scenario_dir
        )
        ratio_index_path = ratio_dir / "inject_indices.npy"
        ratio_indices = np.load(
            ratio_index_path,
            allow_pickle=False,
        ).astype(
            np.int64,
            copy=False,
        ).reshape(-1)

        injected_audit.append(
            {
                "injection_ratio": ratio_label,
                "injection_ratio_value": ratio_value,
                "injected_index_file": str(ratio_index_path),
                "injected_index_count": int(ratio_indices.size),
                "folder": str(scenario_dir),
                "scenario": scenario_dir.name,
                "selected_pixels": int(outputs.final_labels.sum()),
                "old_vs_corrected_jaccard": old_new_jaccard,
                "final_raw_threshold": outputs.final_threshold,
                "mean_reliability": float(np.mean(outputs.reliability, dtype=np.float64)),
            }
        )

        # Retain only the reference 0.1% module scores needed for ablation.
        if is_reference_ablation_scenario_dir(scenario_dir):
            if find_injected_indices(scenario_dir) is not None:
                ablation_injected_scores[scenario_dir] = {
                    key: np.array(value, copy=True)
                    for key, value in module_scores.items()
                }

        del outputs, module_scores, old_mask
        gc.collect()

    write_audit_csv(injected_audit, "08_Corrected_Pure_Injected_Audit.csv")

    # ----------------------------------------------------------------------
    # C. Pure module ablation replacement results
    # ----------------------------------------------------------------------
    write_corrected_pure_ablation_outputs(
        clean_ablation_scores=clean_ablation_scores,
        injected_ablation_scores=ablation_injected_scores,
    )

    # The retained reference scenario scores are large; release them before
    # stability processing.
    del ablation_injected_scores
    gc.collect()

    # ----------------------------------------------------------------------
    # D. Stability seeds (Pure only)
    # ----------------------------------------------------------------------
    stability_audit = recompute_stability_pure(valid_mask)
    write_audit_csv(stability_audit, "08_Corrected_Pure_Stability_Audit.csv")

    # ----------------------------------------------------------------------
    # E. Final validation
    # ----------------------------------------------------------------------
    log("\n" + "=" * 78)
    log("CORRECTION COMPLETED")
    log("=" * 78)
    log("Main clean Pure artifacts were overwritten using their established names.")
    log("Injected Pure artifacts were overwritten in every discovered scenario folder.")
    log("Pure stability artifacts were overwritten in every usable seed folder.")
    log("Full and all baselines were left untouched.")
    log("")
    log("IMPORTANT NEXT STEP:")
    log("  Rerun downstream analysis only (04 -> 05 -> 06 -> 07).")
    log("  Do NOT rerun model training.")
    log("")
    log("For 06 ablation, use the corrected Pure-specific file")
    log("  pure_spatial_score_normalized.npy")
    log("for Pure variants; continue using spatial_score_normalized.npy for Full.")
    log("")
    log("The script also writes Pure-only corrected module-ablation replacement CSVs")
    log("inside the existing 06_Ablation_Study_Results_Seed_42 folder.")


def validate_final_locked_weight_files() -> None:
    """Fail loudly unless the saved main/stability Pure parameter files use 0.75/0.25."""
    checks = [("main", MAIN_CLEAN_DIR / "full_pure_final_normalization_params.npz")]
    for seed in STABILITY_SEEDS:
        folder = ROOT_DIR / f"07_Proposed_Results_Seed_{seed}"
        path = folder / "full_pure_final_normalization_params.npz"
        if path.exists():
            checks.append((f"stability seed {seed}", path))

    for label, path in checks:
        with np.load(path, allow_pickle=False) as z:
            if "pure_base_weight" not in z.files or "pure_spatial_weight" not in z.files:
                raise KeyError(f"{label}: missing Pure weight keys in {path}")
            bw = float(z["pure_base_weight"])
            sw = float(z["pure_spatial_weight"])
        log(f"FINAL WEIGHT CHECK | {label}: Pure = {bw:.8f} / {sw:.8f}")
        if not np.isclose(bw, PURE_BASE_WEIGHT, atol=1e-8):
            raise RuntimeError(f"{label}: saved Pure base weight is {bw}, expected {PURE_BASE_WEIGHT}")
        if not np.isclose(sw, PURE_SPATIAL_WEIGHT, atol=1e-8):
            raise RuntimeError(f"{label}: saved Pure spatial weight is {sw}, expected {PURE_SPATIAL_WEIGHT}")

    log("FINAL LOCKED-WEIGHT VALIDATION PASSED: Pure uses 0.75 AE + 0.25 Pure spatial.")


if __name__ == "__main__":
    try:
        main()
        validate_final_locked_weight_files()
    except Exception as exc:
        log("\nERROR: corrected Pure recomputation stopped.")
        log(f"{type(exc).__name__}: {exc}")
        raise
