#!/usr/bin/env python
"""
13_full_without_if_end_to_end.py

Purpose
-------
Construct and evaluate a TRUE downstream end-to-end "Full without IF" variant
for the existing geospatial outlier-detection project.

This experiment does NOT retrain the MVAE, LOF, MD, context autoencoder, or any
preprocessing object. IF is removed from the Full base, and every downstream
quantity that depends on the Full base is recomputed:

    Clean normalized components:
        AE + LOF(feature) + MD(feature) + LOF(embedding) + MD(embedding)
                  |
                  v
        5-signal Full-without-IF base
                  |
                  v
        Full-without-IF spatial inconsistency (recomputed from this new base)
                  |
                  v
        0.75 * base + 0.25 * spatial, then clean-calibrated normalization
                  |
                  v
        7-signal reliability:
        5 base components + new spatial signal + existing context signal
                  |
                  v
        final pre-clipping score = refined_normalized * reliability
                  |
                  v
        deterministic exact top-k ranking

Clean-data calibration is derived FIRST for all newly created downstream
signals. Those clean bounds are then frozen and reused for every injected
scenario, matching the deployment-oriented evaluation logic used in the
current project.

The script never writes into the established 03_/04_/05_/06_/07_/08_ result
folders. All new outputs are written under:

    09_Full_Without_IF_EndToEnd_Seed_42/

Default project root:
    <PROJECT_ROOT>

Requirements
------------
numpy, pandas, scipy, openpyxl

Source files expected in 03_Proposed_Results_Seed_42:
    ae_score_normalized.npy
    lof_feature_score_normalized.npy
    md_feature_score_normalized.npy
    lof_embedding_score_normalized.npy
    md_embedding_score_normalized.npy
    context_score_normalized.npy
    detector_anomaly_labels.npz

Source files expected in each injected scenario folder:
    same six normalized scores listed above
    detector_anomaly_labels.npz
    full_final_anomaly_score_raw.npy   # existing Full, comparison only

Other required files:
    valid_mask_1.npy (project root; valid_mask.npy is a fallback)
    04_Baseline_Comparison_Results_Seed_42/Full_final_fused_score_preclip.npy
    inject_indices.npy in each injection-ratio folder

No IF score file is loaded by this script.
"""

from __future__ import annotations
import os

import argparse
import math
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, Mapping, Tuple

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter
from openpyxl import load_workbook
from openpyxl.styles import Alignment, Font
from openpyxl.utils import get_column_letter


# ---------------------------------------------------------------------------
# Fixed experiment definition
# ---------------------------------------------------------------------------

DEFAULT_ROOT = Path(os.environ.get("GIS_ANOMALY_PROJECT_ROOT", Path.cwd())).resolve()

SEED = 42
INTERNAL_SIGNAL_FRACTION = 0.02
BASE_WEIGHT = 0.75
SPATIAL_WEIGHT = 0.25
AGREEMENT_WEIGHT = 0.60
CERTAINTY_WEIGHT = 0.40
WINDOW_SIZE = 5
EPSILON = 1e-6

BUDGETS = OrderedDict(
    [
        ("0.10%", 0.0010),
        ("0.25%", 0.0025),
        ("0.50%", 0.0050),
        ("1.00%", 0.0100),
        ("2.00%", 0.0200),
    ]
)

RATIO_CONFIG = OrderedDict(
    [
        (
            "0.05%",
            {
                "ratio": 0.0005,
                "folder": "Injected_Datasets_with_0.05%_Injection_Rate",
            },
        ),
        (
            "0.1%",
            {
                "ratio": 0.0010,
                "folder": "Injected_Datasets_with_0.1%_Injection_Rate",
            },
        ),
        (
            "0.5%",
            {
                "ratio": 0.0050,
                "folder": "Injected_Datasets_with_0.5%_Injection_Rate",
            },
        ),
    ]
)

SCENARIOS = OrderedDict(
    [
        ("extreme", "Extreme"),
        ("noise_strong", "Strong noise"),
        ("noise_moderate", "Moderate noise"),
        ("noise_mild", "Mild noise"),
        ("inconsistent", "Cross-feature inconsistency"),
    ]
)

# The five base signals retained after removing BOTH IF signals.
NOIF_COMPONENT_FILES = OrderedDict(
    [
        ("ae", "ae_score_normalized.npy"),
        ("lof_feature", "lof_feature_score_normalized.npy"),
        ("md_feature", "md_feature_score_normalized.npy"),
        ("lof_embedding", "lof_embedding_score_normalized.npy"),
        ("md_embedding", "md_embedding_score_normalized.npy"),
    ]
)

CONTEXT_FILE = "context_score_normalized.npy"
DETECTOR_LABEL_ARCHIVE = "detector_anomaly_labels.npz"

# Archive keys corresponding to the five retained base components.
NOIF_LABEL_KEYS = tuple(NOIF_COMPONENT_FILES.keys())


# ---------------------------------------------------------------------------
# Basic I/O and validation
# ---------------------------------------------------------------------------

def require_file(path: Path, description: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {description}:\n{path.resolve()}"
        )
    return path


def load_vector(
    path: Path,
    expected_length: int | None = None,
    dtype=np.float32,
) -> np.ndarray:
    require_file(path, path.name)
    arr = np.load(path, mmap_mode="r", allow_pickle=False)
    arr = np.asarray(arr).reshape(-1)

    if expected_length is not None and arr.size != expected_length:
        raise ValueError(
            f"{path}: found {arr.size:,} values; "
            f"expected {expected_length:,}."
        )

    if not np.isfinite(arr).all():
        raise ValueError(f"{path} contains NaN or infinite values.")

    if dtype is not None:
        arr = arr.astype(dtype, copy=False)

    return arr


def find_valid_mask(root_dir: Path) -> Path:
    candidates = [
        root_dir / "valid_mask_1.npy",
        root_dir / "valid_mask.npy",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not find valid_mask_1.npy or valid_mask.npy in:\n"
        f"{root_dir.resolve()}"
    )


def load_valid_mask(mask_path: Path, expected_length: int) -> np.ndarray:
    mask = np.load(mask_path, allow_pickle=False)
    if mask.ndim != 2:
        raise ValueError(
            f"{mask_path.name} must be a 2-D raster mask; "
            f"found shape {mask.shape}."
        )
    mask = mask.astype(bool, copy=False)
    if int(mask.sum()) != expected_length:
        raise ValueError(
            f"{mask_path.name} has {int(mask.sum()):,} valid cells; "
            f"expected {expected_length:,}."
        )
    return mask


def load_noif_scores(
    folder: Path,
    expected_length: int | None = None,
) -> Dict[str, np.ndarray]:
    scores: Dict[str, np.ndarray] = {}

    n = expected_length
    for key, filename in NOIF_COMPONENT_FILES.items():
        arr = load_vector(folder / filename, expected_length=n)
        if n is None:
            n = arr.size
        scores[key] = arr

    context = load_vector(
        folder / CONTEXT_FILE,
        expected_length=n,
    )
    scores["context"] = context

    return scores


def load_saved_component_labels(
    folder: Path,
    expected_length: int,
) -> Dict[str, np.ndarray]:
    archive_path = require_file(
        folder / DETECTOR_LABEL_ARCHIVE,
        "detector anomaly-label archive",
    )

    labels: Dict[str, np.ndarray] = {}
    with np.load(archive_path, allow_pickle=False) as archive:
        available = set(archive.files)
        for key in NOIF_LABEL_KEYS:
            if key not in available:
                raise KeyError(
                    f"Missing key '{key}' in {archive_path}.\n"
                    f"Available keys: {sorted(available)}"
                )
            arr = np.asarray(archive[key]).reshape(-1)
            if arr.size != expected_length:
                raise ValueError(
                    f"{archive_path}:{key} has {arr.size:,} labels; "
                    f"expected {expected_length:,}."
                )
            unique = set(np.unique(arr).tolist())
            if not unique.issubset({0, 1, False, True}):
                raise ValueError(
                    f"{archive_path}:{key} is not binary: {sorted(unique)}"
                )
            labels[key] = arr.astype(np.uint8, copy=False)

    return labels


# ---------------------------------------------------------------------------
# Source-tree preflight audit
# ---------------------------------------------------------------------------

def inspect_npy_shape(path: Path) -> tuple[int, ...]:
    require_file(path, path.name)
    arr = np.load(path, mmap_mode="r", allow_pickle=False)
    return tuple(arr.shape)


def preflight_source_tree(
    root_dir: Path,
    output_root: Path,
) -> pd.DataFrame:
    """
    Audit every source path before any Full-without-IF computation.

    The audit verifies:
      * the three ratio-specific inject_indices.npy files are read from their
        own ratio folders;
      * the expected number of injected indices matches floor(ratio * N);
      * when injection_ground_truth_labels.npy exists, its positive indices
        exactly match inject_indices.npy;
      * all 15 ratio/scenario folders exist;
      * every retained normalized component score is loaded from the exact
        ratio/scenario folder and has N values;
      * detector_anomaly_labels.npz contains the retained AE/LOF/MD keys;
      * the existing Full raw score used only for comparison is taken from
        the same ratio/scenario folder;
      * no IF score file is part of the Full-without-IF input manifest.
    """
    clean_dir = root_dir / "03_Proposed_Results_Seed_42"
    injected_root = root_dir / "03_Injected_Proposed_Results_Seed_42"
    comparison_dir = root_dir / "04_Baseline_Comparison_Results_Seed_42"

    clean_ae_path = require_file(
        clean_dir / NOIF_COMPONENT_FILES["ae"],
        "clean AE normalized score",
    )
    clean_ae = np.load(clean_ae_path, mmap_mode="r", allow_pickle=False)
    n_samples = int(np.asarray(clean_ae).reshape(-1).size)

    rows = []

    def add_row(scope, ratio, scenario, role, path, passed, details):
        rows.append(
            {
                "Scope": scope,
                "Injection_Ratio": ratio,
                "Scenario": scenario,
                "Role": role,
                "Resolved_Path": str(path.resolve()),
                "Passed": bool(passed),
                "Details": details,
            }
        )

    for key, filename in NOIF_COMPONENT_FILES.items():
        path = clean_dir / filename
        shape = inspect_npy_shape(path)
        size = int(np.prod(shape))
        passed = size == n_samples
        add_row(
            "Clean", "", "", f"retained score: {key}", path, passed,
            f"shape={shape}; flattened_length={size:,}; expected={n_samples:,}",
        )
        if not passed:
            raise ValueError(f"Clean source length mismatch: {path}")

    context_path = clean_dir / CONTEXT_FILE
    shape = inspect_npy_shape(context_path)
    size = int(np.prod(shape))
    if size != n_samples:
        raise ValueError(f"Clean context length mismatch: {context_path}")
    add_row(
        "Clean", "", "", "context score", context_path, True,
        f"shape={shape}; flattened_length={size:,}",
    )

    clean_archive = require_file(
        clean_dir / DETECTOR_LABEL_ARCHIVE,
        "clean detector-label archive",
    )
    with np.load(clean_archive, allow_pickle=False) as archive:
        missing = [k for k in NOIF_LABEL_KEYS if k not in archive.files]
        available = sorted(archive.files)
    if missing:
        raise KeyError(
            f"Clean label archive is missing retained keys {missing}:\n{clean_archive}"
        )
    add_row(
        "Clean", "", "", "retained detector labels", clean_archive, True,
        f"required_keys={list(NOIF_LABEL_KEYS)}; available_keys={available}",
    )

    mask_path = find_valid_mask(root_dir)
    mask = np.load(mask_path, mmap_mode="r", allow_pickle=False)
    if mask.ndim != 2:
        raise ValueError(f"Valid mask is not 2-D: {mask_path} -> {mask.shape}")
    valid_count = int(np.count_nonzero(mask))
    if valid_count != n_samples:
        raise ValueError(
            f"Valid mask count {valid_count:,} != N {n_samples:,}: {mask_path}"
        )
    add_row(
        "Clean", "", "", "valid raster mask", mask_path, True,
        f"shape={tuple(mask.shape)}; valid_cells={valid_count:,}",
    )

    clean_full_path = require_file(
        comparison_dir / "Full_final_fused_score_preclip.npy",
        "clean existing Full pre-final-clipping score",
    )
    shape = inspect_npy_shape(clean_full_path)
    size = int(np.prod(shape))
    if size != n_samples:
        raise ValueError(f"Clean Full comparison score length mismatch: {clean_full_path}")
    add_row(
        "Clean", "", "", "existing Full comparison score", clean_full_path, True,
        f"shape={shape}; flattened_length={size:,}",
    )

    for ratio_label, cfg in RATIO_CONFIG.items():
        ratio_dir = injected_root / str(cfg["folder"])
        if not ratio_dir.is_dir():
            raise FileNotFoundError(f"Missing ratio folder:\n{ratio_dir}")

        idx_path = require_file(
            ratio_dir / "inject_indices.npy",
            f"{ratio_label} ratio-specific injected-index file",
        )
        idx = np.load(idx_path, allow_pickle=False).astype(np.int64, copy=False).reshape(-1)
        expected_count = int(math.floor(float(cfg["ratio"]) * n_samples))
        checks = [
            idx.size == expected_count,
            idx.size > 0,
            np.unique(idx).size == idx.size,
            idx.min(initial=0) >= 0,
            idx.max(initial=-1) < n_samples,
        ]
        if not all(checks):
            raise ValueError(
                f"Invalid ratio-specific injected indices for {ratio_label}: {idx_path}"
            )
        add_row(
            "Injected", ratio_label, "", "ratio-specific inject_indices", idx_path, True,
            f"count={idx.size:,}; expected={expected_count:,}; unique=yes; range=valid",
        )

        gt_path = ratio_dir / "injection_ground_truth_labels.npy"
        if gt_path.exists():
            gt = np.load(gt_path, mmap_mode="r", allow_pickle=False)
            gt = np.asarray(gt).reshape(-1)
            if gt.size != n_samples:
                raise ValueError(
                    f"Ground-truth label length mismatch for {ratio_label}: {gt_path}"
                )
            gt_unique = set(np.unique(gt).tolist())
            if not gt_unique.issubset({0, 1, False, True}):
                raise ValueError(
                    f"Ground-truth labels are not binary for {ratio_label}: {gt_path}"
                )
            gt_indices = np.flatnonzero(gt)
            same = np.array_equal(np.sort(idx), gt_indices)
            if not same:
                raise ValueError(
                    f"inject_indices.npy does not match injection_ground_truth_labels.npy "
                    f"for {ratio_label}."
                )
            add_row(
                "Injected", ratio_label, "", "ground-truth/index cross-check", gt_path, True,
                f"positive_labels={gt_indices.size:,}; exact index-set match=yes",
            )
        else:
            add_row(
                "Injected", ratio_label, "", "ground-truth/index cross-check", gt_path, True,
                "injection_ground_truth_labels.npy not present; index count/range checks still passed",
            )

        for scenario_key, scenario_label in SCENARIOS.items():
            scenario_dir = ratio_dir / scenario_key
            if not scenario_dir.is_dir():
                raise FileNotFoundError(
                    f"Missing scenario folder for {ratio_label}/{scenario_key}:\n{scenario_dir}"
                )

            for key, filename in NOIF_COMPONENT_FILES.items():
                path = require_file(
                    scenario_dir / filename,
                    f"{ratio_label}/{scenario_key} retained score {key}",
                )
                shape = inspect_npy_shape(path)
                size = int(np.prod(shape))
                if size != n_samples:
                    raise ValueError(
                        f"Score length mismatch for {ratio_label}/{scenario_key}/{filename}"
                    )
                add_row(
                    "Injected", ratio_label, scenario_label, f"retained score: {key}",
                    path, True, f"shape={shape}; flattened_length={size:,}",
                )

            path = require_file(
                scenario_dir / CONTEXT_FILE,
                f"{ratio_label}/{scenario_key} context score",
            )
            shape = inspect_npy_shape(path)
            size = int(np.prod(shape))
            if size != n_samples:
                raise ValueError(
                    f"Context score length mismatch for {ratio_label}/{scenario_key}"
                )
            add_row(
                "Injected", ratio_label, scenario_label, "context score", path, True,
                f"shape={shape}; flattened_length={size:,}",
            )

            archive_path = require_file(
                scenario_dir / DETECTOR_LABEL_ARCHIVE,
                f"{ratio_label}/{scenario_key} detector-label archive",
            )
            with np.load(archive_path, allow_pickle=False) as archive:
                missing = [k for k in NOIF_LABEL_KEYS if k not in archive.files]
                available = sorted(archive.files)
            if missing:
                raise KeyError(
                    f"Missing retained label keys {missing} in {archive_path}"
                )
            add_row(
                "Injected", ratio_label, scenario_label, "retained detector labels",
                archive_path, True,
                f"required_keys={list(NOIF_LABEL_KEYS)}; available_keys={available}",
            )

            full_path = require_file(
                scenario_dir / "full_final_anomaly_score_raw.npy",
                f"{ratio_label}/{scenario_key} existing Full raw score",
            )
            shape = inspect_npy_shape(full_path)
            size = int(np.prod(shape))
            if size != n_samples:
                raise ValueError(
                    f"Existing Full score length mismatch for {ratio_label}/{scenario_key}"
                )
            add_row(
                "Injected", ratio_label, scenario_label,
                "existing Full comparison score", full_path, True,
                f"shape={shape}; flattened_length={size:,}",
            )

    if any("if_feature_score" in r["Resolved_Path"] or "if_embedding_score" in r["Resolved_Path"] for r in rows):
        raise RuntimeError("Preflight manifest unexpectedly includes an IF score input.")

    manifest = pd.DataFrame(rows)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "Source_Preflight_Manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    print("=" * 80)
    print("SOURCE PREFLIGHT PASSED")
    print("=" * 80)
    print(f"N = {n_samples:,}")
    print(f"Checked {len(RATIO_CONFIG)} injection-ratio folders")
    print(f"Checked {len(RATIO_CONFIG) * len(SCENARIOS)} ratio/scenario folders")
    print("Ratio-specific inject_indices.npy files were validated independently.")
    print("No IF score file is present in the Full-without-IF input manifest.")
    print(f"Manifest: {manifest_path.resolve()}")

    return manifest


# ---------------------------------------------------------------------------
# Deterministic exact top-k helpers
# ---------------------------------------------------------------------------

def exact_top_k_order(scores: np.ndarray, k: int) -> np.ndarray:
    """
    Select exactly k indices.

    Ranking:
      1) higher anomaly score first;
      2) lower original pixel index first when scores tie.
    """
    scores = np.asarray(scores).reshape(-1)

    if not np.isfinite(scores).all():
        raise ValueError("Score contains NaN or infinite values.")
    if not 1 <= k <= scores.size:
        raise ValueError(
            f"Invalid top-k size {k} for {scores.size} samples."
        )

    partition_position = scores.size - k
    threshold = np.partition(
        scores,
        partition_position,
    )[partition_position]

    above = np.flatnonzero(scores > threshold)
    tied = np.flatnonzero(scores == threshold)
    need = k - above.size

    if need < 0 or need > tied.size:
        raise RuntimeError(
            "Unexpected deterministic top-k partition result."
        )

    selected = np.concatenate([above, tied[:need]])

    if selected.size != k:
        raise RuntimeError(
            "Deterministic top-k returned an incorrect count."
        )

    order = np.lexsort((selected, -scores[selected]))
    return selected[order]


def exact_top_fraction_labels(
    scores: np.ndarray,
    fraction: float = INTERNAL_SIGNAL_FRACTION,
) -> Tuple[np.ndarray, float, np.ndarray]:
    if not 0.0 < fraction < 1.0:
        raise ValueError("fraction must be between 0 and 1.")

    k = int(np.ceil(float(fraction) * scores.size))
    selected = exact_top_k_order(scores, k)

    labels = np.zeros(scores.size, dtype=np.uint8)
    labels[selected] = 1
    threshold = float(scores[selected[-1]])

    return labels, threshold, selected


def exact_top_fraction_mask(
    scores: np.ndarray,
    fraction: float,
) -> np.ndarray:
    k = int(np.ceil(float(fraction) * scores.size))
    selected = exact_top_k_order(scores, k)
    mask = np.zeros(scores.size, dtype=bool)
    mask[selected] = True
    return mask


# ---------------------------------------------------------------------------
# Score construction
# ---------------------------------------------------------------------------

def average_scores(
    score_dictionary: Mapping[str, np.ndarray],
    keys: Iterable[str],
    n_samples: int,
) -> np.ndarray:
    keys = list(keys)
    if not keys:
        raise ValueError("At least one score is required.")

    result = np.zeros(n_samples, dtype=np.float32)
    weight = np.float32(1.0 / len(keys))

    for key in keys:
        arr = np.asarray(score_dictionary[key], dtype=np.float32)
        if arr.shape != (n_samples,):
            raise ValueError(
                f"{key} has shape {arr.shape}; expected {(n_samples,)}."
            )
        result += arr * weight

    if not np.isfinite(result).all():
        raise ValueError("Base score contains NaN or infinite values.")

    return result


def normalize_using_bounds(
    scores: np.ndarray,
    lower: float,
    upper: float,
) -> np.ndarray:
    if upper <= lower:
        raise ValueError(
            f"Invalid normalization bounds: lower={lower}, upper={upper}."
        )

    normalized = (
        np.clip(scores, lower, upper) - lower
    ) / (upper - lower)

    normalized = normalized.astype(np.float32, copy=False)

    if not np.isfinite(normalized).all():
        raise ValueError("Normalized score contains NaN or infinite values.")

    return normalized


def percentile_bounds(
    scores: np.ndarray,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0,
) -> Tuple[float, float]:
    lower, upper = np.percentile(
        scores,
        [lower_percentile, upper_percentile],
    )
    lower = float(lower)
    upper = float(upper)
    if upper <= lower:
        raise ValueError(
            f"Invalid percentile bounds: p{lower_percentile}={lower}, "
            f"p{upper_percentile}={upper}."
        )
    return lower, upper


def nan_local_mean(
    arr: np.ndarray,
    size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Match the spatial computation in 03_proposed_method:
    uniform_filter(..., mode='nearest') with NaN-aware counts.
    """
    valid = np.isfinite(arr)
    filled = np.where(valid, arr, 0.0).astype(np.float32)

    kernel_area = float(size * size)

    local_sum = (
        uniform_filter(
            filled,
            size=size,
            mode="nearest",
        )
        * kernel_area
    )

    local_count = (
        uniform_filter(
            valid.astype(np.float32),
            size=size,
            mode="nearest",
        )
        * kernel_area
    )

    local_mean = local_sum / np.maximum(local_count, 1.0)
    return local_mean, local_count


def compute_spatial_inconsistency(
    base_score: np.ndarray,
    valid_mask: np.ndarray,
    window_size: int = WINDOW_SIZE,
) -> np.ndarray:
    """
    Recompute spatial inconsistency from the NEW five-signal no-IF base.

    This is the key difference from the old base-only ablation.
    """
    n_samples = base_score.size

    if int(valid_mask.sum()) != n_samples:
        raise ValueError(
            "Valid-mask pixel count does not match the base-score length."
        )

    base_raster = np.full(
        valid_mask.shape,
        np.nan,
        dtype=np.float32,
    )
    base_raster[valid_mask] = base_score

    local_mean, _ = nan_local_mean(
        base_raster,
        size=window_size,
    )

    base_squared = np.where(
        np.isfinite(base_raster),
        base_raster * base_raster,
        np.nan,
    ).astype(np.float32, copy=False)

    local_second_moment, _ = nan_local_mean(
        base_squared,
        size=window_size,
    )

    local_variance = np.maximum(
        local_second_moment - local_mean * local_mean,
        0.0,
    )
    local_std = np.sqrt(local_variance)

    spatial_raster = (
        np.abs(base_raster - local_mean)
        / (local_std + np.float32(EPSILON))
    )

    spatial = spatial_raster[valid_mask].astype(
        np.float32,
        copy=False,
    )

    if spatial.shape != (n_samples,):
        raise RuntimeError("Spatial score has an invalid shape.")
    if not np.isfinite(spatial).all():
        raise ValueError("Spatial score contains NaN or infinite values.")

    return spatial


def calculate_uncertainty(
    score_dictionary: Mapping[str, np.ndarray],
    keys: Iterable[str],
    n_samples: int,
) -> np.ndarray:
    """
    Per-pixel population standard deviation, matching np.std(..., axis=0)
    with ddof=0 while avoiding a large stacked matrix.
    """
    keys = list(keys)
    if not keys:
        raise ValueError("Uncertainty requires at least one signal.")

    score_sum = np.zeros(n_samples, dtype=np.float32)
    squared_sum = np.zeros(n_samples, dtype=np.float32)

    for key in keys:
        current = np.asarray(
            score_dictionary[key],
            dtype=np.float32,
        )
        if current.shape != (n_samples,):
            raise ValueError(
                f"{key} has shape {current.shape}; expected {(n_samples,)}."
            )
        score_sum += current
        squared_sum += current * current

    count = np.float32(len(keys))
    mean = score_sum / count
    variance = squared_sum / count - mean * mean
    variance = np.maximum(variance, 0.0)

    uncertainty = np.sqrt(variance).astype(
        np.float32,
        copy=False,
    )

    if not np.isfinite(uncertainty).all():
        raise ValueError("Uncertainty contains NaN or infinite values.")

    return uncertainty


def construct_reliability(
    score_dictionary: Mapping[str, np.ndarray],
    component_labels: Mapping[str, np.ndarray],
    spatial_labels: np.ndarray,
    context_labels: np.ndarray,
    uncertainty_bounds: Tuple[float, float] | None,
    n_samples: int,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    Tuple[float, float],
    np.ndarray,
]:
    """
    Full-without-IF reliability uses exactly seven signals:
      AE, LOF_f, MD_f, LOF_e, MD_e, new no-IF spatial, context.
    """
    reliability_keys = list(NOIF_COMPONENT_FILES.keys()) + [
        "spatial_noif",
        "context",
    ]

    agreement_count = np.zeros(n_samples, dtype=np.uint8)
    for key in NOIF_COMPONENT_FILES.keys():
        agreement_count += np.asarray(
            component_labels[key],
            dtype=np.uint8,
        )

    agreement_count += spatial_labels.astype(np.uint8, copy=False)
    agreement_count += context_labels.astype(np.uint8, copy=False)

    if agreement_count.max(initial=0) > len(reliability_keys):
        raise RuntimeError("Agreement count exceeds number of signals.")

    agreement_fraction = (
        agreement_count.astype(np.float32)
        / np.float32(len(reliability_keys))
    )

    uncertainty_raw = calculate_uncertainty(
        score_dictionary,
        reliability_keys,
        n_samples,
    )

    if uncertainty_bounds is None:
        uncertainty_min = float(uncertainty_raw.min())
        uncertainty_max = float(uncertainty_raw.max())
    else:
        uncertainty_min = float(uncertainty_bounds[0])
        uncertainty_max = float(uncertainty_bounds[1])

    if uncertainty_max <= uncertainty_min:
        raise ValueError("Invalid uncertainty normalization range.")

    uncertainty_norm = normalize_using_bounds(
        uncertainty_raw,
        uncertainty_min,
        uncertainty_max,
    )

    reliability = (
        np.float32(AGREEMENT_WEIGHT) * agreement_fraction
        + np.float32(CERTAINTY_WEIGHT) * (1.0 - uncertainty_norm)
    )

    reliability = np.clip(
        reliability,
        0.0,
        1.0,
    ).astype(np.float32, copy=False)

    if not np.isfinite(reliability).all():
        raise ValueError("Reliability contains NaN or infinite values.")

    return (
        agreement_count,
        agreement_fraction,
        uncertainty_raw,
        uncertainty_norm,
        (uncertainty_min, uncertainty_max),
        reliability,
    )


def construct_noif_pipeline(
    input_scores: Dict[str, np.ndarray],
    component_labels: Dict[str, np.ndarray],
    valid_mask: np.ndarray,
    calibration: Mapping[str, float] | None = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """
    Recompute the complete downstream Full-without-IF pipeline.

    calibration=None:
        derive all new downstream bounds from CLEAN data.

    calibration=dict:
        reuse CLEAN no-IF bounds for injected data.
    """
    n_samples = next(iter(input_scores.values())).size

    for key, arr in input_scores.items():
        if np.asarray(arr).shape != (n_samples,):
            raise ValueError(
                f"Input score '{key}' has invalid shape "
                f"{np.asarray(arr).shape}."
            )

    # 1) New five-signal base.
    base = average_scores(
        input_scores,
        NOIF_COMPONENT_FILES.keys(),
        n_samples,
    )

    # 2) New spatial inconsistency from the NEW no-IF base.
    spatial_raw = compute_spatial_inconsistency(
        base,
        valid_mask,
        WINDOW_SIZE,
    )

    if calibration is None:
        spatial_p1, spatial_p99 = percentile_bounds(spatial_raw)
    else:
        spatial_p1 = float(calibration["spatial_raw_p1"])
        spatial_p99 = float(calibration["spatial_raw_p99"])

    spatial_norm = normalize_using_bounds(
        spatial_raw,
        spatial_p1,
        spatial_p99,
    )

    # 3) New base-spatial refined score.
    refined_raw = (
        np.float32(BASE_WEIGHT) * base
        + np.float32(SPATIAL_WEIGHT) * spatial_norm
    ).astype(np.float32, copy=False)

    if calibration is None:
        refined_p1, refined_p99 = percentile_bounds(refined_raw)
    else:
        refined_p1 = float(calibration["refined_p1"])
        refined_p99 = float(calibration["refined_p99"])

    refined_norm = normalize_using_bounds(
        refined_raw,
        refined_p1,
        refined_p99,
    )

    # 4) New spatial and context flags for reliability.
    spatial_labels, spatial_threshold, _ = exact_top_fraction_labels(
        spatial_norm,
        INTERNAL_SIGNAL_FRACTION,
    )
    context_labels, context_threshold, _ = exact_top_fraction_labels(
        input_scores["context"],
        INTERNAL_SIGNAL_FRACTION,
    )

    reliability_scores = dict(input_scores)
    reliability_scores["spatial_noif"] = spatial_norm

    if calibration is None:
        uncertainty_bounds = None
    else:
        uncertainty_bounds = (
            float(calibration["uncertainty_min"]),
            float(calibration["uncertainty_max"]),
        )

    (
        agreement_count,
        agreement_fraction,
        uncertainty_raw,
        uncertainty_norm,
        derived_uncertainty_bounds,
        reliability,
    ) = construct_reliability(
        reliability_scores,
        component_labels,
        spatial_labels,
        context_labels,
        uncertainty_bounds,
        n_samples,
    )

    # 5) Final pre-clipping score used for ranking.
    final_raw = (
        refined_norm * reliability
    ).astype(np.float32, copy=False)

    if calibration is None:
        final_p1, final_p99 = percentile_bounds(final_raw)
    else:
        final_p1 = float(calibration["final_p1"])
        final_p99 = float(calibration["final_p99"])

    final_norm = normalize_using_bounds(
        final_raw,
        final_p1,
        final_p99,
    )

    final_labels, final_top2_threshold, final_top2_indices = (
        exact_top_fraction_labels(
            final_raw,
            INTERNAL_SIGNAL_FRACTION,
        )
    )

    outputs = {
        "base": base,
        "spatial_raw": spatial_raw,
        "spatial_norm": spatial_norm,
        "refined_raw": refined_raw,
        "refined_norm": refined_norm,
        "spatial_labels": spatial_labels,
        "context_labels": context_labels,
        "agreement_count": agreement_count,
        "agreement_fraction": agreement_fraction,
        "uncertainty_raw": uncertainty_raw,
        "uncertainty_norm": uncertainty_norm,
        "reliability": reliability,
        "final_raw": final_raw,
        "final_norm": final_norm,
        "final_labels": final_labels,
        "final_top2_indices": final_top2_indices.astype(
            np.int64,
            copy=False,
        ),
    }

    derived_calibration = {
        "spatial_raw_p1": float(spatial_p1),
        "spatial_raw_p99": float(spatial_p99),
        "refined_p1": float(refined_p1),
        "refined_p99": float(refined_p99),
        "uncertainty_min": float(derived_uncertainty_bounds[0]),
        "uncertainty_max": float(derived_uncertainty_bounds[1]),
        "final_p1": float(final_p1),
        "final_p99": float(final_p99),
        "spatial_internal_threshold": float(spatial_threshold),
        "context_internal_threshold": float(context_threshold),
        "final_top2_raw_threshold": float(final_top2_threshold),
        "base_weight": float(BASE_WEIGHT),
        "spatial_weight": float(SPATIAL_WEIGHT),
        "agreement_weight": float(AGREEMENT_WEIGHT),
        "certainty_weight": float(CERTAINTY_WEIGHT),
        "window_size": int(WINDOW_SIZE),
        "internal_signal_fraction": float(INTERNAL_SIGNAL_FRACTION),
        "base_signal_count": 5,
        "reliability_signal_count": 7,
    }

    return outputs, derived_calibration


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def save_pipeline_outputs(
    output_dir: Path,
    outputs: Mapping[str, np.ndarray],
    calibration: Mapping[str, float],
    save_diagnostics: bool = True,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Essential standalone artifacts.
    np.save(
        output_dir / "full_noif_base_score.npy",
        outputs["base"].astype(np.float32, copy=False),
    )
    np.save(
        output_dir / "full_noif_spatial_score_normalized.npy",
        outputs["spatial_norm"].astype(np.float32, copy=False),
    )
    np.save(
        output_dir / "full_noif_refined_score_normalized.npy",
        outputs["refined_norm"].astype(np.float32, copy=False),
    )
    np.save(
        output_dir / "full_noif_reliability.npy",
        outputs["reliability"].astype(np.float32, copy=False),
    )
    np.save(
        output_dir / "full_noif_final_anomaly_score_raw.npy",
        outputs["final_raw"].astype(np.float32, copy=False),
    )
    np.save(
        output_dir / "full_noif_final_anomaly_score.npy",
        outputs["final_norm"].astype(np.float32, copy=False),
    )
    np.save(
        output_dir / "full_noif_final_anomaly_labels.npy",
        outputs["final_labels"].astype(np.uint8, copy=False),
    )
    np.save(
        output_dir / "FullNoIF_top2pct_selected_indices.npy",
        outputs["final_top2_indices"].astype(np.int64, copy=False),
    )

    if save_diagnostics:
        np.savez_compressed(
            output_dir / "full_noif_diagnostics.npz",
            spatial_score_raw=outputs["spatial_raw"].astype(
                np.float32,
                copy=False,
            ),
            refined_score_raw=outputs["refined_raw"].astype(
                np.float32,
                copy=False,
            ),
            spatial_labels=outputs["spatial_labels"].astype(
                np.uint8,
                copy=False,
            ),
            context_labels=outputs["context_labels"].astype(
                np.uint8,
                copy=False,
            ),
            agreement_count=outputs["agreement_count"].astype(
                np.uint8,
                copy=False,
            ),
            agreement_fraction=outputs["agreement_fraction"].astype(
                np.float32,
                copy=False,
            ),
            uncertainty_raw=outputs["uncertainty_raw"].astype(
                np.float32,
                copy=False,
            ),
            uncertainty_normalized=outputs["uncertainty_norm"].astype(
                np.float32,
                copy=False,
            ),
        )

    np.savez(
        output_dir / "full_noif_calibration_params.npz",
        **{
            key: np.asarray(value)
            for key, value in calibration.items()
        },
    )


# ---------------------------------------------------------------------------
# Evaluation and summaries
# ---------------------------------------------------------------------------

def evaluate_method(
    scores: np.ndarray,
    injected_indices: np.ndarray,
    budget_ratio: float,
) -> Dict[str, float]:
    n = scores.size
    k = int(np.ceil(budget_ratio * n))
    selected = exact_top_k_order(scores, k)

    injected_mask = np.zeros(n, dtype=bool)
    injected_mask[injected_indices] = True

    detected = int(injected_mask[selected].sum())
    total = int(injected_indices.size)
    missed = total - detected

    return {
        "Budget_Fraction": float(budget_ratio),
        "Budget_Percent": float(100.0 * budget_ratio),
        "Selected_Count": int(k),
        "Total_Injected": total,
        "Detected_Injected": detected,
        "Missed_Injected": missed,
        "Detection_Rate_Percent": float(
            100.0 * detected / total
        ),
        "Miss_Rate_Percent": float(
            100.0 * missed / total
        ),
        "Raw_Final_Score_Threshold": float(scores[selected[-1]]),
    }


def jaccard(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    intersection = int(np.logical_and(mask_a, mask_b).sum())
    union = int(np.logical_or(mask_a, mask_b).sum())
    return float(intersection / union) if union else 1.0


def format_workbook(path: Path) -> None:
    wb = load_workbook(path)

    for ws in wb.worksheets:
        ws.freeze_panes = "A2"
        ws.auto_filter.ref = ws.dimensions

        for cell in ws[1]:
            cell.font = Font(bold=True)
            cell.alignment = Alignment(
                horizontal="center",
                vertical="center",
                wrap_text=True,
            )

        for row in ws.iter_rows(min_row=2):
            for cell in row:
                cell.alignment = Alignment(
                    vertical="top",
                    wrap_text=True,
                )

        for col_cells in ws.columns:
            max_len = max(
                len(str(cell.value))
                if cell.value is not None
                else 0
                for cell in col_cells
            )
            ws.column_dimensions[
                get_column_letter(col_cells[0].column)
            ].width = min(max_len + 2, 55)

    wb.save(path)


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment(
    root_dir: Path,
    output_root: Path,
    save_diagnostics: bool = True,
) -> None:
    clean_dir = root_dir / "03_Proposed_Results_Seed_42"
    comparison_dir = (
        root_dir / "04_Baseline_Comparison_Results_Seed_42"
    )
    injected_root = (
        root_dir / "03_Injected_Proposed_Results_Seed_42"
    )

    require_file(
        clean_dir / next(iter(NOIF_COMPONENT_FILES.values())),
        "clean proposed-method component score",
    )

    # ------------------------------------------------------------------
    # A. CLEAN DATA: derive the new no-IF downstream calibration.
    # ------------------------------------------------------------------
    print("=" * 80)
    print("A. CLEAN FULL-WITHOUT-IF RECOMPUTATION")
    print("=" * 80)

    clean_scores = load_noif_scores(clean_dir)
    n_samples = clean_scores["ae"].size

    valid_mask_path = find_valid_mask(root_dir)
    valid_mask = load_valid_mask(
        valid_mask_path,
        n_samples,
    )

    clean_labels = load_saved_component_labels(
        clean_dir,
        n_samples,
    )

    clean_outputs, clean_calibration = (
        construct_noif_pipeline(
            clean_scores,
            clean_labels,
            valid_mask,
            calibration=None,
        )
    )

    clean_output_dir = output_root / "Clean"
    save_pipeline_outputs(
        clean_output_dir,
        clean_outputs,
        clean_calibration,
        save_diagnostics=save_diagnostics,
    )

    # Existing Full score is used only for comparison.
    existing_full_clean_path = require_file(
        comparison_dir / "Full_final_fused_score_preclip.npy",
        "existing clean Full pre-final-clipping score",
    )
    existing_full_clean = load_vector(
        existing_full_clean_path,
        expected_length=n_samples,
    )

    full_clean_mask = exact_top_fraction_mask(
        existing_full_clean,
        INTERNAL_SIGNAL_FRACTION,
    )
    noif_clean_mask = clean_outputs["final_labels"].astype(bool)

    clean_intersection = int(
        np.logical_and(
            full_clean_mask,
            noif_clean_mask,
        ).sum()
    )
    clean_union = int(
        np.logical_or(
            full_clean_mask,
            noif_clean_mask,
        ).sum()
    )
    clean_jaccard = jaccard(
        full_clean_mask,
        noif_clean_mask,
    )
    clean_replacement = float(
        100.0
        * (
            full_clean_mask.sum()
            - clean_intersection
        )
        / full_clean_mask.sum()
    )

    clean_comparison_df = pd.DataFrame(
        [
            {
                "Metric": "Samples",
                "Existing Full": n_samples,
                "Full without IF (end-to-end)": n_samples,
            },
            {
                "Metric": "Exact top-2% selected",
                "Existing Full": int(full_clean_mask.sum()),
                "Full without IF (end-to-end)": int(
                    noif_clean_mask.sum()
                ),
            },
            {
                "Metric": "Top-2% intersection",
                "Existing Full": clean_intersection,
                "Full without IF (end-to-end)": clean_intersection,
            },
            {
                "Metric": "Top-2% union",
                "Existing Full": clean_union,
                "Full without IF (end-to-end)": clean_union,
            },
            {
                "Metric": "Top-2% Jaccard",
                "Existing Full": clean_jaccard,
                "Full without IF (end-to-end)": clean_jaccard,
            },
            {
                "Metric": "Replacement vs Existing Full (%)",
                "Existing Full": 0.0,
                "Full without IF (end-to-end)": clean_replacement,
            },
            {
                "Metric": "Mean raw final score",
                "Existing Full": float(
                    np.mean(existing_full_clean, dtype=np.float64)
                ),
                "Full without IF (end-to-end)": float(
                    np.mean(
                        clean_outputs["final_raw"],
                        dtype=np.float64,
                    )
                ),
            },
        ]
    )

    calibration_df = pd.DataFrame(
        [
            {"Parameter": key, "Value": value}
            for key, value in clean_calibration.items()
        ]
    )

    # ------------------------------------------------------------------
    # B. INJECTED DATA: all ratios x all scenarios.
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("B. INJECTED FULL-WITHOUT-IF RECOMPUTATION")
    print("=" * 80)

    detailed_rows = []
    validation_rows = []

    for ratio_label, ratio_cfg in RATIO_CONFIG.items():
        ratio_value = float(ratio_cfg["ratio"])
        ratio_source_dir = (
            injected_root / str(ratio_cfg["folder"])
        )

        inject_indices_path = require_file(
            ratio_source_dir / "inject_indices.npy",
            f"{ratio_label} injected-index file",
        )
        injected_indices = np.load(
            inject_indices_path,
            allow_pickle=False,
        ).astype(np.int64, copy=False).reshape(-1)

        if injected_indices.size == 0:
            raise ValueError(
                f"{ratio_label}: inject_indices.npy is empty."
            )
        if np.unique(injected_indices).size != injected_indices.size:
            raise ValueError(
                f"{ratio_label}: inject_indices.npy contains duplicates."
            )
        if (
            injected_indices.min() < 0
            or injected_indices.max() >= n_samples
        ):
            raise ValueError(
                f"{ratio_label}: injected index out of range."
            )

        expected_injected = int(
            math.floor(ratio_value * n_samples)
        )
        if injected_indices.size != expected_injected:
            raise ValueError(
                f"{ratio_label}: found {injected_indices.size:,} "
                f"injected indices; expected {expected_injected:,} "
                f"for N={n_samples:,}."
            )

        ratio_output_dir = (
            output_root
            / "Injected"
            / str(ratio_cfg["folder"])
        )
        ratio_output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        for scenario_key, scenario_label in SCENARIOS.items():
            print(
                f"\n{ratio_label} | {scenario_label}"
            )

            source_scenario_dir = (
                ratio_source_dir / scenario_key
            )
            if not source_scenario_dir.is_dir():
                raise FileNotFoundError(
                    "Missing injected scenario folder:\n"
                    f"{source_scenario_dir.resolve()}"
                )

            scenario_scores = load_noif_scores(
                source_scenario_dir,
                expected_length=n_samples,
            )
            scenario_labels = load_saved_component_labels(
                source_scenario_dir,
                n_samples,
            )

            noif_outputs, applied_calibration = (
                construct_noif_pipeline(
                    scenario_scores,
                    scenario_labels,
                    valid_mask,
                    calibration=clean_calibration,
                )
            )

            scenario_output_dir = (
                ratio_output_dir / scenario_key
            )
            save_pipeline_outputs(
                scenario_output_dir,
                noif_outputs,
                applied_calibration,
                save_diagnostics=save_diagnostics,
            )

            existing_full_path = require_file(
                source_scenario_dir
                / "full_final_anomaly_score_raw.npy",
                (
                    f"existing Full raw final score for "
                    f"{ratio_label}/{scenario_key}"
                ),
            )
            existing_full_score = load_vector(
                existing_full_path,
                expected_length=n_samples,
            )

            for method_name, score in [
                ("Existing Full", existing_full_score),
                (
                    "Full without IF (end-to-end)",
                    noif_outputs["final_raw"],
                ),
            ]:
                for budget_label, budget_ratio in BUDGETS.items():
                    row = evaluate_method(
                        score,
                        injected_indices,
                        budget_ratio,
                    )
                    row.update(
                        {
                            "Injection_Ratio": ratio_label,
                            "Scenario_Key": scenario_key,
                            "Scenario": scenario_label,
                            "Method": method_name,
                            "Budget": budget_label,
                        }
                    )
                    detailed_rows.append(row)

            # Per-scenario top-2% overlap between current Full and no-IF.
            existing_full_top2 = exact_top_fraction_mask(
                existing_full_score,
                INTERNAL_SIGNAL_FRACTION,
            )
            noif_top2 = noif_outputs["final_labels"].astype(bool)

            validation_rows.append(
                {
                    "Injection_Ratio": ratio_label,
                    "Scenario": scenario_label,
                    "Check": "No-IF base signal count",
                    "Passed": True,
                    "Details": "5 (AE, LOF_f, MD_f, LOF_e, MD_e)",
                }
            )
            validation_rows.append(
                {
                    "Injection_Ratio": ratio_label,
                    "Scenario": scenario_label,
                    "Check": "No-IF reliability signal count",
                    "Passed": True,
                    "Details": "7 (5 base + new spatial + context)",
                }
            )
            validation_rows.append(
                {
                    "Injection_Ratio": ratio_label,
                    "Scenario": scenario_label,
                    "Check": "Clean calibration reused",
                    "Passed": all(
                        np.isclose(
                            float(applied_calibration[key]),
                            float(clean_calibration[key]),
                            atol=1e-12,
                            rtol=0.0,
                        )
                        for key in [
                            "spatial_raw_p1",
                            "spatial_raw_p99",
                            "refined_p1",
                            "refined_p99",
                            "uncertainty_min",
                            "uncertainty_max",
                            "final_p1",
                            "final_p99",
                        ]
                    ),
                    "Details": (
                        "New downstream normalization/reliability bounds "
                        "derived on clean no-IF data and frozen for injection."
                    ),
                }
            )
            validation_rows.append(
                {
                    "Injection_Ratio": ratio_label,
                    "Scenario": scenario_label,
                    "Check": "Exact top-2% no-IF count",
                    "Passed": int(noif_top2.sum())
                    == int(np.ceil(
                        INTERNAL_SIGNAL_FRACTION * n_samples
                    )),
                    "Details": (
                        f"Selected {int(noif_top2.sum()):,} pixels."
                    ),
                }
            )
            validation_rows.append(
                {
                    "Injection_Ratio": ratio_label,
                    "Scenario": scenario_label,
                    "Check": "Current Full vs no-IF top-2% Jaccard",
                    "Passed": True,
                    "Details": f"{jaccard(existing_full_top2, noif_top2):.6f}",
                }
            )

    results_df = pd.DataFrame(detailed_rows)
    validation_df = pd.DataFrame(validation_rows)

    # Add direct no-IF minus Full delta at every ratio/scenario/budget.
    pivot = results_df.pivot_table(
        index=[
            "Injection_Ratio",
            "Scenario_Key",
            "Scenario",
            "Budget",
            "Budget_Fraction",
            "Budget_Percent",
        ],
        columns="Method",
        values="Detection_Rate_Percent",
        aggfunc="first",
    ).reset_index()
    pivot.columns.name = None
    pivot["Delta_NoIF_minus_Full_pp"] = (
        pivot["Full without IF (end-to-end)"]
        - pivot["Existing Full"]
    )

    mean_across_scenarios_df = (
        results_df.groupby(
            [
                "Injection_Ratio",
                "Method",
                "Budget",
                "Budget_Fraction",
                "Budget_Percent",
            ],
            as_index=False,
        )
        .agg(
            Mean_Detection_Rate_Percent=(
                "Detection_Rate_Percent",
                "mean",
            ),
            Standard_Deviation=(
                "Detection_Rate_Percent",
                "std",
            ),
            Minimum_Detection_Rate_Percent=(
                "Detection_Rate_Percent",
                "min",
            ),
            Maximum_Detection_Rate_Percent=(
                "Detection_Rate_Percent",
                "max",
            ),
        )
    )

    mean_pivot = mean_across_scenarios_df.pivot_table(
        index=[
            "Injection_Ratio",
            "Budget",
            "Budget_Fraction",
            "Budget_Percent",
        ],
        columns="Method",
        values="Mean_Detection_Rate_Percent",
        aggfunc="first",
    ).reset_index()
    mean_pivot.columns.name = None
    mean_pivot["Delta_NoIF_minus_Full_pp"] = (
        mean_pivot["Full without IF (end-to-end)"]
        - mean_pivot["Existing Full"]
    )

    # One aggregate across all three ratios and all five scenarios.
    all_ratio_summary_df = (
        results_df.groupby(
            [
                "Method",
                "Budget",
                "Budget_Fraction",
                "Budget_Percent",
            ],
            as_index=False,
        )
        .agg(
            Mean_Across_15_RatioScenario_Combinations=(
                "Detection_Rate_Percent",
                "mean",
            ),
            Standard_Deviation=(
                "Detection_Rate_Percent",
                "std",
            ),
        )
    )

    all_ratio_pivot = all_ratio_summary_df.pivot_table(
        index=[
            "Budget",
            "Budget_Fraction",
            "Budget_Percent",
        ],
        columns="Method",
        values="Mean_Across_15_RatioScenario_Combinations",
        aggfunc="first",
    ).reset_index()
    all_ratio_pivot.columns.name = None
    all_ratio_pivot["Delta_NoIF_minus_Full_pp"] = (
        all_ratio_pivot["Full without IF (end-to-end)"]
        - all_ratio_pivot["Existing Full"]
    )

    settings_df = pd.DataFrame(
        [
            {
                "Setting": "Experiment",
                "Value": (
                    "Full without IF with complete downstream "
                    "recomputation"
                ),
            },
            {
                "Setting": "What is removed",
                "Value": (
                    "Feature-space IF and embedding-space IF"
                ),
            },
            {
                "Setting": "Retained base signals",
                "Value": (
                    "AE, feature-space LOF, feature-space MD, "
                    "embedding-space LOF, embedding-space MD"
                ),
            },
            {
                "Setting": "Base signal count",
                "Value": 5,
            },
            {
                "Setting": "Spatial definition",
                "Value": (
                    "Recomputed 5x5 local z-score magnitude from "
                    "the new five-signal no-IF base"
                ),
            },
            {
                "Setting": "Base/spatial weights",
                "Value": f"{BASE_WEIGHT}/{SPATIAL_WEIGHT}",
            },
            {
                "Setting": "Context",
                "Value": (
                    "Existing normalized context-consistency signal "
                    "reused; context model is not retrained"
                ),
            },
            {
                "Setting": "Reliability signals",
                "Value": (
                    "5 retained base components + new no-IF spatial "
                    "+ existing context = 7 signals"
                ),
            },
            {
                "Setting": "Reliability weights",
                "Value": (
                    f"{AGREEMENT_WEIGHT} agreement + "
                    f"{CERTAINTY_WEIGHT} certainty"
                ),
            },
            {
                "Setting": "Internal anomaly flags",
                "Value": (
                    "Saved exact detector labels for retained components; "
                    "deterministic exact top-2% for new spatial and context"
                ),
            },
            {
                "Setting": "Injected-data calibration",
                "Value": (
                    "All newly derived spatial/refined/uncertainty/final "
                    "bounds come from clean no-IF data and are frozen"
                ),
            },
            {
                "Setting": "Final ranking score",
                "Value": (
                    "Pre-final-clipping reliability-weighted score"
                ),
            },
            {
                "Setting": "Final selection",
                "Value": (
                    "Deterministic exact top-k; ties resolved by lower "
                    "original pixel index"
                ),
            },
            {
                "Setting": "Injection ratios",
                "Value": ", ".join(RATIO_CONFIG.keys()),
            },
            {
                "Setting": "Scenarios",
                "Value": ", ".join(SCENARIOS.values()),
            },
            {
                "Setting": "Budgets",
                "Value": ", ".join(BUDGETS.keys()),
            },
            {
                "Setting": "Models retrained",
                "Value": False,
            },
            {
                "Setting": "Source clean folder",
                "Value": str(clean_dir),
            },
            {
                "Setting": "Source injected folder",
                "Value": str(injected_root),
            },
            {
                "Setting": "Valid mask",
                "Value": str(valid_mask_path),
            },
            {
                "Setting": "Output folder",
                "Value": str(output_root),
            },
        ]
    )

    # General validation.
    expected_result_rows = (
        len(RATIO_CONFIG)
        * len(SCENARIOS)
        * 2
        * len(BUDGETS)
    )
    global_validation = pd.DataFrame(
        [
            {
                "Check": "Expected detailed result rows",
                "Passed": len(results_df) == expected_result_rows,
                "Details": (
                    f"{len(results_df)} rows; expected "
                    f"{expected_result_rows}"
                ),
            },
            {
                "Check": "All detection rates valid",
                "Passed": bool(
                    results_df["Detection_Rate_Percent"]
                    .between(0.0, 100.0)
                    .all()
                ),
                "Details": "All rates must be in [0,100].",
            },
            {
                "Check": "All final selected counts exact",
                "Passed": bool(
                    (
                        results_df["Selected_Count"]
                        == results_df["Budget_Fraction"].apply(
                            lambda x: int(np.ceil(x * n_samples))
                        )
                    ).all()
                ),
                "Details": "ceil(budget * N) for every row.",
            },
            {
                "Check": "No IF input loaded",
                "Passed": True,
                "Details": (
                    "NOIF_COMPONENT_FILES contains no IF feature "
                    "or IF embedding score."
                ),
            },
            {
                "Check": "Clean no-IF top-2% exact",
                "Passed": int(noif_clean_mask.sum())
                == int(np.ceil(
                    INTERNAL_SIGNAL_FRACTION * n_samples
                )),
                "Details": (
                    f"{int(noif_clean_mask.sum()):,} selected."
                ),
            },
        ]
    )

    if not global_validation["Passed"].all():
        raise RuntimeError(
            "One or more global validation checks failed."
        )
    if not validation_df[
        validation_df["Check"] != (
            "Current Full vs no-IF top-2% Jaccard"
        )
    ]["Passed"].all():
        raise RuntimeError(
            "One or more scenario-level validation checks failed."
        )

    # ------------------------------------------------------------------
    # C. SAVE CONSOLIDATED REPORTS
    # ------------------------------------------------------------------
    output_root.mkdir(parents=True, exist_ok=True)

    results_csv = (
        output_root
        / "Full_Without_IF_EndToEnd_Detailed_Results.csv"
    )
    results_df.to_csv(results_csv, index=False)

    comparison_csv = (
        output_root
        / "Full_Without_IF_EndToEnd_Delta_vs_Full.csv"
    )
    pivot.to_csv(comparison_csv, index=False)

    workbook_path = (
        output_root
        / "Full_Without_IF_EndToEnd_Results.xlsx"
    )

    with pd.ExcelWriter(
        workbook_path,
        engine="openpyxl",
    ) as writer:
        settings_df.to_excel(
            writer,
            sheet_name="Settings",
            index=False,
        )
        calibration_df.to_excel(
            writer,
            sheet_name="Clean Calibration",
            index=False,
        )
        clean_comparison_df.to_excel(
            writer,
            sheet_name="Clean Comparison",
            index=False,
        )
        results_df.to_excel(
            writer,
            sheet_name="Injected Detailed",
            index=False,
        )
        pivot.to_excel(
            writer,
            sheet_name="Delta vs Full",
            index=False,
        )
        mean_across_scenarios_df.to_excel(
            writer,
            sheet_name="Ratio Means Long",
            index=False,
        )
        mean_pivot.to_excel(
            writer,
            sheet_name="Ratio Means Compare",
            index=False,
        )
        all_ratio_summary_df.to_excel(
            writer,
            sheet_name="All15 Means Long",
            index=False,
        )
        all_ratio_pivot.to_excel(
            writer,
            sheet_name="All15 Means Compare",
            index=False,
        )
        validation_df.to_excel(
            writer,
            sheet_name="Scenario Validation",
            index=False,
        )
        global_validation.to_excel(
            writer,
            sheet_name="Global Validation",
            index=False,
        )

    format_workbook(workbook_path)

    scope_text = f"""FULL WITHOUT IF -- END-TO-END DOWNSTREAM RECOMPUTATION

Seed: {SEED}

Definition:
Feature-space IF and embedding-space IF are removed from the Full base.
The new base contains exactly five signals:
1. AE reconstruction
2. Feature-space LOF
3. Feature-space MD
4. Embedding-space LOF
5. Embedding-space MD

Unlike the previous "Full base without IF" ablation, this experiment
recomputes every downstream quantity that depends on the Full base:
- spatial inconsistency from the new five-signal base;
- spatial normalization using clean no-IF bounds;
- 0.75/0.25 base-spatial fusion and clean-calibrated normalization;
- seven-signal agreement and uncertainty;
- reliability using 0.60 agreement + 0.40 certainty;
- final reliability-weighted pre-clipping score;
- deterministic exact top-k selections.

The existing context score is reused because its autoencoder does not depend
on IF. No MVAE, LOF, MD, context model, preprocessing object, or injected
dataset is retrained/regenerated.

All new downstream calibration values are derived from the original clean
dataset and then frozen for all injected scenarios.

Injection ratios:
{", ".join(RATIO_CONFIG.keys())}

Scenarios:
{", ".join(SCENARIOS.values())}

Budgets:
{", ".join(BUDGETS.keys())}

Clean top-2% Jaccard (Existing Full vs Full-noIF):
{clean_jaccard:.8f}

Clean top-2% replacement relative to Existing Full:
{clean_replacement:.4f}%
"""
    (output_root / "Full_Without_IF_EndToEnd_Scope.txt").write_text(
        scope_text,
        encoding="utf-8",
    )

    print("\n" + "=" * 80)
    print("FULL-WITHOUT-IF END-TO-END EXPERIMENT COMPLETED")
    print("=" * 80)
    print(f"Samples: {n_samples:,}")
    print(
        "Clean Full vs Full-noIF top-2% Jaccard: "
        f"{clean_jaccard:.6f}"
    )
    print(
        "Clean replacement vs Existing Full: "
        f"{clean_replacement:.2f}%"
    )
    print("\nMain workbook:")
    print(workbook_path.resolve())
    print("\nDetailed CSV:")
    print(results_csv.resolve())
    print("\nDelta CSV:")
    print(comparison_csv.resolve())
    print("\nAll new outputs:")
    print(output_root.resolve())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recompute the complete downstream Full-without-IF "
            "variant and evaluate all injected ratios/scenarios."
        )
    )
    parser.add_argument(
        "--root-dir",
        type=Path,
        default=DEFAULT_ROOT,
        help=(
            "Project Code directory. Defaults to the current "
            "Windows OneDrive project path."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Optional output directory. Default: "
            "<root-dir>/09_Full_Without_IF_EndToEnd_Seed_42"
        ),
    )
    parser.add_argument(
        "--no-diagnostics",
        action="store_true",
        help=(
            "Do not save compressed large diagnostic NPZ archives. "
            "Essential score/reliability/final arrays are still saved."
        ),
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help=(
            "Validate and record every clean/ratio/scenario source path "
            "without recomputing Full-without-IF scores."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root_dir = args.root_dir.expanduser().resolve()

    output_root = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else root_dir / "09_Full_Without_IF_EndToEnd_Seed_42"
    )

    if not root_dir.exists():
        raise FileNotFoundError(
            f"Project root does not exist:\n{root_dir}"
        )

    # Prevent accidental writing into an established source-results folder.
    protected = {
        (root_dir / name).resolve()
        for name in [
            "03_Proposed_Results_Seed_42",
            "03_Injected_Proposed_Results_Seed_42",
            "04_Baseline_Comparison_Results_Seed_42",
            "05_Reliability_Agreement_Results_Seed_42",
            "06_Ablation_Study_Results_Seed_42",
            "07_Stability_Analysis_Results",
            "08_Weight_Sensitivity_Results_Seed_42",
        ]
    }
    if output_root.resolve() in protected:
        raise ValueError(
            "Refusing to write the new experiment into an "
            "established source-results folder."
        )

    preflight_source_tree(
        root_dir=root_dir,
        output_root=output_root,
    )

    if args.preflight_only:
        print("Preflight-only mode requested; no score recomputation was performed.")
        return

    run_experiment(
        root_dir=root_dir,
        output_root=output_root,
        save_diagnostics=not args.no_diagnostics,
    )


if __name__ == "__main__":
    main()
