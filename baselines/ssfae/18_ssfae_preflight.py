"""
SSF-AE baseline preflight for the heterogeneous geospatial anomaly-detection paper.

Purpose
-------
This script performs ONLY non-training diagnostics needed before implementing
the SSF-AE baseline (Zhang et al., Neurocomputing 2026).

It does not modify any existing result directory.

Expected project files in ROOT_DIR
----------------------------------
X.npy
valid_mask_1.npy

The mapping is taken from the existing feature-engineering pipeline:
    X[row] is constructed by indexing each raster feature with valid_mask_1.npy
in NumPy boolean-mask order.

Outputs
-------
18_SSF_AE_Baseline_Results_Seed_42/preflight/
    SSFAE_Preflight_Summary.json
    SSFAE_PCA_Explained_Variance.csv
    SSFAE_Window_Diagnostics.csv

Run
---
python 18_ssfae_preflight.py

Optional:
python 18_ssfae_preflight.py --root "C:\\...\\Outlier Paper\\Code"
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
from pathlib import Path

import numpy as np
import pandas as pd

EXPECTED_N = 4_290_958
EXPECTED_D = 23
EXPECTED_MASK_SHAPE = (2897, 2696)

PCA_TARGET = 0.9990
WINDOW = 40
STRIDES_TO_CHECK = (10, 20)
VIT_PATCH_DEPTH = 4
CNN_POOL_FACTOR_DEPTH = 4
CHUNK_ROWS = 100_000

DEFAULT_ROOT = Path(os.environ.get("GIS_ANOMALY_PROJECT_ROOT", Path.cwd())).resolve()
OUTPUT_NAME = "18_SSF_AE_Baseline_Results_Seed_42"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--chunk-rows", type=int, default=CHUNK_ROWS)
    return p.parse_args()


def jsonable(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, Path):
        return str(x)
    return x


def load_and_validate(root: Path):
    x_path = root / "X.npy"
    mask_path = root / "valid_mask_1.npy"

    if not x_path.exists():
        raise FileNotFoundError(f"Missing required file: {x_path}")
    if not mask_path.exists():
        raise FileNotFoundError(f"Missing required file: {mask_path}")

    X = np.load(x_path, mmap_mode="r")
    mask = np.load(mask_path, mmap_mode="r")

    if X.ndim != 2:
        raise ValueError(f"X must be 2D, found {X.shape}")
    if X.shape != (EXPECTED_N, EXPECTED_D):
        raise ValueError(
            f"Unexpected X shape {X.shape}; expected "
            f"{(EXPECTED_N, EXPECTED_D)}"
        )

    if mask.ndim != 2:
        raise ValueError(f"valid_mask must be 2D, found {mask.shape}")
    if tuple(mask.shape) != EXPECTED_MASK_SHAPE:
        raise ValueError(
            f"Unexpected mask shape {mask.shape}; expected "
            f"{EXPECTED_MASK_SHAPE}"
        )

    mask_bool = np.asarray(mask, dtype=bool)
    n_valid = int(mask_bool.sum())

    if n_valid != X.shape[0]:
        raise ValueError(
            f"Mask contains {n_valid:,} valid pixels but X contains "
            f"{X.shape[0]:,} rows."
        )

    # Verify the row<->raster mapping without allocating a 23-channel raster.
    # A synthetic row-id vector is written through the mask and read back.
    row_ids = np.arange(X.shape[0], dtype=np.int64)
    row_raster = np.full(mask_bool.shape, -1, dtype=np.int64)
    row_raster[mask_bool] = row_ids
    recovered = row_raster[mask_bool]
    if not np.array_equal(recovered, row_ids):
        raise RuntimeError("Boolean-mask row/raster round-trip failed.")

    return X, mask_bool, x_path, mask_path


def streaming_feature_moments_and_covariance(X, chunk_rows: int):
    """
    Exact full-clean mean/covariance accumulation in float64, without loading
    the complete 4.29M x 23 matrix into RAM.

    Uses parallel-merge covariance updates for numerical stability.
    """
    n_total = 0
    mean = np.zeros(X.shape[1], dtype=np.float64)
    M2 = np.zeros((X.shape[1], X.shape[1]), dtype=np.float64)

    finite_ok = True
    feature_min = np.full(X.shape[1], np.inf, dtype=np.float64)
    feature_max = np.full(X.shape[1], -np.inf, dtype=np.float64)

    for start in range(0, X.shape[0], chunk_rows):
        end = min(start + chunk_rows, X.shape[0])
        block = np.asarray(X[start:end], dtype=np.float64)

        if not np.isfinite(block).all():
            finite_ok = False
            bad = np.argwhere(~np.isfinite(block))[0]
            raise ValueError(
                "Non-finite value found in X at approximate global position "
                f"row={start + int(bad[0])}, col={int(bad[1])}"
            )

        feature_min = np.minimum(feature_min, block.min(axis=0))
        feature_max = np.maximum(feature_max, block.max(axis=0))

        nb = block.shape[0]
        mb = block.mean(axis=0)
        centered = block - mb
        M2b = centered.T @ centered

        if n_total == 0:
            mean = mb
            M2 = M2b
            n_total = nb
        else:
            delta = mb - mean
            new_n = n_total + nb
            mean = mean + delta * (nb / new_n)
            M2 = M2 + M2b + np.outer(delta, delta) * (
                n_total * nb / new_n
            )
            n_total = new_n

        print(f"PCA moments: processed {end:,}/{X.shape[0]:,} rows")

    cov = M2 / (n_total - 1)

    return {
        "n": n_total,
        "mean": mean,
        "cov": cov,
        "feature_min": feature_min,
        "feature_max": feature_max,
        "finite_ok": finite_ok,
    }


def pca_from_covariance(cov: np.ndarray, target: float):
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = np.maximum(eigvals[order], 0.0)
    eigvecs = eigvecs[:, order]

    total = float(eigvals.sum())
    if total <= 0:
        raise ValueError("Covariance has zero total variance.")

    ratio = eigvals / total
    cumulative = np.cumsum(ratio)
    k = int(np.searchsorted(cumulative, target, side="left") + 1)

    return eigvals, eigvecs, ratio, cumulative, k


def integral_window_counts(mask: np.ndarray, window: int, stride: int):
    H, W = mask.shape
    if window > H or window > W:
        raise ValueError("Window larger than raster.")

    rows = np.arange(0, H - window + 1, stride, dtype=np.int32)
    cols = np.arange(0, W - window + 1, stride, dtype=np.int32)

    # Integral image with a zero border.
    ii = np.pad(
        mask.astype(np.int32),
        ((1, 0), (1, 0)),
        mode="constant",
        constant_values=0,
    ).cumsum(axis=0).cumsum(axis=1)

    # Vectorized rectangle sums for all row/column start pairs.
    r0 = rows[:, None]
    c0 = cols[None, :]
    r1 = r0 + window
    c1 = c0 + window

    counts = (
        ii[r1, c1]
        - ii[r0, c1]
        - ii[r1, c0]
        + ii[r0, c0]
    )

    fractions = counts.astype(np.float64) / float(window * window)

    thresholds = [0.25, 0.50, 0.75, 0.90, 1.00]
    diag = {
        "stride": stride,
        "n_row_positions": int(len(rows)),
        "n_col_positions": int(len(cols)),
        "n_windows": int(counts.size),
        "valid_fraction_mean": float(fractions.mean()),
        "valid_fraction_median": float(np.median(fractions)),
        "valid_fraction_min": float(fractions.min()),
        "valid_fraction_max": float(fractions.max()),
    }

    for q in [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]:
        diag[f"valid_fraction_q{int(q*100):02d}"] = float(
            np.quantile(fractions, q)
        )

    for t in thresholds:
        diag[f"windows_valid_ge_{int(t*100):03d}pct"] = int(
            np.sum(fractions >= t)
        )

    return diag


def hardware_summary():
    d = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count_logical": os.cpu_count(),
    }

    try:
        import psutil
        vm = psutil.virtual_memory()
        d["ram_total_gb"] = vm.total / (1024**3)
        d["ram_available_gb"] = vm.available / (1024**3)
    except Exception as e:
        d["psutil_status"] = f"unavailable: {e}"

    try:
        import torch
        d["torch_version"] = torch.__version__
        d["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            d["cuda_device"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            d["cuda_vram_gb"] = props.total_memory / (1024**3)
        d["mps_available"] = bool(
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        )
    except Exception as e:
        d["torch_status"] = f"unavailable: {e}"

    return d


def main():
    args = parse_args()
    root = args.root.resolve()

    out_dir = root / OUTPUT_NAME / "preflight"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("SSF-AE PRE-FLIGHT")
    print("=" * 72)
    print("Project root:", root)

    X, mask, x_path, mask_path = load_and_validate(root)

    print("\nInput validation passed.")
    print("X:", X.shape, X.dtype)
    print("Mask:", mask.shape, mask.dtype)
    print("Valid pixels:", int(mask.sum()))
    print("Row/raster boolean-mask round-trip: PASS")

    print("\nComputing full-clean PCA covariance in streaming mode...")
    moments = streaming_feature_moments_and_covariance(
        X, args.chunk_rows
    )

    eigvals, eigvecs, ratio, cumulative, k = pca_from_covariance(
        moments["cov"], PCA_TARGET
    )

    pca_df = pd.DataFrame({
        "component": np.arange(1, EXPECTED_D + 1),
        "eigenvalue": eigvals,
        "explained_variance_ratio": ratio,
        "cumulative_explained_variance": cumulative,
    })
    pca_csv = out_dir / "SSFAE_PCA_Explained_Variance.csv"
    pca_df.to_csv(pca_csv, index=False)

    pad_to_4 = (-k) % 4
    effective_depth = k + pad_to_4

    print(f"\nPCA target: {PCA_TARGET:.4%}")
    print(f"Retained PCs: {k}")
    print(f"Cumulative variance at k: {cumulative[k-1]:.8f}")
    print(
        "Padding needed for depth factor 4:",
        pad_to_4,
        f"(effective depth={effective_depth})",
    )

    print("\nComputing window diagnostics...")
    window_rows = []
    for stride in STRIDES_TO_CHECK:
        diag = integral_window_counts(mask, WINDOW, stride)
        window_rows.append(diag)
        print(
            f"stride={stride}: windows={diag['n_windows']:,}, "
            f"mean valid fraction={diag['valid_fraction_mean']:.4f}, "
            f">=75% valid={diag['windows_valid_ge_075pct']:,}"
        )

    window_df = pd.DataFrame(window_rows)
    window_csv = out_dir / "SSFAE_Window_Diagnostics.csv"
    window_df.to_csv(window_csv, index=False)

    hw = hardware_summary()

    summary = {
        "protocol_status": "preflight_only_no_training",
        "root": root,
        "input": {
            "X_path": x_path,
            "mask_path": mask_path,
            "X_shape": list(X.shape),
            "X_dtype": str(X.dtype),
            "mask_shape": list(mask.shape),
            "valid_pixels": int(mask.sum()),
            "row_raster_roundtrip": "PASS",
        },
        "pca": {
            "fit_source": "full clean X.npy only",
            "target_cumulative_variance": PCA_TARGET,
            "retained_components": k,
            "cumulative_variance_at_k": float(cumulative[k - 1]),
            "depth_padding_to_multiple_of_4": pad_to_4,
            "effective_depth_after_padding": effective_depth,
            "note": (
                "Do not freeze padding/architecture until this preflight "
                "output is reviewed."
            ),
        },
        "paper_parameter_ambiguity": {
            "window_size": WINDOW,
            "methodology_text_stride": 10,
            "parameter_table_stride": 20,
            "action": (
                "Both strides are diagnosed. Freeze one before training; "
                "do not tune using final injected-test labels."
            ),
        },
        "window_diagnostics": window_rows,
        "feature_min": moments["feature_min"],
        "feature_max": moments["feature_max"],
        "hardware": hw,
        "next_step": (
            "Review PCA retained-component count, valid-window support, and "
            "hardware. Then freeze SSF-AE architecture/training settings and "
            "run a small runtime benchmark before full training."
        ),
    }

    summary_json = out_dir / "SSFAE_Preflight_Summary.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=jsonable)

    print("\nSaved:")
    print(" ", summary_json)
    print(" ", pca_csv)
    print(" ", window_csv)
    print("\nSTOP HERE. No SSF-AE model has been trained.")


if __name__ == "__main__":
    main()
