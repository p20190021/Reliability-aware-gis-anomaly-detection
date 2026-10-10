"""
SSF-AE protocol-freeze preparation (LIGHTWEIGHT; NO PYTORCH REQUIRED)

Place this file in:
    ...\Outlier Paper\Code\SSFAE\

Reads shared master inputs only:
    ...\Outlier Paper\Code\X.npy
    ...\Outlier Paper\Code\valid_mask_1.npy

Writes only inside:
    ...\Outlier Paper\Code\SSFAE\results\protocol_freeze\

What this script freezes
------------------------
1. Clean feature-wise standardization for heterogeneous GIS variables.
2. PCA on the standardized clean feature matrix.
3. The paper's 99.90% cumulative-variance rule.
4. Expected retained depth = 21 PCs for the current dataset.
5. Internal depth alignment by zero-padding 21 -> 24 only inside the model.
6. Spatial window size = 40.
7. Sliding stride = 20 (Table 1 setting).
8. Training/validation windows = fully valid windows only.
9. Deterministic 80/20 clean-window split, seed 42.
10. Inference-window coordinates that cover every valid raster pixel.

No model is trained.
No injected data are read.
No final-test labels are used.
PyTorch is not required.

Run
---
From the SSFAE folder:
    python 20_ssfae_freeze_preprocessing_and_windows.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_N = 4_290_958
EXPECTED_D = 23
EXPECTED_MASK_SHAPE = (2897, 2696)

PCA_TARGET = 0.999
EXPECTED_K = 21

WINDOW = 40
STRIDE = 20
MODEL_DEPTH_MULTIPLE = 4

TRAIN_FRACTION = 0.80
SPLIT_SEED = 42
CHUNK_ROWS = 100_000

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "results" / "protocol_freeze"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--chunk-rows", type=int, default=CHUNK_ROWS)
    return p.parse_args()


def streaming_mean_cov(X, chunk_rows: int):
    n_total = 0
    mean = np.zeros(X.shape[1], dtype=np.float64)
    M2 = np.zeros((X.shape[1], X.shape[1]), dtype=np.float64)

    for start in range(0, X.shape[0], chunk_rows):
        end = min(start + chunk_rows, X.shape[0])
        block = np.asarray(X[start:end], dtype=np.float64)

        if not np.isfinite(block).all():
            bad = np.argwhere(~np.isfinite(block))[0]
            raise ValueError(
                f"Non-finite X value at row {start + int(bad[0])}, "
                f"column {int(bad[1])}"
            )

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
            M2 = (
                M2
                + M2b
                + np.outer(delta, delta)
                * (n_total * nb / new_n)
            )
            n_total = new_n

        print(f"Moments: processed {end:,}/{X.shape[0]:,} rows")

    cov = M2 / (n_total - 1)
    std = np.sqrt(np.diag(cov))

    if np.any(std <= 0):
        bad = np.flatnonzero(std <= 0)
        raise ValueError(
            f"Zero-variance feature(s): {bad.tolist()}"
        )

    return mean, cov, std


def standardized_pca(cov: np.ndarray, std: np.ndarray):
    corr = cov / np.outer(std, std)
    corr = (corr + corr.T) / 2.0
    np.fill_diagonal(corr, 1.0)

    eigvals, eigvecs = np.linalg.eigh(corr)
    order = np.argsort(eigvals)[::-1]

    eigvals = np.maximum(eigvals[order], 0.0)
    eigvecs = eigvecs[:, order]

    ratio = eigvals / eigvals.sum()
    cumulative = np.cumsum(ratio)

    k = int(
        np.searchsorted(
            cumulative,
            PCA_TARGET,
            side="left",
        )
        + 1
    )

    return corr, eigvals, eigvecs, ratio, cumulative, k


def edge_cover_positions(length: int, window: int, stride: int):
    """
    Standard stride grid plus an explicit final start at length-window
    when the stride does not land exactly on the final edge.
    """
    if window > length:
        raise ValueError("Window exceeds dimension length.")

    pos = list(range(0, length - window + 1, stride))
    last = length - window

    if pos[-1] != last:
        pos.append(last)

    return np.asarray(pos, dtype=np.int32)


def integral_image(mask: np.ndarray):
    return np.pad(
        mask.astype(np.int32),
        ((1, 0), (1, 0)),
        mode="constant",
        constant_values=0,
    ).cumsum(axis=0).cumsum(axis=1)


def window_valid_counts(mask: np.ndarray, rows, cols, window: int):
    ii = integral_image(mask)

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
    return counts


def starts_from_grid(rows, cols, keep_matrix):
    rr, cc = np.nonzero(keep_matrix)

    starts = np.column_stack(
        [rows[rr], cols[cc]]
    ).astype(np.int32)

    return starts


def rectangle_coverage(shape, starts, window: int):
    """
    Coverage count via 2D difference array.
    """
    H, W = shape
    diff = np.zeros((H + 1, W + 1), dtype=np.int32)

    for r, c in starts:
        r = int(r)
        c = int(c)
        r2 = r + window
        c2 = c + window

        diff[r, c] += 1
        diff[r2, c] -= 1
        diff[r, c2] -= 1
        diff[r2, c2] += 1

    coverage = diff.cumsum(axis=0).cumsum(axis=1)[:-1, :-1]
    return coverage


def main():
    args = parse_args()

    project_root = (
        args.project_root.resolve()
        if args.project_root is not None
        else DEFAULT_PROJECT_ROOT.resolve()
    )

    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else DEFAULT_OUTPUT_DIR.resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    x_path = project_root / "X.npy"
    mask_path = project_root / "valid_mask_1.npy"

    if not x_path.exists():
        raise FileNotFoundError(x_path)
    if not mask_path.exists():
        raise FileNotFoundError(mask_path)

    X = np.load(x_path, mmap_mode="r")
    mask = np.load(mask_path, mmap_mode="r")
    mask = np.asarray(mask, dtype=bool)

    if X.shape != (EXPECTED_N, EXPECTED_D):
        raise ValueError(
            f"Unexpected X shape {X.shape}; expected "
            f"{(EXPECTED_N, EXPECTED_D)}"
        )
    if mask.shape != EXPECTED_MASK_SHAPE:
        raise ValueError(
            f"Unexpected mask shape {mask.shape}; expected "
            f"{EXPECTED_MASK_SHAPE}"
        )
    if int(mask.sum()) != X.shape[0]:
        raise ValueError(
            "valid_mask pixel count does not match X row count."
        )

    print("=" * 76)
    print("SSF-AE PROTOCOL-FREEZE PREPARATION")
    print("=" * 76)
    print("Script folder :", SCRIPT_DIR)
    print("Project root  :", project_root)
    print("Output folder :", output_dir)
    print("X             :", X.shape, X.dtype)
    print("Mask          :", mask.shape, "valid =", int(mask.sum()))
    print("\nNo PyTorch is required. No model will be trained.\n")

    # ------------------------------------------------------------------
    # 1. Freeze clean standardization + standardized PCA
    # ------------------------------------------------------------------
    mean, cov, std = streaming_mean_cov(X, args.chunk_rows)

    (
        corr,
        eigvals,
        eigvecs,
        ratio,
        cumulative,
        k,
    ) = standardized_pca(cov, std)

    print("\nStandardized PCA")
    print("----------------")
    print(f"Target cumulative variance : {PCA_TARGET:.4%}")
    print(f"Retained components        : {k}")
    print(f"Cumulative variance at k   : {cumulative[k-1]:.10f}")

    if k != EXPECTED_K:
        raise RuntimeError(
            f"Expected k={EXPECTED_K} from the reviewed diagnostic, "
            f"but recomputation produced k={k}. Stop and review."
        )

    pad_depth = (-k) % MODEL_DEPTH_MULTIPLE
    model_depth = k + pad_depth

    print(f"Internal zero-padding      : {pad_depth}")
    print(f"Model-aligned depth        : {model_depth}")

    # pca_components uses sklearn-like orientation: (k, n_features)
    pca_components = eigvecs[:, :k].T

    np.savez(
        output_dir / "SSFAE_Preprocessing_Params.npz",
        feature_mean=mean.astype(np.float64),
        feature_std=std.astype(np.float64),
        correlation_matrix=corr.astype(np.float64),
        pca_eigenvalues=eigvals.astype(np.float64),
        pca_explained_variance_ratio=ratio.astype(np.float64),
        pca_cumulative_variance=cumulative.astype(np.float64),
        pca_components=pca_components.astype(np.float64),
        retained_components=np.int64(k),
        pca_target=np.float64(PCA_TARGET),
        internal_pad_depth=np.int64(pad_depth),
        model_aligned_depth=np.int64(model_depth),
    )

    pca_table = pd.DataFrame({
        "component": np.arange(1, EXPECTED_D + 1),
        "eigenvalue": eigvals,
        "explained_variance_ratio": ratio,
        "cumulative_explained_variance": cumulative,
    })
    pca_table.to_csv(
        output_dir / "SSFAE_Frozen_PCA_Table.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # 2. Freeze spatial windows
    # ------------------------------------------------------------------
    rows = edge_cover_positions(mask.shape[0], WINDOW, STRIDE)
    cols = edge_cover_positions(mask.shape[1], WINDOW, STRIDE)

    counts = window_valid_counts(mask, rows, cols, WINDOW)

    fully_valid = counts == (WINDOW * WINDOW)
    any_valid = counts > 0

    full_starts = starts_from_grid(rows, cols, fully_valid)
    inference_starts = starts_from_grid(rows, cols, any_valid)

    print("\nWindow protocol")
    print("---------------")
    print("Window size                :", WINDOW)
    print("Stride                     :", STRIDE)
    print("Row positions              :", len(rows))
    print("Column positions           :", len(cols))
    print("Fully valid train/val      :", len(full_starts))
    print("Inference windows (>0 valid):", len(inference_starts))

    # Deterministic 80/20 split of fully valid clean windows.
    rng = np.random.default_rng(SPLIT_SEED)
    perm = rng.permutation(len(full_starts))

    n_train = int(np.floor(TRAIN_FRACTION * len(full_starts)))
    train_starts = full_starts[perm[:n_train]]
    val_starts = full_starts[perm[n_train:]]

    np.save(
        output_dir / "SSFAE_Train_Window_Starts.npy",
        train_starts,
    )
    np.save(
        output_dir / "SSFAE_Validation_Window_Starts.npy",
        val_starts,
    )
    np.save(
        output_dir / "SSFAE_Inference_Window_Starts.npy",
        inference_starts,
    )

    # ------------------------------------------------------------------
    # 3. Verify inference coverage of every valid pixel
    # ------------------------------------------------------------------
    coverage = rectangle_coverage(
        mask.shape,
        inference_starts,
        WINDOW,
    )

    valid_coverage = coverage[mask]

    uncovered_valid = int(np.sum(valid_coverage == 0))
    min_cov = int(valid_coverage.min())
    max_cov = int(valid_coverage.max())
    mean_cov = float(valid_coverage.mean())

    print("\nInference coverage")
    print("------------------")
    print("Uncovered valid pixels :", uncovered_valid)
    print("Minimum coverage       :", min_cov)
    print("Mean coverage          :", f"{mean_cov:.4f}")
    print("Maximum coverage       :", max_cov)

    if uncovered_valid != 0:
        raise RuntimeError(
            f"{uncovered_valid:,} valid pixels are not covered by "
            "the frozen inference windows."
        )

    coverage_summary = {
        "valid_pixels": int(mask.sum()),
        "uncovered_valid_pixels": uncovered_valid,
        "min_window_coverage": min_cov,
        "mean_window_coverage": mean_cov,
        "max_window_coverage": max_cov,
        "coverage_percentiles": {
            "p01": float(np.quantile(valid_coverage, 0.01)),
            "p05": float(np.quantile(valid_coverage, 0.05)),
            "p25": float(np.quantile(valid_coverage, 0.25)),
            "p50": float(np.quantile(valid_coverage, 0.50)),
            "p75": float(np.quantile(valid_coverage, 0.75)),
            "p95": float(np.quantile(valid_coverage, 0.95)),
            "p99": float(np.quantile(valid_coverage, 0.99)),
        },
    }

    # ------------------------------------------------------------------
    # 4. Save reviewer-safe frozen protocol
    # ------------------------------------------------------------------
    protocol = {
        "status": "protocol_frozen_no_model_training",
        "paths": {
            "script_folder": str(SCRIPT_DIR),
            "project_root_read_only": str(project_root),
            "output_dir": str(output_dir),
        },
        "input": {
            "X_path": str(x_path),
            "mask_path": str(mask_path),
            "X_shape": list(X.shape),
            "mask_shape": list(mask.shape),
            "valid_pixels": int(mask.sum()),
        },
        "preprocessing": {
            "rationale": (
                "The 23 GIS features have heterogeneous physical scales. "
                "Feature-wise standardization is therefore frozen before "
                "PCA so covariance PCA is not dominated by measurement "
                "scale. Parameters are fitted on the original clean "
                "dataset only and must be reused unchanged for injected "
                "datasets."
            ),
            "feature_standardization": "clean mean/std z-score",
            "pca_space": "standardized clean features / correlation PCA",
            "pca_target": PCA_TARGET,
            "retained_components": int(k),
            "cumulative_variance_at_k": float(cumulative[k - 1]),
            "internal_zero_padding": int(pad_depth),
            "model_aligned_depth": int(model_depth),
            "padding_policy": (
                "Padding is internal tensor alignment only. Reconstructed "
                "outputs are cropped back to the retained PCA depth before "
                "loss/anomaly-score calculation so padded channels never "
                "contribute to the reconstruction error."
            ),
        },
        "windows": {
            "window_size": WINDOW,
            "stride": STRIDE,
            "stride_rationale": (
                "The SSF-AE paper is internally inconsistent (10 in "
                "method text, 20 in Table 1). The frozen implementation "
                "uses 20 because it is the explicit basic-parameter-table "
                "setting."
            ),
            "edge_coverage_policy": (
                "Append the final H-window and W-window starts when the "
                "stride grid does not exactly reach the raster edge."
            ),
            "fully_valid_windows": int(len(full_starts)),
            "training_windows": int(len(train_starts)),
            "validation_windows": int(len(val_starts)),
            "train_fraction": TRAIN_FRACTION,
            "split_seed": SPLIT_SEED,
            "inference_windows_with_any_valid_pixel": int(
                len(inference_starts)
            ),
            "training_mask_policy": (
                "Only 100%-valid clean windows are used for model fitting "
                "and clean validation."
            ),
            "inference_mask_policy": (
                "All windows containing at least one valid pixel may be "
                "used. Reconstruction errors are accumulated only at valid "
                "pixels."
            ),
        },
        "coverage": coverage_summary,
        "still_unfrozen_for_model_stage": {
            "post_pca_normalization": (
                "The paper states that PCA-reduced dimensions are "
                "normalized but does not provide an explicit min/max "
                "formula. Freeze this before training."
            ),
            "augmentation": (
                "The paper describes random spectral-dimension shuffling "
                "and spectral noise, but numerical details are incomplete "
                "and spectral shuffling has no direct GIS interpretation. "
                "Do not invent augmentation silently."
            ),
            "deep_architecture_underspecification": (
                "Some channel/layer widths are described symbolically "
                "rather than fully numerically in the paper. These must "
                "be documented in the independent reimplementation."
            ),
        },
    }

    with open(
        output_dir / "SSFAE_Frozen_Protocol.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(protocol, f, indent=2)

    print("\nSaved only inside:")
    print(" ", output_dir)
    print("\nImportant outputs:")
    print("  SSFAE_Preprocessing_Params.npz")
    print("  SSFAE_Frozen_PCA_Table.csv")
    print("  SSFAE_Train_Window_Starts.npy")
    print("  SSFAE_Validation_Window_Starts.npy")
    print("  SSFAE_Inference_Window_Starts.npy")
    print("  SSFAE_Frozen_Protocol.json")

    print("\nSTOP HERE. No SSF-AE model has been trained.")


if __name__ == "__main__":
    main()
