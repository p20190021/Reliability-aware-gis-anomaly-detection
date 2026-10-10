"""
SSF-AE standardized-PCA diagnostic (LIGHTWEIGHT; NO PYTORCH REQUIRED)

Place this file in:
    ...\Outlier Paper\Code\SSFAE\

It reads the shared master input:
    ...\Outlier Paper\Code\X.npy

It writes ONLY inside:
    ...\Outlier Paper\Code\SSFAE\results\pca_diagnostic\

Purpose
-------
The original SSF-AE paper applies PCA to hyperspectral bands. Our 23 GIS
features have heterogeneous physical scales, so this diagnostic compares:

A) covariance-PCA on the raw engineered features
B) correlation-PCA, i.e. PCA after clean feature-wise standardization

No anomaly labels, injected data, or test performance are used.
No model is trained.

Run
---
From the SSFAE folder:
    python 19_ssfae_standardized_pca_diagnostic.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

EXPECTED_N = 4_290_958
EXPECTED_D = 23
TARGETS = (0.95, 0.99, 0.999)
CHUNK_ROWS = 100_000

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "results" / "pca_diagnostic"


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
                + np.outer(delta, delta) * (n_total * nb / new_n)
            )
            n_total = new_n

        print(f"Processed {end:,}/{X.shape[0]:,} rows")

    cov = M2 / (n_total - 1)
    std = np.sqrt(np.diag(cov))
    return mean, cov, std


def eig_summary(matrix: np.ndarray, label: str):
    vals, vecs = np.linalg.eigh(matrix)
    order = np.argsort(vals)[::-1]
    vals = np.maximum(vals[order], 0.0)
    vecs = vecs[:, order]

    total = float(vals.sum())
    if total <= 0:
        raise ValueError(f"{label}: non-positive total variance")

    ratio = vals / total
    cum = np.cumsum(ratio)

    rows = []
    for i in range(len(vals)):
        rows.append({
            "representation": label,
            "component": i + 1,
            "eigenvalue": float(vals[i]),
            "explained_variance_ratio": float(ratio[i]),
            "cumulative_explained_variance": float(cum[i]),
        })

    retained = {}
    for t in TARGETS:
        k = int(np.searchsorted(cum, t, side="left") + 1)
        retained[f"{t:.3f}"] = {
            "k": k,
            "cumulative_at_k": float(cum[k - 1]),
            "padding_to_multiple_of_4": int((-k) % 4),
            "effective_depth_if_padded": int(k + ((-k) % 4)),
        }

    return pd.DataFrame(rows), retained, vals, vecs, ratio, cum


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
    if not x_path.exists():
        raise FileNotFoundError(
            f"Missing {x_path}\n"
            "Expected X.npy in the parent Code folder."
        )

    X = np.load(x_path, mmap_mode="r")
    if X.shape != (EXPECTED_N, EXPECTED_D):
        raise ValueError(
            f"Unexpected X shape {X.shape}; "
            f"expected {(EXPECTED_N, EXPECTED_D)}"
        )

    print("=" * 72)
    print("SSF-AE STANDARDIZED-PCA DIAGNOSTIC")
    print("=" * 72)
    print("Script folder :", SCRIPT_DIR)
    print("Project root  :", project_root)
    print("Input         :", x_path)
    print("Output folder :", output_dir)
    print("X shape       :", X.shape)
    print("\nNo PyTorch is required. No model will be trained.\n")

    mean, cov, std = streaming_mean_cov(X, args.chunk_rows)

    if np.any(std <= 0):
        bad = np.flatnonzero(std <= 0)
        raise ValueError(
            f"Zero-variance feature(s) found at indices: {bad.tolist()}"
        )

    # Raw covariance PCA.
    raw_df, raw_retained, *_ = eig_summary(
        cov, "raw_covariance_pca"
    )

    # PCA after feature-wise standardization.
    # Cov(StandardScaler(X)) is the correlation matrix.
    denom = np.outer(std, std)
    corr = cov / denom
    corr = (corr + corr.T) / 2.0
    np.fill_diagonal(corr, 1.0)

    std_df, std_retained, *_ = eig_summary(
        corr, "standardized_correlation_pca"
    )

    comparison = pd.concat([raw_df, std_df], ignore_index=True)
    comparison.to_csv(
        output_dir / "SSFAE_Raw_vs_Standardized_PCA.csv",
        index=False,
    )

    feature_stats = pd.DataFrame({
        "feature_index": np.arange(EXPECTED_D),
        "clean_mean": mean,
        "clean_std": std,
        "clean_variance": std ** 2,
    })
    feature_stats.to_csv(
        output_dir / "SSFAE_Clean_Feature_Statistics.csv",
        index=False,
    )

    threshold_rows = []
    for representation, retained in [
        ("raw_covariance_pca", raw_retained),
        ("standardized_correlation_pca", std_retained),
    ]:
        for target_str, info in retained.items():
            threshold_rows.append({
                "representation": representation,
                "target_variance": float(target_str),
                **info,
            })

    threshold_df = pd.DataFrame(threshold_rows)
    threshold_df.to_csv(
        output_dir / "SSFAE_PCA_Threshold_Comparison.csv",
        index=False,
    )

    summary = {
        "status": "diagnostic_only_no_training",
        "script_folder": str(SCRIPT_DIR),
        "project_root": str(project_root),
        "input_X": str(x_path),
        "output_dir": str(output_dir),
        "X_shape": list(X.shape),
        "targets": list(TARGETS),
        "raw_covariance_pca": raw_retained,
        "standardized_correlation_pca": std_retained,
        "decision_rule": (
            "Do not choose preprocessing by final anomaly-detection "
            "performance. Use this diagnostic only to judge whether raw "
            "covariance PCA is dominated by heterogeneous GIS feature scales."
        ),
    }

    with open(
        output_dir / "SSFAE_PCA_Diagnostic_Summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(summary, f, indent=2)

    print("\nRetained component counts")
    print("-------------------------")
    for target in TARGETS:
        key = f"{target:.3f}"
        print(
            f"Target {target:.1%}: "
            f"raw k={raw_retained[key]['k']}, "
            f"standardized k={std_retained[key]['k']}"
        )

    print("\nSaved only inside:")
    print(" ", output_dir)
    print("\nFiles:")
    print("  SSFAE_Raw_vs_Standardized_PCA.csv")
    print("  SSFAE_Clean_Feature_Statistics.csv")
    print("  SSFAE_PCA_Threshold_Comparison.csv")
    print("  SSFAE_PCA_Diagnostic_Summary.json")
    print("\nSTOP HERE. No SSF-AE model has been trained.")


if __name__ == "__main__":
    main()
