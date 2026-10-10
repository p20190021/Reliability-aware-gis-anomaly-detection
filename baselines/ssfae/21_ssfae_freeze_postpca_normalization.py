"""
SSF-AE post-PCA normalization freeze (LIGHTWEIGHT; NO PYTORCH REQUIRED)

Place this file in:
    ...\Outlier Paper\Code\SSFAE\

Reads:
    ...\Outlier Paper\Code\X.npy
    ...\Outlier Paper\Code\SSFAE\results\protocol_freeze\
        SSFAE_Preprocessing_Params.npz

Writes ONLY inside:
    ...\Outlier Paper\Code\SSFAE\results\protocol_freeze\

Purpose
-------
The SSF-AE paper states that PCA-reduced dimensions are normalized before
reconstruction and that both branch decoders use Sigmoid outputs in [0,1],
but the paper does not provide an explicit normalization formula.

For this heterogeneous-GIS reimplementation we therefore freeze the following
transparent choice BEFORE any anomaly-detection evaluation:

    clean z-score -> frozen 21-PC projection -> clean per-PC min-max to [0,1]

The min/max parameters are fitted once on the original clean dataset and must
be reused unchanged for every injected dataset.

IMPORTANT:
- No injected data are read.
- No anomaly labels are read.
- No model is trained.
- PyTorch is not required.
- At later inference, transformed injected values are NOT clipped to [0,1].
  Values outside the clean range are allowed to remain outside [0,1], so the
  frozen clean transform does not hide injected extremes.
- The three model-alignment padding channels are added only AFTER the 21 real
  normalized PCs and never contribute to reconstruction loss/anomaly scoring.

Run
---
From the SSFAE folder:

    python 21_ssfae_freeze_postpca_normalization.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_N = 4_290_958
EXPECTED_D = 23
EXPECTED_K = 21
CHUNK_ROWS = 100_000

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_PROTOCOL_DIR = SCRIPT_DIR / "results" / "protocol_freeze"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", type=Path, default=None)
    p.add_argument("--protocol-dir", type=Path, default=None)
    p.add_argument("--chunk-rows", type=int, default=CHUNK_ROWS)
    return p.parse_args()


def main():
    args = parse_args()

    project_root = (
        args.project_root.resolve()
        if args.project_root is not None
        else DEFAULT_PROJECT_ROOT.resolve()
    )

    protocol_dir = (
        args.protocol_dir.resolve()
        if args.protocol_dir is not None
        else DEFAULT_PROTOCOL_DIR.resolve()
    )
    protocol_dir.mkdir(parents=True, exist_ok=True)

    x_path = project_root / "X.npy"
    prep_path = protocol_dir / "SSFAE_Preprocessing_Params.npz"

    if not x_path.exists():
        raise FileNotFoundError(f"Missing input: {x_path}")

    if not prep_path.exists():
        raise FileNotFoundError(
            f"Missing frozen preprocessing file: {prep_path}\n"
            "Run 20_ssfae_freeze_preprocessing_and_windows.py first."
        )

    X = np.load(x_path, mmap_mode="r")
    if X.shape != (EXPECTED_N, EXPECTED_D):
        raise ValueError(
            f"Unexpected X shape {X.shape}; "
            f"expected {(EXPECTED_N, EXPECTED_D)}"
        )

    prep = np.load(prep_path)

    feature_mean = np.asarray(
        prep["feature_mean"], dtype=np.float64
    )
    feature_std = np.asarray(
        prep["feature_std"], dtype=np.float64
    )
    pca_components = np.asarray(
        prep["pca_components"], dtype=np.float64
    )
    retained_components = int(
        np.asarray(prep["retained_components"]).item()
    )

    if retained_components != EXPECTED_K:
        raise ValueError(
            f"Frozen preprocessing retains {retained_components} PCs; "
            f"expected {EXPECTED_K}."
        )

    if pca_components.shape != (EXPECTED_K, EXPECTED_D):
        raise ValueError(
            f"Unexpected PCA component shape {pca_components.shape}; "
            f"expected {(EXPECTED_K, EXPECTED_D)}"
        )

    if np.any(feature_std <= 0):
        raise ValueError("Frozen feature_std contains non-positive values.")

    print("=" * 76)
    print("SSF-AE POST-PCA NORMALIZATION FREEZE")
    print("=" * 76)
    print("Script folder :", SCRIPT_DIR)
    print("Project root  :", project_root)
    print("Protocol dir  :", protocol_dir)
    print("X             :", X.shape, X.dtype)
    print("Retained PCs  :", retained_components)
    print("\nNo PyTorch is required. No model will be trained.\n")

    pc_min = np.full(EXPECTED_K, np.inf, dtype=np.float64)
    pc_max = np.full(EXPECTED_K, -np.inf, dtype=np.float64)

    # For diagnostics only: exact first and second moments of projected PCs.
    n_total = 0
    pc_sum = np.zeros(EXPECTED_K, dtype=np.float64)
    pc_sum_sq = np.zeros(EXPECTED_K, dtype=np.float64)

    for start in range(0, X.shape[0], args.chunk_rows):
        end = min(start + args.chunk_rows, X.shape[0])

        block = np.asarray(X[start:end], dtype=np.float64)

        if not np.isfinite(block).all():
            bad = np.argwhere(~np.isfinite(block))[0]
            raise ValueError(
                f"Non-finite X value at row {start + int(bad[0])}, "
                f"column {int(bad[1])}"
            )

        z = (block - feature_mean) / feature_std

        # pca_components has shape (21, 23)
        pcs = z @ pca_components.T

        pc_min = np.minimum(pc_min, pcs.min(axis=0))
        pc_max = np.maximum(pc_max, pcs.max(axis=0))

        pc_sum += pcs.sum(axis=0)
        pc_sum_sq += np.square(pcs).sum(axis=0)
        n_total += pcs.shape[0]

        print(
            f"Projected {end:,}/{X.shape[0]:,} rows"
        )

    pc_range = pc_max - pc_min

    if np.any(~np.isfinite(pc_min)) or np.any(~np.isfinite(pc_max)):
        raise ValueError("Non-finite per-PC extrema encountered.")

    if np.any(pc_range <= 0):
        bad = np.flatnonzero(pc_range <= 0)
        raise ValueError(
            f"Non-positive PC range(s): {bad.tolist()}"
        )

    pc_mean = pc_sum / n_total
    pc_var = (
        pc_sum_sq / n_total
        - np.square(pc_mean)
    )
    pc_std = np.sqrt(np.maximum(pc_var, 0.0))

    # PCA scores are theoretically centered around zero after z-scoring.
    # This is the normalized value corresponding to PC score 0 and will
    # later serve as a fixed neutral fill for invalid raster cells.
    neutral_fill = (0.0 - pc_min) / pc_range

    # Small numerical deviations are fine, but neutral fill should remain
    # within the clean min/max range by construction.
    if np.any(neutral_fill < -1e-12) or np.any(neutral_fill > 1 + 1e-12):
        raise RuntimeError(
            "Neutral-fill value fell outside [0,1]; review preprocessing."
        )

    # Save frozen parameters.
    out_npz = protocol_dir / "SSFAE_PostPCA_Normalization_Params.npz"
    np.savez(
        out_npz,
        pc_min=pc_min,
        pc_max=pc_max,
        pc_range=pc_range,
        neutral_fill=neutral_fill,
        clipping_enabled=np.bool_(False),
        retained_components=np.int64(EXPECTED_K),
    )

    diag_df = pd.DataFrame({
        "pc": np.arange(1, EXPECTED_K + 1),
        "clean_pc_min": pc_min,
        "clean_pc_max": pc_max,
        "clean_pc_range": pc_range,
        "clean_pc_mean": pc_mean,
        "clean_pc_std": pc_std,
        "neutral_fill_after_minmax": neutral_fill,
    })

    diag_csv = protocol_dir / "SSFAE_PostPCA_Normalization_Diagnostics.csv"
    diag_df.to_csv(diag_csv, index=False)

    summary = {
        "status": "postpca_normalization_frozen_no_training",
        "script_folder": str(SCRIPT_DIR),
        "project_root_read_only": str(project_root),
        "protocol_dir": str(protocol_dir),
        "input_X": str(x_path),
        "frozen_preprocessing": str(prep_path),
        "retained_pcs": EXPECTED_K,
        "normalization": {
            "type": "clean_per_pc_minmax",
            "formula": "(pc - clean_pc_min) / (clean_pc_max - clean_pc_min)",
            "fit_data": "original clean X.npy only",
            "target_clean_range": "[0,1]",
            "inference_clipping": False,
            "reason_for_no_clipping": (
                "Injected values outside the frozen clean range remain "
                "outside [0,1] rather than being hidden by saturation. "
                "The same clean affine transform is reused unchanged."
            ),
            "sigmoid_compatibility": (
                "Clean training inputs are in [0,1], matching the SSF-AE "
                "decoder Sigmoid output range."
            ),
        },
        "invalid_raster_fill_for_later_inference": {
            "policy": (
                "Use the per-PC normalized value corresponding to PC=0 "
                "(the clean centered PCA mean) for invalid raster cells."
            ),
            "neutral_fill": neutral_fill.tolist(),
            "note": (
                "Invalid cells remain excluded from reconstruction-error "
                "accumulation. The fill only supplies a neutral tensor value "
                "for partially valid inference windows."
            ),
        },
        "padding_policy": {
            "real_normalized_pcs": 21,
            "model_aligned_depth": 24,
            "padding_channels": 3,
            "padding_value": 0.0,
            "loss_policy": (
                "Crop model output back to the first 21 real PCs before "
                "calculating reconstruction loss/anomaly scores."
            ),
        },
        "augmentation_policy": {
            "decision": "omit",
            "reason": (
                "The paper's spectral-dimension shuffling and spectral-noise "
                "augmentation are HSI-specific and insufficiently specified "
                "numerically for a faithful heterogeneous-GIS transfer. "
                "No replacement augmentation is invented."
            ),
        },
    }

    out_json = protocol_dir / "SSFAE_PostPCA_Normalization_Protocol.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nPost-PCA normalization frozen.")
    print("Clean per-PC min/max -> [0,1]")
    print("Inference clipping   : OFF")
    print(
        "Neutral-fill range   : "
        f"{neutral_fill.min():.6f} to {neutral_fill.max():.6f}"
    )

    print("\nSaved only inside:")
    print(" ", protocol_dir)
    print("\nFiles:")
    print("  SSFAE_PostPCA_Normalization_Params.npz")
    print("  SSFAE_PostPCA_Normalization_Diagnostics.csv")
    print("  SSFAE_PostPCA_Normalization_Protocol.json")
    print("\nSTOP HERE. No SSF-AE model has been trained.")


if __name__ == "__main__":
    main()
