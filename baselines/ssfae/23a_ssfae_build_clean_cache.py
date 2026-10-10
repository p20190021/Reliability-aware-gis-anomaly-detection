"""
SSF-AE clean normalized-PCA cache builder (NumPy-only).

WHY THIS SEPARATE SCRIPT EXISTS
-------------------------------
On some Windows Python environments, NumPy/MKL and PyTorch can load separate
copies of Intel's OpenMP runtime (libiomp5md.dll). The SSF-AE cache-building
stage performs large NumPy matrix multiplications, so doing that in the same
process that has already imported PyTorch can trigger OMP Error #15.

This script intentionally does NOT import:
- torch
- pandas

It performs only the frozen clean preprocessing and writes the exact cache
expected by 23_ssfae_clean_training.py.

SCIENTIFIC PROTOCOL
-------------------
- Reads original clean X.npy only.
- Reads frozen preprocessing/PCA/min-max parameters only.
- Does NOT load y.npy.
- Does NOT load flood-risk labels.
- Does NOT load injected datasets or indices.
- Does NOT refit any parameter.
- No clipping is applied after the frozen clean min-max transform.
- Invalid raster locations receive the already frozen neutral-fill value.

EXPECTED LOCATION
-----------------
.../Code/SSFAE/23a_ssfae_build_clean_cache.py

EXPECTED INPUTS
---------------
.../Code/X.npy
.../Code/valid_mask_1.npy
.../Code/SSFAE/results/protocol_freeze/
    SSFAE_Preprocessing_Params.npz
    SSFAE_PostPCA_Normalization_Params.npz

OUTPUT
------
.../Code/SSFAE/results/clean_training/cache/
    SSFAE_Clean_Normalized_PCA_Raster.npy
    SSFAE_Clean_Normalized_PCA_Raster_COMPLETE.json

After this script finishes successfully, run:
    python 23_ssfae_clean_training.py

The training script will detect the existing cache and skip the NumPy
matrix-multiplication cache-building stage.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path

import numpy as np


EXPECTED_N = 4_290_958
EXPECTED_D = 23
EXPECTED_MASK_SHAPE = (2897, 2696)
REAL_DEPTH = 21
CHUNK_ROWS = 100_000

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PROJECT_ROOT = SCRIPT_DIR.parent
PROTOCOL_DIR = SCRIPT_DIR / "results" / "protocol_freeze"
OUTPUT_DIR = SCRIPT_DIR / "results" / "clean_training"
CACHE_DIR = OUTPUT_DIR / "cache"

CACHE_PATH = CACHE_DIR / "SSFAE_Clean_Normalized_PCA_Raster.npy"
MARKER_PATH = CACHE_DIR / "SSFAE_Clean_Normalized_PCA_Raster_COMPLETE.json"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", type=Path, default=None)
    p.add_argument("--chunk-rows", type=int, default=CHUNK_ROWS)
    return p.parse_args()


def sha256_file(path: Path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def load_frozen_params():
    prep_path = PROTOCOL_DIR / "SSFAE_Preprocessing_Params.npz"
    norm_path = PROTOCOL_DIR / "SSFAE_PostPCA_Normalization_Params.npz"

    for path in (prep_path, norm_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing frozen artifact: {path}")

    prep = np.load(prep_path)
    norm = np.load(norm_path)

    feature_mean = np.asarray(prep["feature_mean"], dtype=np.float64)
    feature_std = np.asarray(prep["feature_std"], dtype=np.float64)
    pca_components = np.asarray(prep["pca_components"], dtype=np.float64)
    retained = int(np.asarray(prep["retained_components"]).item())

    pc_min = np.asarray(norm["pc_min"], dtype=np.float64)
    pc_range = np.asarray(norm["pc_range"], dtype=np.float64)
    neutral_fill = np.asarray(norm["neutral_fill"], dtype=np.float64)

    if retained != REAL_DEPTH:
        raise ValueError(
            f"Frozen preprocessing retains {retained} PCs; "
            f"expected {REAL_DEPTH}."
        )

    if feature_mean.shape != (EXPECTED_D,):
        raise ValueError("Unexpected feature_mean shape.")
    if feature_std.shape != (EXPECTED_D,):
        raise ValueError("Unexpected feature_std shape.")
    if pca_components.shape != (REAL_DEPTH, EXPECTED_D):
        raise ValueError(
            f"Unexpected PCA component shape: {pca_components.shape}"
        )
    if pc_min.shape != (REAL_DEPTH,):
        raise ValueError("Unexpected pc_min shape.")
    if pc_range.shape != (REAL_DEPTH,):
        raise ValueError("Unexpected pc_range shape.")
    if neutral_fill.shape != (REAL_DEPTH,):
        raise ValueError("Unexpected neutral_fill shape.")

    if np.any(feature_std <= 0):
        raise ValueError("feature_std contains non-positive value(s).")
    if np.any(pc_range <= 0):
        raise ValueError("pc_range contains non-positive value(s).")

    return {
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "pca_components": pca_components,
        "pc_min": pc_min,
        "pc_range": pc_range,
        "neutral_fill": neutral_fill,
        "prep_path": prep_path,
        "norm_path": norm_path,
    }


def main():
    args = parse_args()

    project_root = (
        args.project_root.resolve()
        if args.project_root is not None
        else DEFAULT_PROJECT_ROOT.resolve()
    )

    x_path = project_root / "X.npy"
    mask_path = project_root / "valid_mask_1.npy"

    if not x_path.exists():
        raise FileNotFoundError(x_path)
    if not mask_path.exists():
        raise FileNotFoundError(mask_path)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # A failed earlier run may have left a partial .npy file. Remove both the
    # partial cache and any stale completion marker before rebuilding.
    if CACHE_PATH.exists():
        print("Removing existing/partial cache:")
        print(" ", CACHE_PATH)
        CACHE_PATH.unlink()

    if MARKER_PATH.exists():
        MARKER_PATH.unlink()

    X = np.load(x_path, mmap_mode="r")
    valid_mask = np.load(mask_path, mmap_mode="r").astype(bool, copy=False)

    if X.shape != (EXPECTED_N, EXPECTED_D):
        raise ValueError(
            f"Unexpected X shape {X.shape}; "
            f"expected {(EXPECTED_N, EXPECTED_D)}"
        )

    if valid_mask.shape != EXPECTED_MASK_SHAPE:
        raise ValueError(
            f"Unexpected mask shape {valid_mask.shape}; "
            f"expected {EXPECTED_MASK_SHAPE}"
        )

    if int(valid_mask.sum()) != EXPECTED_N:
        raise ValueError(
            "valid_mask pixel count does not match X row count."
        )

    params = load_frozen_params()

    print("=" * 76)
    print("SSF-AE NUMPY-ONLY CLEAN CACHE BUILD")
    print("=" * 76)
    print("Project root :", project_root)
    print("X            :", X.shape, X.dtype)
    print("Valid mask   :", valid_mask.shape, int(valid_mask.sum()))
    print("Output cache :", CACHE_PATH)
    print("Torch loaded : NO")
    print("Injected data: NOT LOADED")
    print("Labels       : NOT LOADED")

    H, W = valid_mask.shape

    out = np.lib.format.open_memmap(
        CACHE_PATH,
        mode="w+",
        dtype=np.float32,
        shape=(H, W, REAL_DEPTH),
    )

    # Frozen neutral fill for invalid raster positions.
    out[:] = params["neutral_fill"].astype(np.float32)

    out_flat = out.reshape(H * W, REAL_DEPTH)
    valid_flat = np.flatnonzero(valid_mask.reshape(-1))

    pc_global_min = np.full(REAL_DEPTH, np.inf, dtype=np.float64)
    pc_global_max = np.full(REAL_DEPTH, -np.inf, dtype=np.float64)

    t0 = time.perf_counter()

    for start in range(0, X.shape[0], args.chunk_rows):
        end = min(start + args.chunk_rows, X.shape[0])

        block = np.asarray(X[start:end], dtype=np.float64)

        if not np.isfinite(block).all():
            bad = np.argwhere(~np.isfinite(block))[0]
            raise RuntimeError(
                f"Non-finite clean X value at row "
                f"{start + int(bad[0])}, column {int(bad[1])}"
            )

        # Frozen clean z-score.
        z = (block - params["feature_mean"]) / params["feature_std"]

        # Frozen 21-PC projection.
        pcs = z @ params["pca_components"].T

        # Frozen clean per-PC min-max affine transform.
        # IMPORTANT: no clipping.
        norm = (pcs - params["pc_min"]) / params["pc_range"]

        if not np.isfinite(norm).all():
            raise RuntimeError("Non-finite normalized PCA value encountered.")

        pc_global_min = np.minimum(pc_global_min, norm.min(axis=0))
        pc_global_max = np.maximum(pc_global_max, norm.max(axis=0))

        out_flat[valid_flat[start:end]] = norm.astype(np.float32)

        print(
            f"  transformed {end:,}/{X.shape[0]:,} pixels "
            f"({100.0 * end / X.shape[0]:.2f}%)"
        )

    out.flush()
    del out
    gc.collect()

    elapsed = time.perf_counter() - t0

    clean_min = float(pc_global_min.min())
    clean_max = float(pc_global_max.max())

    # The extrema were fitted from this same clean dataset, so only tiny
    # floating-point deviations outside [0,1] would be acceptable.
    if clean_min < -1e-8 or clean_max > 1.0 + 1e-8:
        raise RuntimeError(
            "Clean normalized PCA unexpectedly outside [0,1]: "
            f"min={clean_min}, max={clean_max}"
        )

    # Final structural audit by reopening the completed .npy.
    cache = np.load(CACHE_PATH, mmap_mode="r")

    expected_shape = (H, W, REAL_DEPTH)
    if cache.shape != expected_shape:
        raise RuntimeError(
            f"Cache shape {cache.shape}; expected {expected_shape}"
        )

    if cache.dtype != np.float32:
        raise RuntimeError(
            f"Cache dtype {cache.dtype}; expected float32"
        )

    # Check a fixed small set of valid positions for finite data.
    valid_ids = np.flatnonzero(valid_mask.reshape(-1))
    check_ids = valid_ids[
        np.linspace(
            0,
            len(valid_ids) - 1,
            num=1000,
            dtype=np.int64,
        )
    ]
    cache_flat = cache.reshape(H * W, REAL_DEPTH)

    if not np.isfinite(cache_flat[check_ids]).all():
        raise RuntimeError("Cache audit found non-finite valid values.")

    del cache

    marker = {
        "status": "complete",
        "scientific_protocol": {
            "clean_X_only": True,
            "labels_loaded": False,
            "injected_data_loaded": False,
            "parameters_refit": False,
            "clipping_applied": False,
        },
        "input": {
            "X_path": str(x_path),
            "mask_path": str(mask_path),
            "X_shape": list(X.shape),
            "mask_shape": list(valid_mask.shape),
        },
        "cache": {
            "path": str(CACHE_PATH),
            "shape": [H, W, REAL_DEPTH],
            "dtype": "float32",
            "clean_normalized_min": clean_min,
            "clean_normalized_max": clean_max,
            "build_seconds": elapsed,
        },
        "frozen_artifact_hashes": {
            "preprocessing_params_sha256": sha256_file(
                params["prep_path"]
            ),
            "postpca_normalization_params_sha256": sha256_file(
                params["norm_path"]
            ),
        },
    }

    with open(MARKER_PATH, "w", encoding="utf-8") as f:
        json.dump(marker, f, indent=2)

    print("\n" + "=" * 76)
    print("SSF-AE CLEAN CACHE COMPLETE")
    print("=" * 76)
    print("Cache shape :", (H, W, REAL_DEPTH))
    print("Cache dtype : float32")
    print("Clean range :", clean_min, "to", clean_max)
    print("Build time  :", f"{elapsed:.2f} s")
    print("Marker      :", MARKER_PATH)
    print("\nNext command:")
    print("  python 23_ssfae_clean_training.py")


if __name__ == "__main__":
    main()
