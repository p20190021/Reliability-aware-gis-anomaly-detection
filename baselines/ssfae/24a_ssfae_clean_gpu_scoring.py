"""
SSF-AE CLEAN FULL-SCENE GPU SCORING ONLY

Purpose
-------
This script deliberately separates PyTorch GPU inference from NumPy/Pandas
post-processing to avoid Windows Intel OpenMP runtime conflicts.

It performs:
- NO training
- NO refitting
- NO injected-data evaluation
- NO labels/context loading
- frozen clean-model inference only
- overlap averaging only
- saving native clean score vectors and coverage

After this finishes, run:
    python 24b_ssfae_clean_score_audit.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from ssfae_core import (
    SSFAEConfig,
    CNN3DAutoencoder,
    ViT3DAutoencoder,
    per_pixel_reconstruction_error,
)

EXPECTED_N = 4_290_958
EXPECTED_MASK_SHAPE = (2897, 2696)
REAL_DEPTH = 21
MODEL_DEPTH = 24
WINDOW = 40
DEFAULT_BATCH_SIZE = 25

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
PROTOCOL_DIR = SCRIPT_DIR / "results" / "protocol_freeze"
TRAINING_DIR = SCRIPT_DIR / "results" / "clean_training"
CACHE_DIR = TRAINING_DIR / "cache"
MODEL_DIR = TRAINING_DIR / "models"
OUTPUT_DIR = SCRIPT_DIR / "results" / "clean_scoring"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument(
        "--device", choices=["auto", "cpu", "cuda"], default="auto"
    )
    p.add_argument("--progress-every", type=int, default=25)
    return p.parse_args()


def resolve_device(name: str):
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sha256_file(path: Path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def load_checkpoint(path: Path, branch: str, device):
    ckpt = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )
    if ckpt.get("branch") != branch:
        raise RuntimeError(
            f"Checkpoint branch mismatch: {ckpt.get('branch')} != {branch}"
        )
    return ckpt


def build_input_batch(raster, starts_batch):
    B = len(starts_batch)
    x = np.zeros(
        (B, 1, MODEL_DEPTH, WINDOW, WINDOW),
        dtype=np.float32,
    )

    for j, (r, c) in enumerate(starts_batch):
        r = int(r)
        c = int(c)
        patch = np.asarray(
            raster[r:r + WINDOW, c:c + WINDOW, :],
            dtype=np.float32,
        )
        if patch.shape != (WINDOW, WINDOW, REAL_DEPTH):
            raise RuntimeError(f"Bad patch shape at {(r,c)}: {patch.shape}")

        x[j, 0, :REAL_DEPTH] = np.transpose(patch, (2, 0, 1))

    return torch.from_numpy(x)


def main():
    args = parse_args()
    device = resolve_device(args.device)

    paths = {
        "mask": PROJECT_ROOT / "valid_mask_1.npy",
        "core": SCRIPT_DIR / "ssfae_core.py",
        "starts": PROTOCOL_DIR / "SSFAE_Inference_Window_Starts.npy",
        "cache": CACHE_DIR / "SSFAE_Clean_Normalized_PCA_Raster.npy",
        "cache_marker": (
            CACHE_DIR / "SSFAE_Clean_Normalized_PCA_Raster_COMPLETE.json"
        ),
        "cnn": MODEL_DIR / "SSFAE_CNN_Best.pt",
        "vit": MODEL_DIR / "SSFAE_ViT_Best.pt",
        "fusion": TRAINING_DIR / "SSFAE_Frozen_Fusion_Params.npz",
        "training_summary": TRAINING_DIR / "SSFAE_Clean_Training_Summary.json",
    }

    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {name}: {path}")

    with open(paths["cache_marker"], "r", encoding="utf-8") as f:
        marker = json.load(f)
    if marker.get("status") != "complete":
        raise RuntimeError("Clean PCA cache marker is not complete.")

    with open(paths["training_summary"], "r", encoding="utf-8") as f:
        training_summary = json.load(f)

    expected_hash = training_summary["audit_hashes"]["ssfae_core_sha256"]
    current_hash = sha256_file(paths["core"])
    if current_hash != expected_hash:
        raise RuntimeError(
            "ssfae_core.py differs from the exact training-time core.\n"
            f"Training hash: {expected_hash}\n"
            f"Current hash : {current_hash}"
        )

    valid_mask = np.load(paths["mask"], mmap_mode="r").astype(bool, copy=False)
    if valid_mask.shape != EXPECTED_MASK_SHAPE:
        raise RuntimeError(f"Unexpected mask shape: {valid_mask.shape}")
    if int(valid_mask.sum()) != EXPECTED_N:
        raise RuntimeError("Valid-mask count mismatch.")

    raster = np.load(paths["cache"], mmap_mode="r")
    expected_raster_shape = (
        EXPECTED_MASK_SHAPE[0],
        EXPECTED_MASK_SHAPE[1],
        REAL_DEPTH,
    )
    if raster.shape != expected_raster_shape:
        raise RuntimeError(f"Unexpected PCA cache shape: {raster.shape}")

    starts = np.load(paths["starts"]).astype(np.int32)
    if starts.ndim != 2 or starts.shape[1] != 2:
        raise RuntimeError("Inference starts must be [n,2].")

    fusion = np.load(paths["fusion"])
    w_cnn = float(fusion["w_cnn"])
    w_vit = float(fusion["w_vit"])
    cnn_train_loss = float(fusion["cnn_training_loss"])
    vit_train_loss = float(fusion["vit_training_loss"])

    if abs((w_cnn + w_vit) - 1.0) > 1e-7:
        raise RuntimeError("Fusion weights do not sum to 1.")

    cfg = SSFAEConfig(
        input_depth=24,
        real_depth=21,
        height=40,
        width=40,
        cnn_c1=16,
        cnn_c2=32,
        cnn_kernel=3,
        patch_size=4,
        embed_dim=64,
        num_heads=6,
        num_encoder_layers=4,
        num_decoder_layers=4,
        ffn_dim=128,
        dropout=0.1,
    )

    cnn = CNN3DAutoencoder(cfg).to(device)
    vit = ViT3DAutoencoder(cfg).to(device)

    cnn_ckpt = load_checkpoint(paths["cnn"], "3D-CNN-AE", device)
    vit_ckpt = load_checkpoint(paths["vit"], "3D-ViT-AE", device)

    cnn.load_state_dict(cnn_ckpt["model_state_dict"])
    vit.load_state_dict(vit_ckpt["model_state_dict"])

    cnn.eval()
    vit.eval()

    H, W = valid_mask.shape

    cnn_sum = np.zeros((H, W), dtype=np.float64)
    vit_sum = np.zeros((H, W), dtype=np.float64)
    fused_sum = np.zeros((H, W), dtype=np.float64)
    coverage = np.zeros((H, W), dtype=np.uint16)

    n_windows = len(starts)
    n_batches = math.ceil(n_windows / args.batch_size)

    print("=" * 76)
    print("SSF-AE CLEAN GPU SCORING ONLY")
    print("=" * 76)
    print("Device            :", device)
    if device.type == "cuda":
        print("GPU               :", torch.cuda.get_device_name(0))
    print("Valid pixels      :", EXPECTED_N)
    print("Inference windows :", n_windows)
    print("Batch size        :", args.batch_size)
    print("Inference batches :", n_batches)
    print("CNN train MSE     :", cnn_train_loss)
    print("ViT train MSE     :", vit_train_loss)
    print("Frozen w_CNN      :", w_cnn)
    print("Frozen w_ViT      :", w_vit)
    print("Injected data     : NOT LOADED")
    print("Labels/context    : NOT LOADED")

    t0 = time.perf_counter()

    with torch.inference_mode():
        for batch_idx, start in enumerate(
            range(0, n_windows, args.batch_size),
            start=1,
        ):
            end = min(start + args.batch_size, n_windows)
            batch_starts = starts[start:end]

            x_cpu = build_input_batch(raster, batch_starts)
            x = x_cpu.to(device, non_blocking=False)

            cnn_recon = cnn(x)
            vit_recon = vit(x)

            cnn_err = (
                per_pixel_reconstruction_error(
                    x, cnn_recon, REAL_DEPTH
                )
                .cpu()
                .numpy()
                .astype(np.float64, copy=False)
            )
            vit_err = (
                per_pixel_reconstruction_error(
                    x, vit_recon, REAL_DEPTH
                )
                .cpu()
                .numpy()
                .astype(np.float64, copy=False)
            )

            fused_err = w_cnn * cnn_err + w_vit * vit_err

            for j, (r, c) in enumerate(batch_starts):
                r = int(r)
                c = int(c)

                local_valid = valid_mask[
                    r:r + WINDOW,
                    c:c + WINDOW,
                ]

                cnn_view = cnn_sum[r:r + WINDOW, c:c + WINDOW]
                vit_view = vit_sum[r:r + WINDOW, c:c + WINDOW]
                fused_view = fused_sum[r:r + WINDOW, c:c + WINDOW]
                cov_view = coverage[r:r + WINDOW, c:c + WINDOW]

                cnn_view[local_valid] += cnn_err[j][local_valid]
                vit_view[local_valid] += vit_err[j][local_valid]
                fused_view[local_valid] += fused_err[j][local_valid]
                cov_view[local_valid] += 1

            if (
                batch_idx == 1
                or batch_idx % args.progress_every == 0
                or batch_idx == n_batches
            ):
                elapsed = time.perf_counter() - t0
                rate = batch_idx / elapsed if elapsed > 0 else 0.0
                remaining = (
                    (n_batches - batch_idx) / rate if rate > 0 else 0.0
                )
                print(
                    f"batch {batch_idx:,}/{n_batches:,} | "
                    f"windows {end:,}/{n_windows:,} | "
                    f"elapsed {elapsed/60:.2f} min | "
                    f"ETA {remaining/60:.2f} min"
                )

    scoring_seconds = time.perf_counter() - t0

    valid_cov = coverage[valid_mask]
    uncovered = int(np.count_nonzero(valid_cov == 0))
    if uncovered != 0:
        raise RuntimeError(f"Uncovered valid pixels: {uncovered}")

    if np.count_nonzero(coverage[~valid_mask]) != 0:
        raise RuntimeError("Invalid raster cells received score coverage.")

    cnn_score = cnn_sum[valid_mask] / valid_cov
    vit_score = vit_sum[valid_mask] / valid_cov
    fused_score = fused_sum[valid_mask] / valid_cov

    if cnn_score.size != EXPECTED_N:
        raise RuntimeError("CNN score length mismatch.")
    if vit_score.size != EXPECTED_N:
        raise RuntimeError("ViT score length mismatch.")
    if fused_score.size != EXPECTED_N:
        raise RuntimeError("Fused score length mismatch.")

    # Lightweight pointwise audit only. Heavy statistical analysis is
    # intentionally deferred to the NumPy-only process.
    max_abs_fusion_error = float(
        np.max(
            np.abs(
                fused_score
                - (w_cnn * cnn_score + w_vit * vit_score)
            )
        )
    )
    if max_abs_fusion_error > 1e-12:
        raise RuntimeError(
            f"Fusion reproduction failed: {max_abs_fusion_error}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cnn_path = OUTPUT_DIR / "SSFAE_Clean_CNN_Score.npy"
    vit_path = OUTPUT_DIR / "SSFAE_Clean_ViT_Score.npy"
    fused_path = OUTPUT_DIR / "SSFAE_Clean_Fused_Score.npy"
    cov_path = OUTPUT_DIR / "SSFAE_Clean_Window_Coverage.npy"

    np.save(cnn_path, np.asarray(cnn_score, dtype=np.float64))
    np.save(vit_path, np.asarray(vit_score, dtype=np.float64))
    np.save(fused_path, np.asarray(fused_score, dtype=np.float64))
    np.save(cov_path, np.asarray(valid_cov, dtype=np.uint16))

    gpu_summary = {
        "status": "clean_gpu_scoring_complete_postaudit_not_run",
        "scientific_protocol": {
            "training_performed": False,
            "refitting_performed": False,
            "labels_loaded": False,
            "flood_context_loaded": False,
            "injected_data_loaded": False,
            "score_orientation": "higher reconstruction MSE = more anomalous",
        },
        "model": {
            "cnn_best_epoch": int(cnn_ckpt["epoch"]),
            "vit_best_epoch": int(vit_ckpt["epoch"]),
            "core_sha256": current_hash,
        },
        "fusion": {
            "cnn_training_loss": cnn_train_loss,
            "vit_training_loss": vit_train_loss,
            "w_cnn": w_cnn,
            "w_vit": w_vit,
            "fusion_reproduction_max_abs_error": max_abs_fusion_error,
        },
        "coverage": {
            "inference_windows": int(n_windows),
            "valid_pixels": EXPECTED_N,
            "uncovered_valid_pixels": uncovered,
            "min": int(valid_cov.min()),
            "max": int(valid_cov.max()),
        },
        "runtime": {
            "batch_size": int(args.batch_size),
            "scoring_seconds": float(scoring_seconds),
        },
        "outputs": {
            "cnn_score": str(cnn_path),
            "vit_score": str(vit_path),
            "fused_score": str(fused_path),
            "coverage": str(cov_path),
        },
        "next_step": "Run 24b_ssfae_clean_score_audit.py in a fresh process.",
    }

    summary_path = OUTPUT_DIR / "SSFAE_Clean_GPU_Scoring_Summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(gpu_summary, f, indent=2)

    print("\n" + "=" * 76)
    print("SSF-AE CLEAN GPU SCORING COMPLETE")
    print("=" * 76)
    print("Scoring time      :", f"{scoring_seconds:.2f} s")
    print("Uncovered pixels  :", uncovered)
    print("Coverage min/max  :", int(valid_cov.min()), "/", int(valid_cov.max()))
    print("Fusion max |delta|:", max_abs_fusion_error)
    print("Saved fused score :", fused_path)
    print("Summary           :", summary_path)
    print("\nNext command:")
    print("  python 24b_ssfae_clean_score_audit.py")


if __name__ == "__main__":
    main()
