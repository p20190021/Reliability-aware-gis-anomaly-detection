"""
SSF-AE FINAL ALL-15 INJECTED FULL-SCENE GPU SCORING

Purpose
-------
Apply the completely frozen clean-trained SSF-AE to the 15 established
injected cases.

IMPORTANT spatial-scoring rule
------------------------------
Unlike an independent row-wise detector, SSF-AE uses 40x40 spatial windows.
An injected pixel can therefore alter the reconstruction errors of neighboring
non-injected pixels that share a window with it.

For that reason, this script DOES NOT patch only the injected pixel scores.
Instead, for every injected case it:
1. starts from the frozen clean normalized PCA raster;
2. replaces only the authoritative injected locations by the corresponding
   frozen-preprocessed injected PCA rows produced by script 26a;
3. runs ALL 11,298 inference windows;
4. computes per-pixel reconstruction MSE over the 21 real PCA dimensions;
5. fuses CNN/ViT errors using the frozen clean-training loss weights;
6. overlap-averages errors for every valid pixel;
7. saves a full 4,290,958-element native fused score vector.

No training, refitting, threshold tuning, final-score normalization, flood
context, or final-test labels are used in this GPU-scoring process.

This script intentionally avoids Pandas / heavy statistical post-processing.
Run the NumPy/Pandas-only script 26c afterwards for exact ranking and metrics.

Run
---
    python 26b_ssfae_score_all15_injected_fullscene.py

Optional:
    python 26b_ssfae_score_all15_injected_fullscene.py --batch-size 25
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
ROOT_DIR = SCRIPT_DIR.parent

PROTOCOL_DIR = SCRIPT_DIR / "results" / "protocol_freeze"
TRAINING_DIR = SCRIPT_DIR / "results" / "clean_training"
CACHE_DIR = TRAINING_DIR / "cache"
MODEL_DIR = TRAINING_DIR / "models"
PREPARED_DIR = SCRIPT_DIR / "results" / "final_injected_preprocessing"
OUTPUT_DIR = SCRIPT_DIR / "results" / "final_injected_scoring"

RATIO_CONFIG = {
    "0.05%": {"tag": "ratio_0p0005"},
    "0.1%": {"tag": "ratio_0p0010"},
    "0.5%": {"tag": "ratio_0p0050"},
}

SCENARIOS = [
    "extreme",
    "noise_strong",
    "noise_moderate",
    "noise_mild",
    "inconsistent",
]

SCENARIO_DISPLAY = {
    "extreme": "Extreme",
    "noise_strong": "Strong noise",
    "noise_moderate": "Moderate noise",
    "noise_mild": "Mild noise",
    "inconsistent": "Cross-feature inconsistency",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )
    p.add_argument("--progress-every", type=int, default=100)
    p.add_argument(
        "--force",
        action="store_true",
        help="Recompute scenario score files even if complete outputs exist.",
    )
    return p.parse_args()


def require_file(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Required file was not found:\n{path}")
    return path


def resolve_device(name):
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


def build_input_batch(
    clean_raster,
    overlay_lookup,
    injected_rows,
    starts_batch,
):
    """
    Build [B,1,24,40,40] inputs from the clean frozen PCA raster, replacing
    only authoritative injected pixels by their frozen-preprocessed values.
    """
    B = len(starts_batch)

    x = np.zeros(
        (B, 1, MODEL_DEPTH, WINDOW, WINDOW),
        dtype=np.float32,
    )

    for j, (r, c) in enumerate(starts_batch):
        r = int(r)
        c = int(c)

        patch = np.array(
            clean_raster[
                r:r + WINDOW,
                c:c + WINDOW,
                :,
            ],
            dtype=np.float32,
            copy=True,
        )

        if patch.shape != (WINDOW, WINDOW, REAL_DEPTH):
            raise RuntimeError(f"Bad clean patch shape at {(r,c)}: {patch.shape}")

        local_lookup = overlay_lookup[
            r:r + WINDOW,
            c:c + WINDOW,
        ]

        hit = local_lookup >= 0

        if hit.any():
            patch[hit] = injected_rows[
                local_lookup[hit]
            ]

        x[j, 0, :REAL_DEPTH] = np.transpose(
            patch,
            (2, 0, 1),
        )

    return torch.from_numpy(x)


def main():
    args = parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")

    device = resolve_device(args.device)

    mask_path = require_file(ROOT_DIR / "valid_mask_1.npy")
    starts_path = require_file(
        PROTOCOL_DIR / "SSFAE_Inference_Window_Starts.npy"
    )
    clean_cache_path = require_file(
        CACHE_DIR / "SSFAE_Clean_Normalized_PCA_Raster.npy"
    )
    clean_cache_marker = require_file(
        CACHE_DIR / "SSFAE_Clean_Normalized_PCA_Raster_COMPLETE.json"
    )
    cnn_path = require_file(MODEL_DIR / "SSFAE_CNN_Best.pt")
    vit_path = require_file(MODEL_DIR / "SSFAE_ViT_Best.pt")
    fusion_path = require_file(
        TRAINING_DIR / "SSFAE_Frozen_Fusion_Params.npz"
    )
    training_summary_path = require_file(
        TRAINING_DIR / "SSFAE_Clean_Training_Summary.json"
    )
    prep_summary_path = require_file(
        PREPARED_DIR / "SSFAE_Injected_Preprocessing_Summary.json"
    )
    core_path = require_file(SCRIPT_DIR / "ssfae_core.py")

    with open(clean_cache_marker, "r", encoding="utf-8") as f:
        cache_marker = json.load(f)
    if cache_marker.get("status") != "complete":
        raise RuntimeError("Clean normalized-PCA cache marker is not complete.")

    with open(training_summary_path, "r", encoding="utf-8") as f:
        training_summary = json.load(f)

    expected_core_hash = training_summary["audit_hashes"]["ssfae_core_sha256"]
    current_core_hash = sha256_file(core_path)

    if current_core_hash != expected_core_hash:
        raise RuntimeError(
            "ssfae_core.py changed after clean training.\n"
            f"Training hash: {expected_core_hash}\n"
            f"Current hash : {current_core_hash}"
        )

    with open(prep_summary_path, "r", encoding="utf-8") as f:
        prep_summary = json.load(f)

    if prep_summary.get("status") != "COMPLETE":
        raise RuntimeError("Injected preprocessing summary is not complete.")

    valid_mask = np.load(
        mask_path,
        mmap_mode="r",
        allow_pickle=False,
    ).astype(bool, copy=False)

    if valid_mask.shape != EXPECTED_MASK_SHAPE:
        raise RuntimeError(f"Unexpected valid-mask shape: {valid_mask.shape}")
    if int(valid_mask.sum()) != EXPECTED_N:
        raise RuntimeError("Valid-mask count mismatch.")

    valid_flat = np.flatnonzero(valid_mask.reshape(-1))

    clean_raster = np.load(
        clean_cache_path,
        mmap_mode="r",
        allow_pickle=False,
    )

    if clean_raster.shape != (
        EXPECTED_MASK_SHAPE[0],
        EXPECTED_MASK_SHAPE[1],
        REAL_DEPTH,
    ):
        raise RuntimeError(f"Unexpected clean PCA raster shape: {clean_raster.shape}")

    starts = np.load(starts_path, allow_pickle=False).astype(np.int32)
    if starts.ndim != 2 or starts.shape[1] != 2:
        raise RuntimeError("Inference window starts must have shape [n,2].")

    fusion = np.load(fusion_path, allow_pickle=False)
    w_cnn = float(fusion["w_cnn"])
    w_vit = float(fusion["w_vit"])

    if abs((w_cnn + w_vit) - 1.0) > 1e-7:
        raise RuntimeError("Frozen fusion weights do not sum to one.")

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

    cnn_ckpt = load_checkpoint(cnn_path, "3D-CNN-AE", device)
    vit_ckpt = load_checkpoint(vit_path, "3D-ViT-AE", device)

    cnn.load_state_dict(cnn_ckpt["model_state_dict"])
    vit.load_state_dict(vit_ckpt["model_state_dict"])

    cnn.eval()
    vit.eval()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("SSF-AE FINAL ALL-15 INJECTED FULL-SCENE GPU SCORING")
    print("=" * 78)
    print("Device            :", device)
    if device.type == "cuda":
        print("GPU               :", torch.cuda.get_device_name(0))
    print("Valid pixels      :", EXPECTED_N)
    print("Inference windows :", len(starts))
    print("Batch size        :", args.batch_size)
    print("Frozen w_CNN      :", w_cnn)
    print("Frozen w_ViT      :", w_vit)
    print("Full spatial rescore:", "YES")
    print("Patch-only score reuse:", "NO")
    print("Training/refitting:", "NO")
    print("Labels/context    :", "NOT LOADED")

    case_summaries = []
    t_all = time.perf_counter()

    H, W = valid_mask.shape
    n_batches = math.ceil(len(starts) / args.batch_size)

    for ratio_label, cfg_ratio in RATIO_CONFIG.items():
        ratio_tag = cfg_ratio["tag"]
        ratio_prep_dir = PREPARED_DIR / ratio_tag

        inject_idx_path = require_file(
            ratio_prep_dir / "inject_indices.npy"
        )

        inject_idx = np.load(
            inject_idx_path,
            allow_pickle=False,
        ).astype(np.int64, copy=False)

        if inject_idx.ndim != 1:
            raise ValueError(f"Bad inject_idx shape for {ratio_label}.")

        # Map authoritative valid-vector indices to raster positions once per ratio.
        raster_flat_positions = valid_flat[inject_idx]

        overlay_lookup = np.full(
            H * W,
            -1,
            dtype=np.int32,
        )

        overlay_lookup[
            raster_flat_positions
        ] = np.arange(
            inject_idx.size,
            dtype=np.int32,
        )

        overlay_lookup = overlay_lookup.reshape(H, W)

        print("\n" + "=" * 78)
        print(
            f"RATIO {ratio_label}: {inject_idx.size:,} injected pixels"
        )
        print("=" * 78)

        for scenario in SCENARIOS:
            scenario_prep_dir = ratio_prep_dir / scenario
            injected_rows_path = require_file(
                scenario_prep_dir / "SSFAE_Injected_Rows_PCA21.npy"
            )

            injected_rows = np.load(
                injected_rows_path,
                mmap_mode="r",
                allow_pickle=False,
            )

            if injected_rows.shape != (inject_idx.size, REAL_DEPTH):
                raise ValueError(
                    f"Unexpected injected PCA row shape for {ratio_label}/{scenario}: "
                    f"{injected_rows.shape}"
                )

            scenario_out = OUTPUT_DIR / ratio_tag / scenario
            scenario_out.mkdir(parents=True, exist_ok=True)

            score_path = scenario_out / "SSFAE_Injected_Fused_Score.npy"
            summary_path = scenario_out / "SSFAE_Injected_GPU_Scoring_Summary.json"

            if score_path.is_file() and summary_path.is_file() and not args.force:
                try:
                    with open(summary_path, "r", encoding="utf-8") as f:
                        old_summary = json.load(f)

                    if old_summary.get("status") == "COMPLETE":
                        existing = np.load(
                            score_path,
                            mmap_mode="r",
                            allow_pickle=False,
                        )
                        if existing.shape == (EXPECTED_N,):
                            print(
                                f"{SCENARIO_DISPLAY[scenario]:28s} | "
                                "reusing COMPLETE score"
                            )
                            case_summaries.append(old_summary)
                            continue
                except Exception:
                    pass

            print(
                f"\n--- {ratio_label} | {SCENARIO_DISPLAY[scenario]} ---"
            )

            fused_sum = np.zeros(
                (H, W),
                dtype=np.float64,
            )

            coverage = np.zeros(
                (H, W),
                dtype=np.uint16,
            )

            t0 = time.perf_counter()

            with torch.inference_mode():
                for batch_idx, start in enumerate(
                    range(0, len(starts), args.batch_size),
                    start=1,
                ):
                    end = min(start + args.batch_size, len(starts))
                    batch_starts = starts[start:end]

                    x_cpu = build_input_batch(
                        clean_raster,
                        overlay_lookup,
                        injected_rows,
                        batch_starts,
                    )

                    x = x_cpu.to(device, non_blocking=False)

                    cnn_recon = cnn(x)
                    vit_recon = vit(x)

                    cnn_err = (
                        per_pixel_reconstruction_error(
                            x,
                            cnn_recon,
                            REAL_DEPTH,
                        )
                        .cpu()
                        .numpy()
                        .astype(np.float64, copy=False)
                    )

                    vit_err = (
                        per_pixel_reconstruction_error(
                            x,
                            vit_recon,
                            REAL_DEPTH,
                        )
                        .cpu()
                        .numpy()
                        .astype(np.float64, copy=False)
                    )

                    fused_err = (
                        w_cnn * cnn_err
                        + w_vit * vit_err
                    )

                    if not np.isfinite(fused_err).all():
                        raise RuntimeError(
                            f"Non-finite fused error in {ratio_label}/{scenario}."
                        )

                    for j, (r, c) in enumerate(batch_starts):
                        r = int(r)
                        c = int(c)

                        local_valid = valid_mask[
                            r:r + WINDOW,
                            c:c + WINDOW,
                        ]

                        fused_view = fused_sum[
                            r:r + WINDOW,
                            c:c + WINDOW,
                        ]

                        cov_view = coverage[
                            r:r + WINDOW,
                            c:c + WINDOW,
                        ]

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
                            (n_batches - batch_idx) / rate
                            if rate > 0
                            else 0.0
                        )

                        print(
                            f"batch {batch_idx:,}/{n_batches:,} | "
                            f"windows {end:,}/{len(starts):,} | "
                            f"elapsed {elapsed:.1f}s | "
                            f"ETA {remaining:.1f}s"
                        )

            seconds = time.perf_counter() - t0

            valid_cov = coverage[valid_mask]

            if np.any(valid_cov == 0):
                raise RuntimeError(
                    f"Uncovered valid pixels in {ratio_label}/{scenario}."
                )

            if np.count_nonzero(coverage[~valid_mask]) != 0:
                raise RuntimeError(
                    f"Invalid raster cells received coverage in "
                    f"{ratio_label}/{scenario}."
                )

            fused_score = fused_sum[valid_mask] / valid_cov

            if fused_score.shape != (EXPECTED_N,):
                raise RuntimeError("Final fused score length mismatch.")

            if not np.isfinite(fused_score).all():
                raise RuntimeError("Final fused score contains non-finite values.")

            if np.any(fused_score < 0.0):
                raise RuntimeError("MSE score unexpectedly contains negative values.")

            np.save(
                score_path,
                np.asarray(fused_score, dtype=np.float64),
            )

            case_summary = {
                "status": "COMPLETE",
                "ratio_label": ratio_label,
                "ratio_tag": ratio_tag,
                "scenario": scenario,
                "scenario_display": SCENARIO_DISPLAY[scenario],
                "total_injected": int(inject_idx.size),
                "score_path": str(score_path),
                "score_shape": [EXPECTED_N],
                "score_min": float(fused_score.min()),
                "score_max": float(fused_score.max()),
                "score_mean": float(fused_score.mean()),
                "injected_score_mean": float(fused_score[inject_idx].mean()),
                "coverage_min": int(valid_cov.min()),
                "coverage_max": int(valid_cov.max()),
                "scoring_seconds": float(seconds),
                "scientific_protocol": {
                    "training_performed": False,
                    "refitting_performed": False,
                    "labels_loaded": False,
                    "flood_context_loaded": False,
                    "final_score_normalization": False,
                    "higher_score_is_more_anomalous": True,
                    "all_spatial_windows_rescored": True,
                    "clean_scores_patched_only_at_injected_indices": False,
                },
            }

            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(case_summary, f, indent=2)

            case_summaries.append(case_summary)

            print(
                f"Saved | score range "
                f"[{case_summary['score_min']:.6g}, "
                f"{case_summary['score_max']:.6g}] | "
                f"{seconds:.2f}s"
            )

    aggregate = {
        "stage": "SSFAE_final_all15_injected_fullscene_gpu_scoring",
        "status": "COMPLETE",
        "cases_expected": 15,
        "cases_complete": len(case_summaries),
        "core_sha256": current_core_hash,
        "cnn_best_epoch": int(cnn_ckpt["epoch"]),
        "vit_best_epoch": int(vit_ckpt["epoch"]),
        "w_cnn": w_cnn,
        "w_vit": w_vit,
        "inference_windows": int(len(starts)),
        "batch_size": int(args.batch_size),
        "scientific_protocol": {
            "training_performed": False,
            "refitting_performed": False,
            "labels_loaded": False,
            "flood_context_loaded": False,
            "final_score_normalization": False,
            "higher_score_is_more_anomalous": True,
            "all_spatial_windows_rescored_per_case": True,
            "reason_full_rescoring_required": (
                "SSF-AE uses spatial windows; injected pixels can alter "
                "reconstruction errors of neighboring pixels sharing a window."
            ),
        },
        "cases": case_summaries,
        "total_seconds_this_run": float(time.perf_counter() - t_all),
        "next_step": (
            "Run 26c_ssfae_final_all15_budget_evaluation.py in a fresh process "
            "for exact top-k ranking, tie audit, and injected detection rates."
        ),
    }

    aggregate_path = OUTPUT_DIR / "SSFAE_All15_GPU_Scoring_Summary.json"
    with open(aggregate_path, "w", encoding="utf-8") as f:
        json.dump(aggregate, f, indent=2)

    print("\n" + "=" * 78)
    print("SSF-AE ALL-15 GPU SCORING COMPLETE")
    print("=" * 78)
    print("Cases complete :", len(case_summaries))
    print("Summary        :", aggregate_path)
    print("\nNext command:")
    print("  python 26c_ssfae_final_all15_budget_evaluation.py")


if __name__ == "__main__":
    main()
