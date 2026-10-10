"""
SSF-AE clean-only training pipeline.

SCIENTIFIC STATUS
-----------------
This script is executed BEFORE final injected evaluation.

It:
- uses original clean X.npy only;
- uses already frozen preprocessing/PCA/min-max parameters;
- uses already frozen clean train/validation spatial windows;
- does NOT load y.npy, flood-risk labels, injected datasets, or injected indices;
- trains the 3D-CNN AE and 3D-ViT AE independently;
- uses clean validation MSE only for early stopping;
- saves the best clean-validation checkpoint for each branch;
- recomputes clean TRAINING MSE for the two frozen best checkpoints;
- freezes SSF-AE's dynamic fusion weights from those clean training losses.

Expected location:
    .../Code/SSFAE/23_ssfae_clean_training.py

Required files:
    .../Code/X.npy
    .../Code/valid_mask_1.npy
    .../Code/SSFAE/ssfae_core.py
    .../Code/SSFAE/results/protocol_freeze/
        SSFAE_Preprocessing_Params.npz
        SSFAE_PostPCA_Normalization_Params.npz
        SSFAE_Train_Window_Starts.npy
        SSFAE_Validation_Window_Starts.npy

IMPORTANT
---------
Before running this script, replace the previous ssfae_core.py with the
updated paper-verified core and rerun:

    python 22_ssfae_model_selftest_benchmark.py --batch-sizes 25

Only if SELF-TEST passes and batch 25 is OK, run:

    python 23_ssfae_clean_training.py
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from ssfae_core import (
    SSFAEConfig,
    CNN3DAutoencoder,
    ViT3DAutoencoder,
    reconstruction_mse,
    dynamic_fusion_weights,
    count_trainable_parameters,
)


# ---------------------------------------------------------------------
# Frozen protocol
# ---------------------------------------------------------------------
SEED = 42

BATCH_SIZE = 25
MAX_EPOCHS = 70
EARLY_STOP_PATIENCE = 15
MIN_IMPROVEMENT = 1e-8

# Paper states Adam but does not report a numerical learning rate.
# 1e-3 is frozen as the standard Adam default before any injected evaluation.
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.0

NUM_WORKERS = 4
CHUNK_ROWS = 100_000

EXPECTED_N = 4_290_958
EXPECTED_D = 23
EXPECTED_MASK_SHAPE = (2897, 2696)

REAL_DEPTH = 21
MODEL_DEPTH = 24
WINDOW = 40

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PROJECT_ROOT = SCRIPT_DIR.parent
PROTOCOL_DIR = SCRIPT_DIR / "results" / "protocol_freeze"
OUTPUT_DIR = SCRIPT_DIR / "results" / "clean_training"
CACHE_DIR = OUTPUT_DIR / "cache"
MODEL_DIR = OUTPUT_DIR / "models"
HISTORY_DIR = OUTPUT_DIR / "histories"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", type=Path, default=None)
    p.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )
    p.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    p.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Rebuild clean normalized-PCA raster cache.",
    )
    return p.parse_args()


def resolve_device(name: str):
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


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
    train_path = PROTOCOL_DIR / "SSFAE_Train_Window_Starts.npy"
    val_path = PROTOCOL_DIR / "SSFAE_Validation_Window_Starts.npy"

    for path in [prep_path, norm_path, train_path, val_path]:
        if not path.exists():
            raise FileNotFoundError(f"Missing frozen artifact: {path}")

    prep = np.load(prep_path)
    norm = np.load(norm_path)

    params = {
        "feature_mean": np.asarray(prep["feature_mean"], dtype=np.float64),
        "feature_std": np.asarray(prep["feature_std"], dtype=np.float64),
        "pca_components": np.asarray(
            prep["pca_components"], dtype=np.float64
        ),
        "pc_min": np.asarray(norm["pc_min"], dtype=np.float64),
        "pc_range": np.asarray(norm["pc_range"], dtype=np.float64),
        "neutral_fill": np.asarray(
            norm["neutral_fill"], dtype=np.float64
        ),
        "train_starts": np.load(train_path).astype(np.int32),
        "val_starts": np.load(val_path).astype(np.int32),
        "paths": {
            "prep": prep_path,
            "norm": norm_path,
            "train": train_path,
            "val": val_path,
        },
    }

    k = int(np.asarray(prep["retained_components"]).item())

    if k != REAL_DEPTH:
        raise ValueError(f"Expected {REAL_DEPTH} PCs, got {k}.")
    if params["pca_components"].shape != (REAL_DEPTH, EXPECTED_D):
        raise ValueError("Unexpected PCA component shape.")
    if params["pc_min"].shape != (REAL_DEPTH,):
        raise ValueError("Unexpected pc_min shape.")
    if params["pc_range"].shape != (REAL_DEPTH,):
        raise ValueError("Unexpected pc_range shape.")
    if params["neutral_fill"].shape != (REAL_DEPTH,):
        raise ValueError("Unexpected neutral_fill shape.")

    return params


def build_clean_normalized_pca_raster(
    X,
    valid_mask,
    params,
    cache_path: Path,
):
    """
    Create reusable memory-mapped [H,W,21] clean normalized PCA raster.
    Invalid cells receive the frozen neutral PC=0 fill.
    """
    H, W = valid_mask.shape

    out = np.lib.format.open_memmap(
        cache_path,
        mode="w+",
        dtype=np.float32,
        shape=(H, W, REAL_DEPTH),
    )

    out[:] = params["neutral_fill"].astype(np.float32)

    out_flat = out.reshape(H * W, REAL_DEPTH)
    valid_flat = np.flatnonzero(valid_mask.reshape(-1))

    if valid_flat.size != X.shape[0]:
        raise RuntimeError("Mask/X row count mismatch.")

    global_min = np.full(REAL_DEPTH, np.inf, dtype=np.float64)
    global_max = np.full(REAL_DEPTH, -np.inf, dtype=np.float64)

    print("\nBuilding clean normalized-PCA raster cache:")
    print(" ", cache_path)

    for start in range(0, X.shape[0], CHUNK_ROWS):
        end = min(start + CHUNK_ROWS, X.shape[0])

        block = np.asarray(X[start:end], dtype=np.float64)

        z = (block - params["feature_mean"]) / params["feature_std"]
        pcs = z @ params["pca_components"].T
        norm = (pcs - params["pc_min"]) / params["pc_range"]

        if not np.isfinite(norm).all():
            raise RuntimeError("Non-finite normalized PCA values.")

        global_min = np.minimum(global_min, norm.min(axis=0))
        global_max = np.maximum(global_max, norm.max(axis=0))

        out_flat[valid_flat[start:end]] = norm.astype(np.float32)

        print(f"  transformed {end:,}/{X.shape[0]:,} pixels")

    out.flush()
    del out
    gc.collect()

    min_all = float(global_min.min())
    max_all = float(global_max.max())

    if min_all < -1e-8 or max_all > 1.0 + 1e-8:
        raise RuntimeError(
            "Clean normalized PCA exceeds expected [0,1]: "
            f"min={min_all}, max={max_all}"
        )

    print("Clean normalized PCA range:", min_all, "to", max_all)

    return min_all, max_all


class WindowDataset(Dataset):
    """Lazy memory-mapped frozen-window dataset."""

    def __init__(self, raster_path: Path, starts: np.ndarray):
        self.raster_path = Path(raster_path)
        self.starts = np.asarray(starts, dtype=np.int32)
        self._raster = None

        if self.starts.ndim != 2 or self.starts.shape[1] != 2:
            raise ValueError("starts must have shape [n,2].")

    def _get_raster(self):
        if self._raster is None:
            self._raster = np.load(self.raster_path, mmap_mode="r")
        return self._raster

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        raster = self._get_raster()
        r = int(self.starts[idx, 0])
        c = int(self.starts[idx, 1])

        patch = np.asarray(
            raster[r:r + WINDOW, c:c + WINDOW, :],
            dtype=np.float32,
        )

        if patch.shape != (WINDOW, WINDOW, REAL_DEPTH):
            raise RuntimeError(f"Bad patch shape: {patch.shape}")

        real = np.transpose(patch, (2, 0, 1))

        padded = np.zeros(
            (MODEL_DEPTH, WINDOW, WINDOW),
            dtype=np.float32,
        )
        padded[:REAL_DEPTH] = real

        # [C=1,D,H,W]
        return torch.from_numpy(padded[None, ...])


def make_loaders(
    raster_path,
    train_starts,
    val_starts,
    num_workers,
    device,
):
    train_ds = WindowDataset(raster_path, train_starts)
    val_ds = WindowDataset(raster_path, val_starts)

    generator = torch.Generator()
    generator.manual_seed(SEED)

    common = dict(
        batch_size=BATCH_SIZE,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
    )

    train_loader = DataLoader(
        train_ds,
        shuffle=True,
        generator=generator,
        **common,
    )

    val_loader = DataLoader(
        val_ds,
        shuffle=False,
        **common,
    )

    return train_loader, val_loader


@torch.no_grad()
def evaluate_loss(model, loader, device):
    model.eval()

    total_sq = 0.0
    total_elements = 0

    for x in loader:
        x = x.to(device, non_blocking=True)
        y = model(x)

        t = x[:, :, :REAL_DEPTH, :, :]
        r = y[:, :, :REAL_DEPTH, :, :]

        total_sq += float(torch.sum((r - t) ** 2).cpu())
        total_elements += int(t.numel())

    return total_sq / total_elements


def train_branch(
    branch_name,
    model,
    train_loader,
    val_loader,
    device,
    checkpoint_path,
    history_path,
):
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    best_val = np.inf
    best_epoch = None
    no_improve = 0
    history = []

    start_all = time.perf_counter()

    print("\n" + "=" * 76)
    print(f"TRAINING {branch_name}")
    print("=" * 76)

    for epoch in range(1, MAX_EPOCHS + 1):
        start_epoch = time.perf_counter()

        model.train()
        train_weighted = 0.0
        train_elements = 0

        for x in train_loader:
            x = x.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            y = model(x)
            loss = reconstruction_mse(x, y, REAL_DEPTH)
            loss.backward()
            optimizer.step()

            n = x.shape[0] * REAL_DEPTH * WINDOW * WINDOW
            train_weighted += float(loss.detach().cpu()) * n
            train_elements += int(n)

        train_loss = train_weighted / train_elements
        val_loss = evaluate_loss(model, val_loader, device)

        improved = val_loss < best_val - MIN_IMPROVEMENT

        if improved:
            best_val = val_loss
            best_epoch = epoch
            no_improve = 0

            torch.save(
                {
                    "branch": branch_name,
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_loss": best_val,
                    "seed": SEED,
                    "batch_size": BATCH_SIZE,
                    "learning_rate": LEARNING_RATE,
                    "max_epochs": MAX_EPOCHS,
                    "patience": EARLY_STOP_PATIENCE,
                },
                checkpoint_path,
            )
        else:
            no_improve += 1

        elapsed = time.perf_counter() - start_epoch

        history.append(
            {
                "epoch": epoch,
                "train_mse": train_loss,
                "validation_mse": val_loss,
                "best_validation_mse": best_val,
                "is_best": bool(improved),
                "epochs_without_improvement": no_improve,
                "epoch_seconds": elapsed,
            }
        )

        pd.DataFrame(history).to_csv(history_path, index=False)

        print(
            f"{branch_name} epoch {epoch:03d}/{MAX_EPOCHS} | "
            f"train={train_loss:.8f} | "
            f"val={val_loss:.8f} | "
            f"best={best_val:.8f} (epoch {best_epoch}) | "
            f"{elapsed:.2f}s"
        )

        if no_improve >= EARLY_STOP_PATIENCE:
            print(
                f"Early stopping after {EARLY_STOP_PATIENCE} "
                "epochs without clean-validation improvement."
            )
            break

    total_seconds = time.perf_counter() - start_all

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    frozen_train_mse = evaluate_loss(model, train_loader, device)
    frozen_val_mse = evaluate_loss(model, val_loader, device)

    print(f"\n{branch_name} frozen checkpoint")
    print("  best epoch     :", best_epoch)
    print("  train MSE      :", frozen_train_mse)
    print("  validation MSE :", frozen_val_mse)
    print("  total seconds  :", total_seconds)

    return {
        "best_epoch": int(best_epoch),
        "best_validation_mse_during_training": float(best_val),
        "frozen_train_mse": float(frozen_train_mse),
        "frozen_validation_mse": float(frozen_val_mse),
        "epochs_run": len(history),
        "training_seconds": float(total_seconds),
        "checkpoint": str(checkpoint_path),
        "history": str(history_path),
    }


def main():
    args = parse_args()
    set_seed(SEED)

    device = resolve_device(args.device)

    project_root = (
        args.project_root.resolve()
        if args.project_root is not None
        else DEFAULT_PROJECT_ROOT.resolve()
    )

    for d in [OUTPUT_DIR, CACHE_DIR, MODEL_DIR, HISTORY_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    x_path = project_root / "X.npy"
    mask_path = project_root / "valid_mask_1.npy"

    if not x_path.exists():
        raise FileNotFoundError(x_path)
    if not mask_path.exists():
        raise FileNotFoundError(mask_path)

    X = np.load(x_path, mmap_mode="r")
    valid_mask = np.load(mask_path, mmap_mode="r").astype(bool, copy=False)

    if X.shape != (EXPECTED_N, EXPECTED_D):
        raise ValueError(f"Unexpected X shape: {X.shape}")
    if valid_mask.shape != EXPECTED_MASK_SHAPE:
        raise ValueError(f"Unexpected mask shape: {valid_mask.shape}")
    if int(valid_mask.sum()) != EXPECTED_N:
        raise ValueError("Mask valid-count mismatch.")

    params = load_frozen_params()

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
    cfg.validate()

    print("=" * 76)
    print("SSF-AE CLEAN-ONLY TRAINING")
    print("=" * 76)
    print("Device        :", device)
    print("PyTorch       :", torch.__version__)
    if device.type == "cuda":
        print("GPU            :", torch.cuda.get_device_name(0))
    print("X              :", X.shape)
    print("Train windows  :", len(params["train_starts"]))
    print("Val windows    :", len(params["val_starts"]))
    print("Batch size     :", BATCH_SIZE)
    print("Max epochs     :", MAX_EPOCHS)
    print("Patience       :", EARLY_STOP_PATIENCE)
    print("Adam LR        :", LEARNING_RATE)
    print("Injected data  : NOT LOADED")
    print("Labels         : NOT LOADED")

    cache_path = CACHE_DIR / "SSFAE_Clean_Normalized_PCA_Raster.npy"

    cache_min = None
    cache_max = None

    if args.rebuild_cache or not cache_path.exists():
        cache_min, cache_max = build_clean_normalized_pca_raster(
            X,
            valid_mask,
            params,
            cache_path,
        )
    else:
        arr = np.load(cache_path, mmap_mode="r")
        expected_cache_shape = (
            EXPECTED_MASK_SHAPE[0],
            EXPECTED_MASK_SHAPE[1],
            REAL_DEPTH,
        )
        if arr.shape != expected_cache_shape:
            raise ValueError(
                f"Unexpected existing cache shape: {arr.shape}"
            )
        del arr
        print("\nUsing existing cache:")
        print(" ", cache_path)

    train_loader, val_loader = make_loaders(
        cache_path,
        params["train_starts"],
        params["val_starts"],
        args.num_workers,
        device,
    )

    cnn = CNN3DAutoencoder(cfg).to(device)
    print("\nCNN trainable parameters:", f"{count_trainable_parameters(cnn):,}")

    cnn_result = train_branch(
        "3D-CNN-AE",
        cnn,
        train_loader,
        val_loader,
        device,
        MODEL_DIR / "SSFAE_CNN_Best.pt",
        HISTORY_DIR / "SSFAE_CNN_Training_History.csv",
    )

    del cnn
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    vit = ViT3DAutoencoder(cfg).to(device)
    print("\nViT trainable parameters:", f"{count_trainable_parameters(vit):,}")

    vit_result = train_branch(
        "3D-ViT-AE",
        vit,
        train_loader,
        val_loader,
        device,
        MODEL_DIR / "SSFAE_ViT_Best.pt",
        HISTORY_DIR / "SSFAE_ViT_Training_History.csv",
    )

    del vit
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    cnn_loss = cnn_result["frozen_train_mse"]
    vit_loss = vit_result["frozen_train_mse"]

    w_cnn, w_vit = dynamic_fusion_weights(cnn_loss, vit_loss)
    w_cnn = float(w_cnn.item())
    w_vit = float(w_vit.item())

    fusion_path = OUTPUT_DIR / "SSFAE_Frozen_Fusion_Params.npz"
    np.savez(
        fusion_path,
        cnn_training_loss=np.float64(cnn_loss),
        vit_training_loss=np.float64(vit_loss),
        w_cnn=np.float64(w_cnn),
        w_vit=np.float64(w_vit),
    )

    core_path = SCRIPT_DIR / "ssfae_core.py"

    summary = {
        "status": "clean_training_complete_injected_data_not_used",
        "scientific_protocol": {
            "training_data": "original clean data only",
            "labels_loaded": False,
            "flood_context_loaded": False,
            "injected_data_loaded": False,
            "hyperparameter_tuning_on_final_injections": False,
            "branch_training": "independent_sequential",
            "early_stopping_source": "frozen clean validation windows",
        },
        "device": {
            "torch_version": torch.__version__,
            "device": str(device),
            "gpu": (
                torch.cuda.get_device_name(0)
                if device.type == "cuda"
                else None
            ),
        },
        "data": {
            "X_path": str(x_path),
            "mask_path": str(mask_path),
            "X_shape": list(X.shape),
            "mask_shape": list(valid_mask.shape),
            "train_windows": int(len(params["train_starts"])),
            "validation_windows": int(len(params["val_starts"])),
            "clean_pca_cache": str(cache_path),
            "cache_clean_min": cache_min,
            "cache_clean_max": cache_max,
        },
        "architecture": {
            **cfg.__dict__,
            "status": "frozen independent paper-faithful reimplementation",
        },
        "training": {
            "seed": SEED,
            "batch_size": BATCH_SIZE,
            "optimizer": "Adam",
            "learning_rate": LEARNING_RATE,
            "learning_rate_note": (
                "Paper specifies Adam but not numerical LR; "
                "1e-3 frozen as standard Adam default."
            ),
            "weight_decay": WEIGHT_DECAY,
            "loss": "MSE over first 21 real normalized PCA channels only",
            "max_epochs": MAX_EPOCHS,
            "early_stopping_patience": EARLY_STOP_PATIENCE,
            "min_improvement": MIN_IMPROVEMENT,
            "augmentation": "none",
        },
        "cnn": cnn_result,
        "vit": vit_result,
        "fusion": {
            "cnn_training_loss": cnn_loss,
            "vit_training_loss": vit_loss,
            "w_cnn": w_cnn,
            "w_vit": w_vit,
            "formula": (
                "w_cnn=L_vit/(L_cnn+L_vit); "
                "w_vit=L_cnn/(L_cnn+L_vit)"
            ),
        },
        "audit_hashes": {
            "ssfae_core_sha256": (
                sha256_file(core_path) if core_path.exists() else None
            ),
            "training_script_sha256": sha256_file(Path(__file__)),
            "preprocessing_params_sha256": sha256_file(
                params["paths"]["prep"]
            ),
            "postpca_norm_params_sha256": sha256_file(
                params["paths"]["norm"]
            ),
            "train_windows_sha256": sha256_file(
                params["paths"]["train"]
            ),
            "val_windows_sha256": sha256_file(
                params["paths"]["val"]
            ),
        },
        "next_step": (
            "Do not alter the architecture or clean-training configuration "
            "after this point based on injected results. Next validate complete "
            "clean-scene scoring, score orientation, overlap averaging, and "
            "exact top-2% score distribution before final 15-case evaluation."
        ),
    }

    summary_path = OUTPUT_DIR / "SSFAE_Clean_Training_Summary.json"

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 76)
    print("SSF-AE CLEAN TRAINING COMPLETE")
    print("=" * 76)
    print("CNN frozen train MSE :", cnn_loss)
    print("ViT frozen train MSE :", vit_loss)
    print("Frozen fusion weight CNN:", w_cnn)
    print("Frozen fusion weight ViT:", w_vit)
    print("Fusion params:", fusion_path)
    print("Summary:", summary_path)
    print("\nSTOP HERE. No injected dataset has been evaluated.")


if __name__ == "__main__":
    main()
