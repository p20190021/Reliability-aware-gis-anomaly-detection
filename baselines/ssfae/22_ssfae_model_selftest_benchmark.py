"""
SSF-AE PyTorch model self-test and runtime/memory benchmark.

This script does NOT load the 4.29M-pixel dataset and does NOT train a model.
It only uses synthetic 40x40x24 tensors to test the independent SSF-AE
reimplementation before a full run is attempted.

Place both files in:
    ...\Outlier Paper\Code\SSFAE\
        ssfae_core.py
        22_ssfae_model_selftest_benchmark.py

Outputs:
    ...\Code\SSFAE\results\model_benchmark\
        SSFAE_Model_Benchmark.csv
        SSFAE_Model_Benchmark.json

Suggested first server run:
    python 22_ssfae_model_selftest_benchmark.py --batch-sizes 1 2 4

If the GPU has comfortable headroom, then:
    python 22_ssfae_model_selftest_benchmark.py --batch-sizes 8 16 25

No final injected-test labels are used.
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ssfae_core import (
    SSFAEConfig,
    CNN3DAutoencoder,
    ViT3DAutoencoder,
    reconstruction_mse,
    per_pixel_reconstruction_error,
    dynamic_fusion_weights,
    fused_per_pixel_error,
    count_trainable_parameters,
)


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "results" / "model_benchmark"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        default=[1, 2, 4],
    )
    p.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )
    p.add_argument(
        "--warmup",
        type=int,
        default=1,
    )
    p.add_argument(
        "--repeats",
        type=int,
        default=2,
    )
    return p.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable.")
        return torch.device("cuda")
    return torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


def sync(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def reset_cuda_peak(device: torch.device):
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def peak_cuda_gb(device: torch.device):
    if device.type != "cuda":
        return None
    return torch.cuda.max_memory_allocated() / (1024 ** 3)


def make_input(batch_size: int, cfg: SSFAEConfig, device):
    # Clean normalized PCA channels in [0,1].
    x = torch.rand(
        batch_size,
        1,
        cfg.input_depth,
        cfg.height,
        cfg.width,
        device=device,
        dtype=torch.float32,
    )

    # Last three alignment channels are exact zero-padding.
    x[:, :, cfg.real_depth :, :, :] = 0.0
    return x


def self_test(device: torch.device, cfg: SSFAEConfig):
    print("\nRunning shape/math self-test...")

    cnn = CNN3DAutoencoder(cfg).to(device)
    vit = ViT3DAutoencoder(cfg).to(device)
    cnn.eval()
    vit.eval()

    x = make_input(1, cfg, device)

    with torch.no_grad():
        y_cnn = cnn(x)
        y_vit = vit(x)

    assert y_cnn.shape == x.shape, (y_cnn.shape, x.shape)
    assert y_vit.shape == x.shape, (y_vit.shape, x.shape)

    assert torch.isfinite(y_cnn).all()
    assert torch.isfinite(y_vit).all()

    assert float(y_cnn.min()) >= 0.0
    assert float(y_cnn.max()) <= 1.0
    assert float(y_vit.min()) >= 0.0
    assert float(y_vit.max()) <= 1.0

    cnn_loss = reconstruction_mse(
        x, y_cnn, cfg.real_depth
    )
    vit_loss = reconstruction_mse(
        x, y_vit, cfg.real_depth
    )

    cnn_pix = per_pixel_reconstruction_error(
        x, y_cnn, cfg.real_depth
    )
    vit_pix = per_pixel_reconstruction_error(
        x, y_vit, cfg.real_depth
    )

    assert cnn_pix.shape == (
        1, cfg.height, cfg.width
    )
    assert vit_pix.shape == (
        1, cfg.height, cfg.width
    )

    w_cnn, w_vit = dynamic_fusion_weights(
        cnn_loss, vit_loss
    )

    assert torch.allclose(
        w_cnn + w_vit,
        torch.tensor(
            1.0,
            device=(w_cnn + w_vit).device,
            dtype=(w_cnn + w_vit).dtype,
        ),
        atol=1e-6,
    )

    fused = fused_per_pixel_error(
        x,
        y_cnn,
        y_vit,
        cnn_loss,
        vit_loss,
        cfg.real_depth,
    )

    assert fused.shape == (
        1, cfg.height, cfg.width
    )
    assert torch.isfinite(fused).all()

    print("SELF-TEST: PASS")
    print("CNN output shape :", tuple(y_cnn.shape))
    print("ViT output shape :", tuple(y_vit.shape))
    print("CNN parameters   :", f"{count_trainable_parameters(cnn):,}")
    print("ViT parameters   :", f"{count_trainable_parameters(vit):,}")
    print("CNN test MSE     :", float(cnn_loss))
    print("ViT test MSE     :", float(vit_loss))
    print("Fusion weights   :", float(w_cnn), float(w_vit))

    del x, y_cnn, y_vit, cnn, vit
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def benchmark_branch(
    branch_name: str,
    model,
    batch_size: int,
    cfg: SSFAEConfig,
    device: torch.device,
    warmup: int,
    repeats: int,
):
    reset_cuda_peak(device)

    x = make_input(batch_size, cfg, device)
    model.train()

    # Use Adam only to exercise training-state allocation.
    optimizer = torch.optim.Adam(model.parameters())

    status = "ok"
    error_message = ""
    elapsed = []
    loss_value = None

    try:
        for _ in range(warmup):
            optimizer.zero_grad(set_to_none=True)
            y = model(x)
            loss = reconstruction_mse(
                x, y, cfg.real_depth
            )
            loss.backward()
            optimizer.step()
            sync(device)

        for _ in range(repeats):
            optimizer.zero_grad(set_to_none=True)

            sync(device)
            t0 = time.perf_counter()

            y = model(x)
            loss = reconstruction_mse(
                x, y, cfg.real_depth
            )
            loss.backward()
            optimizer.step()

            sync(device)
            elapsed.append(time.perf_counter() - t0)
            loss_value = float(loss.detach().cpu())

    except RuntimeError as e:
        if (
            "out of memory" in str(e).lower()
            or "cuda" in str(e).lower()
            and "memory" in str(e).lower()
        ):
            status = "oom"
            error_message = str(e)
        else:
            raise

    result = {
        "branch": branch_name,
        "batch_size": batch_size,
        "status": status,
        "mean_train_step_seconds": (
            float(np.mean(elapsed))
            if elapsed else None
        ),
        "min_train_step_seconds": (
            float(np.min(elapsed))
            if elapsed else None
        ),
        "last_synthetic_loss": loss_value,
        "peak_cuda_allocated_gb": peak_cuda_gb(device),
        "error": error_message,
    }

    del x, optimizer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return result


def main():
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)

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
    print("SSF-AE MODEL SELF-TEST / BENCHMARK")
    print("=" * 76)
    print("Python       :", platform.python_version())
    print("PyTorch      :", torch.__version__)
    print("Device       :", device)
    print("CUDA         :", torch.cuda.is_available())

    if device.type == "cuda":
        print("GPU          :", torch.cuda.get_device_name(0))
        props = torch.cuda.get_device_properties(0)
        print(
            "GPU memory   :",
            f"{props.total_memory / (1024**3):.2f} GB",
        )

    print("\nCandidate reimplementation configuration")
    print("-----------------------------------------")
    for k, v in cfg.__dict__.items():
        print(f"{k:22s}: {v}")

    self_test(device, cfg)

    rows = []

    for bs in args.batch_sizes:
        print(f"\nBenchmark batch size {bs}")

        cnn = CNN3DAutoencoder(cfg).to(device)
        r = benchmark_branch(
            "3D-CNN-AE",
            cnn,
            bs,
            cfg,
            device,
            args.warmup,
            args.repeats,
        )
        rows.append(r)
        print(
            " CNN:",
            r["status"],
            "time=",
            r["mean_train_step_seconds"],
            "peakGB=",
            r["peak_cuda_allocated_gb"],
        )
        del cnn
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        vit = ViT3DAutoencoder(cfg).to(device)
        r = benchmark_branch(
            "3D-ViT-AE",
            vit,
            bs,
            cfg,
            device,
            args.warmup,
            args.repeats,
        )
        rows.append(r)
        print(
            " ViT:",
            r["status"],
            "time=",
            r["mean_train_step_seconds"],
            "peakGB=",
            r["peak_cuda_allocated_gb"],
        )
        del vit
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    df = pd.DataFrame(rows)

    csv_path = OUTPUT_DIR / "SSFAE_Model_Benchmark.csv"
    df.to_csv(csv_path, index=False)

    report = {
        "status": "synthetic_selftest_benchmark_only",
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": (
            torch.cuda.get_device_name(0)
            if device.type == "cuda"
            else None
        ),
        "configuration": cfg.__dict__,
        "configuration_status": (
            "candidate reimplementation settings; benchmark before "
            "freezing full training configuration"
        ),
        "results": rows,
        "important_note": (
            "This benchmark does not train SSF-AE on project data and "
            "does not use injected labels. It tests shape correctness, "
            "backpropagation, runtime, and memory feasibility only."
        ),
    }

    json_path = OUTPUT_DIR / "SSFAE_Model_Benchmark.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\nSaved:")
    print(" ", csv_path)
    print(" ", json_path)
    print("\nNo project-data model training was performed.")


if __name__ == "__main__":
    main()
