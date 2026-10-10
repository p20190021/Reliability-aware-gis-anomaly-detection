"""
SSF-AE FINAL ALL-15 INJECTED PREPROCESSING - NUMPY ONLY

Purpose
-------
Prepare the exact injected rows needed by the frozen SSF-AE model for the
three established injection ratios and five established scenarios.

This script:
- reuses the exact injected matrices already generated for the proposed method;
- reuses each ratio's authoritative inject_indices.npy;
- validates injection_ground_truth_labels.npy when present;
- validates a deterministic sample of non-injected rows against clean X.npy;
- applies ONLY the frozen SSF-AE preprocessing:
      clean feature z-standardization
      -> frozen PCA(21)
      -> frozen per-PC min-max affine transform
- applies NO clipping after the frozen min-max transform;
- transforms ONLY injected rows and saves them in inject_idx order;
- imports NO PyTorch, specifically to avoid the Windows OpenMP conflict.

It performs NO training, NO refitting, NO model selection, and NO tuning.

Expected injected source
------------------------
Code/03_Injected_Proposed_Results_Seed_42/
    Injected_Datasets_with_0.05%_Injection_Rate/
        inject_indices.npy
        injection_ground_truth_labels.npy   [if present]
        X_injected_extreme.npy
        X_injected_noise_strong.npy
        X_injected_noise_moderate.npy
        X_injected_noise_mild.npy
        X_injected_inconsistent.npy
    ... 0.1% ...
    ... 0.5% ...

Outputs
-------
Code/SSFAE/results/final_injected_preprocessing/
    ratio_0p0005/
    ratio_0p0010/
    ratio_0p0050/
        inject_indices.npy
        <scenario>/SSFAE_Injected_Rows_PCA21.npy
    SSFAE_Injected_Preprocessing_Summary.json

Run
---
    python 26a_ssfae_prepare_all15_injected_rows.py
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np


EXPECTED_N = 4_290_958
EXPECTED_D = 23
REAL_DEPTH = 21
NONINJECTED_EQUALITY_CHECK_N = 2048

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent

INJECTED_ROOT = ROOT_DIR / "03_Injected_Proposed_Results_Seed_42"
PROTOCOL_DIR = SCRIPT_DIR / "results" / "protocol_freeze"
OUTPUT_DIR = SCRIPT_DIR / "results" / "final_injected_preprocessing"

PREP_PATH = PROTOCOL_DIR / "SSFAE_Preprocessing_Params.npz"
NORM_PATH = PROTOCOL_DIR / "SSFAE_PostPCA_Normalization_Params.npz"

RATIO_CONFIG = {
    "0.05%": {
        "ratio": 0.0005,
        "folder": "Injected_Datasets_with_0.05%_Injection_Rate",
        "tag": "ratio_0p0005",
    },
    "0.1%": {
        "ratio": 0.0010,
        "folder": "Injected_Datasets_with_0.1%_Injection_Rate",
        "tag": "ratio_0p0010",
    },
    "0.5%": {
        "ratio": 0.0050,
        "folder": "Injected_Datasets_with_0.5%_Injection_Rate",
        "tag": "ratio_0p0050",
    },
}

SCENARIO_FILES = {
    "extreme": "X_injected_extreme.npy",
    "noise_strong": "X_injected_noise_strong.npy",
    "noise_moderate": "X_injected_noise_moderate.npy",
    "noise_mild": "X_injected_noise_mild.npy",
    "inconsistent": "X_injected_inconsistent.npy",
}

SCENARIO_DISPLAY = {
    "extreme": "Extreme",
    "noise_strong": "Strong noise",
    "noise_moderate": "Moderate noise",
    "noise_mild": "Mild noise",
    "inconsistent": "Cross-feature inconsistency",
}


def require_file(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Required file was not found:\n{path}")
    return path


def require_dir(path: Path):
    if not path.is_dir():
        raise FileNotFoundError(f"Required directory was not found:\n{path}")
    return path


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
    require_file(PREP_PATH)
    require_file(NORM_PATH)

    prep = np.load(PREP_PATH, allow_pickle=False)
    norm = np.load(NORM_PATH, allow_pickle=False)

    feature_mean = np.asarray(prep["feature_mean"], dtype=np.float64)
    feature_std = np.asarray(prep["feature_std"], dtype=np.float64)
    components = np.asarray(prep["pca_components"], dtype=np.float64)
    retained = int(np.asarray(prep["retained_components"]).item())

    pc_min = np.asarray(norm["pc_min"], dtype=np.float64)
    pc_range = np.asarray(norm["pc_range"], dtype=np.float64)

    if feature_mean.shape != (EXPECTED_D,):
        raise ValueError(f"Unexpected feature_mean shape: {feature_mean.shape}")
    if feature_std.shape != (EXPECTED_D,):
        raise ValueError(f"Unexpected feature_std shape: {feature_std.shape}")
    if retained != REAL_DEPTH:
        raise ValueError(f"Frozen retained PCA count is {retained}, expected {REAL_DEPTH}.")
    if components.shape != (REAL_DEPTH, EXPECTED_D):
        raise ValueError(f"Unexpected PCA components shape: {components.shape}")
    if pc_min.shape != (REAL_DEPTH,):
        raise ValueError(f"Unexpected pc_min shape: {pc_min.shape}")
    if pc_range.shape != (REAL_DEPTH,):
        raise ValueError(f"Unexpected pc_range shape: {pc_range.shape}")
    if np.any(feature_std <= 0):
        raise ValueError("Frozen feature_std contains non-positive values.")
    if np.any(pc_range <= 0):
        raise ValueError("Frozen pc_range contains non-positive values.")

    return feature_mean, feature_std, components, pc_min, pc_range


def load_injection_definition(ratio_dir: Path):
    idx_path = require_file(ratio_dir / "inject_indices.npy")
    inject_idx = np.load(idx_path, allow_pickle=False).astype(np.int64, copy=False)

    if inject_idx.ndim != 1:
        raise ValueError(f"inject_indices.npy must be 1-D: {idx_path}")
    if np.any(inject_idx < 0) or np.any(inject_idx >= EXPECTED_N):
        raise IndexError(f"Out-of-range injection indices: {idx_path}")
    if np.unique(inject_idx).size != inject_idx.size:
        raise ValueError(f"Duplicate injection indices: {idx_path}")

    labels_path = ratio_dir / "injection_ground_truth_labels.npy"
    labels_validated = False

    if labels_path.is_file():
        labels = np.load(labels_path, mmap_mode="r", allow_pickle=False)
        labels = np.asarray(labels).reshape(-1)

        if labels.size != EXPECTED_N:
            raise ValueError(f"Ground-truth label length mismatch: {labels_path}")

        gt_idx = np.flatnonzero(labels > 0)

        if not np.array_equal(np.sort(inject_idx), gt_idx):
            raise ValueError(
                f"inject_indices.npy does not match injection_ground_truth_labels.npy:\n"
                f"{ratio_dir}"
            )

        labels_validated = True

    return inject_idx, idx_path, labels_validated


def verify_noninjected_rows_unchanged(clean_raw, injected_raw, inject_idx):
    injected_mask = np.zeros(EXPECTED_N, dtype=bool)
    injected_mask[inject_idx] = True

    rng = np.random.default_rng(20260814)
    candidates = np.flatnonzero(~injected_mask)
    n_check = min(NONINJECTED_EQUALITY_CHECK_N, candidates.size)
    sampled = rng.choice(candidates, size=n_check, replace=False)

    a = np.asarray(clean_raw[sampled])
    b = np.asarray(injected_raw[sampled])

    exact_equal = bool(np.array_equal(a, b))
    max_abs = float(
        np.max(
            np.abs(
                a.astype(np.float64)
                - b.astype(np.float64)
            )
        )
    )

    if not exact_equal:
        raise RuntimeError(
            "A sampled non-injected row differs from X.npy. "
            "Do not continue until the injected dataset is verified."
        )

    return {
        "n_checked": int(n_check),
        "exact_equal": exact_equal,
        "max_abs_difference": max_abs,
    }


def transform_injected_rows(
    raw_rows,
    feature_mean,
    feature_std,
    components,
    pc_min,
    pc_range,
):
    x = np.asarray(raw_rows, dtype=np.float64)

    if x.ndim != 2 or x.shape[1] != EXPECTED_D:
        raise ValueError(f"Unexpected injected-row matrix shape: {x.shape}")

    if not np.isfinite(x).all():
        raise ValueError("Injected rows contain NaN or infinite values.")

    z = (x - feature_mean) / feature_std
    pcs = z @ components.T

    # Frozen clean per-PC min-max affine transform.
    # IMPORTANT: inference clipping remains OFF.
    normalized = (pcs - pc_min) / pc_range

    if not np.isfinite(normalized).all():
        raise ValueError("Transformed injected PCA rows contain non-finite values.")

    return normalized.astype(np.float32, copy=False)


def main():
    require_dir(INJECTED_ROOT)

    clean_path = require_file(ROOT_DIR / "X.npy")
    clean_raw = np.load(clean_path, mmap_mode="r", allow_pickle=False)

    if clean_raw.shape != (EXPECTED_N, EXPECTED_D):
        raise ValueError(f"Unexpected clean X shape: {clean_raw.shape}")

    (
        feature_mean,
        feature_std,
        components,
        pc_min,
        pc_range,
    ) = load_frozen_params()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("SSF-AE FINAL ALL-15 INJECTED PREPROCESSING")
    print("=" * 78)
    print("Clean X        :", clean_raw.shape)
    print("Injected root  :", INJECTED_ROOT)
    print("PCA dimensions :", REAL_DEPTH)
    print("Inference clip : OFF")
    print("PyTorch loaded : NO")
    print("Training/tuning: NO")

    summary_rows = []
    t_all = time.perf_counter()

    for ratio_label, cfg in RATIO_CONFIG.items():
        ratio_dir = require_dir(INJECTED_ROOT / cfg["folder"])
        inject_idx, idx_path, labels_validated = load_injection_definition(ratio_dir)

        ratio_out = OUTPUT_DIR / cfg["tag"]
        ratio_out.mkdir(parents=True, exist_ok=True)
        np.save(ratio_out / "inject_indices.npy", inject_idx)

        actual_ratio = inject_idx.size / EXPECTED_N

        print("\n" + "-" * 78)
        print(
            f"{ratio_label}: {inject_idx.size:,}/{EXPECTED_N:,} "
            f"({100.0 * actual_ratio:.6f}%)"
        )
        print("-" * 78)

        for scenario, filename in SCENARIO_FILES.items():
            scenario_path = require_file(ratio_dir / filename)
            injected_raw = np.load(
                scenario_path,
                mmap_mode="r",
                allow_pickle=False,
            )

            if injected_raw.shape != clean_raw.shape:
                raise ValueError(
                    f"Shape mismatch for {scenario_path}: "
                    f"{injected_raw.shape} vs {clean_raw.shape}"
                )

            equality_check = verify_noninjected_rows_unchanged(
                clean_raw,
                injected_raw,
                inject_idx,
            )

            t0 = time.perf_counter()

            # Materialize ONLY authoritative injected rows.
            raw_rows = np.asarray(injected_raw[inject_idx])

            transformed = transform_injected_rows(
                raw_rows,
                feature_mean,
                feature_std,
                components,
                pc_min,
                pc_range,
            )

            scenario_out = ratio_out / scenario
            scenario_out.mkdir(parents=True, exist_ok=True)

            out_path = scenario_out / "SSFAE_Injected_Rows_PCA21.npy"
            np.save(out_path, transformed)

            seconds = time.perf_counter() - t0

            row = {
                "ratio_label": ratio_label,
                "ratio_fraction_target": float(cfg["ratio"]),
                "ratio_fraction_actual": float(actual_ratio),
                "ratio_tag": cfg["tag"],
                "scenario": scenario,
                "scenario_display": SCENARIO_DISPLAY[scenario],
                "total_injected": int(inject_idx.size),
                "source_matrix": str(scenario_path),
                "source_indices": str(idx_path),
                "ground_truth_labels_validated": bool(labels_validated),
                "noninjected_equality_check": equality_check,
                "output": str(out_path),
                "output_shape": list(transformed.shape),
                "transformed_min": float(transformed.min()),
                "transformed_max": float(transformed.max()),
                "transformed_mean": float(transformed.mean()),
                "values_below_0": int(np.count_nonzero(transformed < 0.0)),
                "values_above_1": int(np.count_nonzero(transformed > 1.0)),
                "preprocessing_seconds": float(seconds),
            }

            summary_rows.append(row)

            print(
                f"{SCENARIO_DISPLAY[scenario]:28s} | "
                f"rows={inject_idx.size:,} | "
                f"range=[{row['transformed_min']:.6g}, "
                f"{row['transformed_max']:.6g}] | "
                f"{seconds:.2f}s"
            )

    summary = {
        "stage": "SSFAE_final_all15_injected_preprocessing",
        "status": "COMPLETE",
        "scientific_protocol": {
            "training_performed": False,
            "refitting_performed": False,
            "injected_matrices_regenerated": False,
            "injected_rows_only_transformed": True,
            "inference_clipping": False,
            "uses_frozen_clean_preprocessing_only": True,
            "no_pytorch_in_process": True,
        },
        "source_root": str(INJECTED_ROOT),
        "clean_X": str(clean_path),
        "frozen_artifact_hashes": {
            "preprocessing_params_sha256": sha256_file(PREP_PATH),
            "postpca_normalization_params_sha256": sha256_file(NORM_PATH),
        },
        "ratios": RATIO_CONFIG,
        "scenarios": SCENARIO_FILES,
        "cases": summary_rows,
        "total_seconds": float(time.perf_counter() - t_all),
        "next_step": (
            "Run 26b_ssfae_score_all15_injected_fullscene.py. "
            "It must full-rescore spatial windows rather than patch only injected "
            "pixel scores, because SSF-AE uses local/global spatial context."
        ),
    }

    summary_path = OUTPUT_DIR / "SSFAE_Injected_Preprocessing_Summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 78)
    print("SSF-AE INJECTED PREPROCESSING COMPLETE")
    print("=" * 78)
    print("Cases prepared :", len(summary_rows))
    print("Summary        :", summary_path)
    print("\nNext command:")
    print("  python 26b_ssfae_score_all15_injected_fullscene.py")


if __name__ == "__main__":
    main()
