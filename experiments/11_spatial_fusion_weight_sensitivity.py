"""
11_spatial_fusion_weight_sensitivity.py
=======================================

Purpose
-------
Evaluate multiple spatial-fusion weight settings for BOTH proposed variants
without retraining or rescoring any model.

For each weight pair (alpha, beta), alpha + beta = 1:

    Full:
        S_ref_raw = alpha * Full_base + beta * Full_spatial_normalized
        S_ref_norm = normalize_with_weight_specific_clean_bounds(S_ref_raw)
        S_final = S_ref_norm * saved_Full_reliability

    Pure:
        S_ref_raw = alpha * AE_normalized + beta * Pure_spatial_normalized
        S_ref_norm = normalize_with_weight_specific_clean_bounds(S_ref_raw)
        S_final = S_ref_norm * saved_Pure_reliability

The final anomaly-ranking score is S_final BEFORE the final output clipping/
normalization stage. Exact top-2% labels are generated from this score.

No model is trained, refitted, or rescored.

Existing verified project files are READ ONLY. All new results are written to:
    08_Weight_Sensitivity_Results_Seed_42

Storage policy
--------------
To keep the new experiment compact, each weight setting has ONLY ONE result
archive containing both Full and Pure results for:
    - the clean dataset
    - all 15 injected scenario/ratio combinations

Each archive is:
    Full_Pure_Anomaly_Results.npz

and contains only anomaly scores and exact-top-2% labels. Example keys:
    clean__full_score
    clean__full_labels
    clean__pure_score
    clean__pure_labels

    inj_0p05__extreme__full_score
    inj_0p05__extreme__full_labels
    inj_0p05__extreme__pure_score
    inj_0p05__extreme__pure_labels
    ...

A single global CSV is also written:
    Weight_Sensitivity_Summary.csv

The CSV contains clean calibration bounds, clean-mask overlap, injected
detection rates at all five budgets, and reference-0.75 validation results.

IMPORTANT
---------
The script validates alpha=0.75, beta=0.25 against the already verified
Full/Pure reference scores and top-2% masks. It also validates the injected
0.75/0.25 scores against the currently stored raw final scores when those
reference files are present.

If the reference validation fails, the script stops instead of silently
producing a sensitivity analysis from inconsistent source artifacts.
"""

from __future__ import annotations
import os

from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
import csv
import gc
import io
import math
import zipfile

import numpy as np


# ============================================================================
# 1. USER CONFIGURATION
# ============================================================================

ROOT_DIR = Path(os.environ.get("GIS_ANOMALY_PROJECT_ROOT", Path.cwd())).resolve()

CLEAN_DIR = ROOT_DIR / "03_Proposed_Results_Seed_42"
COMPARISON_DIR = ROOT_DIR / "04_Baseline_Comparison_Results_Seed_42"
INJECTED_ROOT = ROOT_DIR / "03_Injected_Proposed_Results_Seed_42"

# Completely separate output location. No existing result folder is modified.
OUTPUT_ROOT = ROOT_DIR / "08_Weight_Sensitivity_Results_Seed_42"

# Weight grid agreed for the spatial-fusion sensitivity analysis.
# alpha = base-score weight; beta = spatial-score weight.
# This is for the first run.
""" WEIGHT_PAIRS = [
    (1.00, 0.00),
    (0.90, 0.10),
    (0.80, 0.20),
    (0.75, 0.25),
    (0.70, 0.30),
    (0.60, 0.40),
    (0.50, 0.50),
] """

# This is for the second run
WEIGHT_PAIRS = [
    (0.40, 0.60),
    (0.30, 0.70),
    (0.20, 0.80),
    (0.10, 0.90),
    (0.00, 1.00),
]

REFERENCE_ALPHA = 0.75
REFERENCE_BETA = 0.25

TOP_LABEL_FRACTION = 0.02

# Same evaluation budgets already used in the paper.
EVALUATION_BUDGETS = [
    0.0010,   # 0.10%
    0.0025,   # 0.25%
    0.0050,   # 0.50%
    0.0100,   # 1.00%
    0.0200,   # 2.00%
]

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

INJECTION_RATIO_CONFIG = {
    "0.05%": {
        "ratio": 0.0005,
        "folder": "Injected_Datasets_with_0.05%_Injection_Rate",
        "key": "0p05",
    },
    "0.1%": {
        "ratio": 0.0010,
        "folder": "Injected_Datasets_with_0.1%_Injection_Rate",
        "key": "0p1",
    },
    "0.5%": {
        "ratio": 0.0050,
        "folder": "Injected_Datasets_with_0.5%_Injection_Rate",
        "key": "0p5",
    },
}

# Full base = AE + three feature-space + three embedding-space scores.
FULL_COMPONENT_FILES = [
    "ae_score_normalized.npy",
    "if_feature_score_normalized.npy",
    "lof_feature_score_normalized.npy",
    "md_feature_score_normalized.npy",
    "if_embedding_score_normalized.npy",
    "lof_embedding_score_normalized.npy",
    "md_embedding_score_normalized.npy",
]

# Files used directly by the two spatial branches.
FULL_SPATIAL_FILE = "spatial_score_normalized.npy"
PURE_SPATIAL_FILE = "pure_spatial_score_normalized.npy"

# Reference files used only for alpha=0.75 / beta=0.25 validation.
CLEAN_REFERENCE_SCORE_FILES = {
    "Full": COMPARISON_DIR / "Full_final_fused_score_preclip.npy",
    "Pure": COMPARISON_DIR / "Pure_final_fused_score_preclip.npy",
}

CLEAN_REFERENCE_LABEL_FILES = {
    "Full": COMPARISON_DIR / "Full_top2pct_anomaly_mask.npy",
    "Pure": COMPARISON_DIR / "Pure_top2pct_anomaly_mask.npy",
}

INJECTED_REFERENCE_SCORE_FILES = {
    "Full": "full_final_anomaly_score_raw.npy",
    "Pure": "pure_final_anomaly_score_raw.npy",
}

# Numerical tolerance for reproducing the already verified reference setting.
REFERENCE_SCORE_ATOL = 1e-5


# ============================================================================
# 2. SMALL UTILITIES
# ============================================================================

def log(message: str = "") -> None:
    print(message, flush=True)


def require_file(path: Path, description: str | None = None) -> Path:
    if not path.exists():
        label = description or path.name
        raise FileNotFoundError(f"Missing {label}:\n{path}")
    if not path.is_file():
        raise FileNotFoundError(f"Expected a file but found something else:\n{path}")
    return path


def load_vector(
    path: Path,
    *,
    expected_length: int | None = None,
    dtype=np.float32,
    mmap: bool = True,
) -> np.ndarray:
    require_file(path)
    arr = np.load(
        path,
        mmap_mode="r" if mmap else None,
        allow_pickle=False,
    )
    arr = np.asarray(arr).reshape(-1)

    if expected_length is not None and arr.size != expected_length:
        raise ValueError(
            f"{path.name}: found {arr.size:,} values; "
            f"expected {expected_length:,}."
        )

    if not np.isfinite(arr).all():
        raise ValueError(f"{path} contains NaN or infinite values.")

    if dtype is not None:
        arr = arr.astype(dtype, copy=False)

    return arr


def robust_bounds(values: np.ndarray) -> tuple[float, float]:
    p1, p99 = np.percentile(values, [1.0, 99.0])
    p1 = float(p1)
    p99 = float(p99)

    if not np.isfinite(p1) or not np.isfinite(p99):
        raise ValueError("Non-finite clean percentile bound.")
    if p99 <= p1:
        raise ValueError(f"Invalid clean bounds: p1={p1}, p99={p99}")

    return p1, p99


def normalize_with_bounds(
    values: np.ndarray,
    p1: float,
    p99: float,
) -> np.ndarray:
    if p99 <= p1:
        raise ValueError(f"Invalid bounds: {p1}, {p99}")

    out = (
        (np.clip(values, p1, p99) - p1)
        / (p99 - p1)
    )
    return out.astype(np.float32, copy=False)


def exact_top_k_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """
    Deterministic exact top-k selection:
      - all scores strictly above the boundary are selected;
      - ties at the boundary are resolved by lower original pixel index.
    """
    scores = np.asarray(scores).reshape(-1)

    if not np.isfinite(scores).all():
        raise ValueError("Scores contain NaN or Inf.")
    if not 1 <= k <= scores.size:
        raise ValueError(f"Invalid k={k} for n={scores.size}")

    boundary_position = scores.size - k
    boundary = np.partition(scores, boundary_position)[boundary_position]

    above = np.flatnonzero(scores > boundary)
    tied = np.flatnonzero(scores == boundary)

    need = k - above.size
    if need < 0:
        raise RuntimeError("Internal exact-top-k error.")
    if tied.size < need:
        raise RuntimeError("Insufficient boundary ties for exact top-k.")

    selected = np.concatenate([above, tied[:need]])

    if selected.size != k:
        raise RuntimeError(
            f"Exact top-k returned {selected.size:,}; expected {k:,}."
        )

    return selected.astype(np.int64, copy=False)


def exact_top_fraction_labels(
    scores: np.ndarray,
    fraction: float = TOP_LABEL_FRACTION,
) -> tuple[np.ndarray, np.ndarray]:
    k = int(math.ceil(float(fraction) * scores.size))
    selected = exact_top_k_indices(scores, k)

    labels = np.zeros(scores.size, dtype=np.uint8)
    labels[selected] = 1

    return labels, selected


def jaccard(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = np.asarray(mask_a).astype(bool, copy=False)
    b = np.asarray(mask_b).astype(bool, copy=False)

    if a.shape != b.shape:
        raise ValueError("Cannot compute Jaccard for different mask shapes.")

    intersection = int(np.logical_and(a, b).sum())
    union = int(np.logical_or(a, b).sum())

    return 1.0 if union == 0 else intersection / union


def max_abs_difference(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")

    # Chunked calculation avoids creating a second 4.29M-element float64 array.
    chunk = 500_000
    maximum = 0.0

    for start in range(0, a.size, chunk):
        stop = min(start + chunk, a.size)
        diff = np.max(
            np.abs(
                np.asarray(a[start:stop], dtype=np.float64)
                - np.asarray(b[start:stop], dtype=np.float64)
            )
        )
        maximum = max(maximum, float(diff))

    return maximum


def weight_tag(alpha: float, beta: float) -> str:
    return f"alpha_{alpha:.2f}_beta_{beta:.2f}"


def condition_key(
    ratio_key: str | None = None,
    scenario: str | None = None,
) -> str:
    if ratio_key is None:
        return "clean"

    if scenario is None:
        raise ValueError("Injected condition requires a scenario.")

    return f"inj_{ratio_key}__{scenario}"


# ============================================================================
# 3. INCREMENTAL NPZ WRITER
# ============================================================================

class IncrementalNPZWriter:
    """
    Write one .npz archive incrementally.

    This avoids holding every clean/injected score from a weight setting in
    memory at the same time. The resulting file is a normal NumPy .npz file
    and can be opened with np.load(...).
    """

    def __init__(self, final_path: Path):
        self.final_path = final_path
        self.temp_path = final_path.with_suffix(final_path.suffix + ".tmp")
        self.zip_file: zipfile.ZipFile | None = None
        self.keys: set[str] = set()

    def __enter__(self):
        self.final_path.parent.mkdir(parents=True, exist_ok=True)

        if self.temp_path.exists():
            self.temp_path.unlink()

        self.zip_file = zipfile.ZipFile(
            self.temp_path,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=1,
            allowZip64=True,
        )
        return self

    def write_array(self, key: str, array: np.ndarray) -> None:
        if self.zip_file is None:
            raise RuntimeError("NPZ writer is not open.")
        if key in self.keys:
            raise KeyError(f"Duplicate NPZ key: {key}")

        buffer = io.BytesIO()
        np.save(
            buffer,
            np.asarray(array),
            allow_pickle=False,
        )
        self.zip_file.writestr(
            f"{key}.npy",
            buffer.getvalue(),
        )
        self.keys.add(key)

    def __exit__(self, exc_type, exc, tb):
        if self.zip_file is not None:
            self.zip_file.close()
            self.zip_file = None

        if exc_type is None:
            if self.final_path.exists():
                self.final_path.unlink()
            self.temp_path.replace(self.final_path)
        else:
            if self.temp_path.exists():
                self.temp_path.unlink()

        return False


# ============================================================================
# 4. PROJECT-SPECIFIC INPUT LOADERS
# ============================================================================

def build_full_base(
    folder: Path,
    *,
    expected_length: int | None = None,
) -> tuple[np.ndarray, int]:
    """
    Reconstruct the Full seven-signal base exactly as in the verified clean
    comparison: equal average of the seven normalized component scores.
    """
    first = load_vector(
        folder / FULL_COMPONENT_FILES[0],
        expected_length=expected_length,
    )

    n = first.size
    base = np.asarray(first, dtype=np.float32).copy()

    for filename in FULL_COMPONENT_FILES[1:]:
        component = load_vector(
            folder / filename,
            expected_length=n,
        )
        base += component
        del component

    base /= float(len(FULL_COMPONENT_FILES))

    if not np.isfinite(base).all():
        raise ValueError(f"Full base contains non-finite values in {folder}")

    return base, n


def load_clean_reliability(
    expected_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    path = require_file(
        CLEAN_DIR / "full_pure_reliability_diagnostics.npz"
    )

    with np.load(path, allow_pickle=False) as z:
        required = ["full_reliability", "pure_reliability"]
        missing = [key for key in required if key not in z.files]
        if missing:
            raise KeyError(
                f"Missing clean reliability keys {missing} in {path}. "
                f"Available keys: {z.files}"
            )

        full = np.asarray(
            z["full_reliability"],
            dtype=np.float32,
        ).reshape(-1)

        pure = np.asarray(
            z["pure_reliability"],
            dtype=np.float32,
        ).reshape(-1)

    if full.size != expected_length or pure.size != expected_length:
        raise ValueError("Clean reliability length mismatch.")

    if not np.isfinite(full).all() or not np.isfinite(pure).all():
        raise ValueError("Clean reliability contains non-finite values.")

    return full, pure


def load_injected_reliability(
    scenario_dir: Path,
    expected_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    full_path = require_file(
        scenario_dir / "full_reliability_diagnostics.npz"
    )
    pure_path = require_file(
        scenario_dir / "pure_reliability_diagnostics.npz"
    )

    with np.load(full_path, allow_pickle=False) as z:
        if "reliability" not in z.files:
            raise KeyError(
                f"'reliability' missing in {full_path}; keys={z.files}"
            )
        full = np.asarray(
            z["reliability"],
            dtype=np.float32,
        ).reshape(-1)

    with np.load(pure_path, allow_pickle=False) as z:
        if "reliability" not in z.files:
            raise KeyError(
                f"'reliability' missing in {pure_path}; keys={z.files}"
            )
        pure = np.asarray(
            z["reliability"],
            dtype=np.float32,
        ).reshape(-1)

    if full.size != expected_length or pure.size != expected_length:
        raise ValueError(
            f"Injected reliability length mismatch in {scenario_dir}"
        )

    if not np.isfinite(full).all() or not np.isfinite(pure).all():
        raise ValueError(
            f"Injected reliability contains non-finite values in {scenario_dir}"
        )

    return full, pure


def find_scenario_dir(
    ratio_dir: Path,
    scenario: str,
) -> Path:
    """
    Locate exactly one scenario-result folder strictly inside the requested
    injection-ratio folder.

    A usable folder must contain all inputs needed by BOTH Full and Pure.
    """
    direct = ratio_dir / scenario
    candidates: list[Path] = []

    if direct.is_dir():
        candidates.append(direct)

    candidates.extend(
        p
        for p in ratio_dir.rglob(scenario)
        if p.is_dir() and p != direct
    )

    required_names = set(FULL_COMPONENT_FILES)
    required_names.update(
        {
            FULL_SPATIAL_FILE,
            PURE_SPATIAL_FILE,
            "full_reliability_diagnostics.npz",
            "pure_reliability_diagnostics.npz",
        }
    )

    usable = [
        p
        for p in candidates
        if all((p / name).exists() for name in required_names)
    ]

    usable = sorted(
        set(usable),
        key=lambda p: str(p).lower(),
    )

    if len(usable) != 1:
        raise FileNotFoundError(
            f"Expected exactly one usable folder for '{scenario}' under\n"
            f"{ratio_dir}\n"
            f"Found {len(usable)} usable folders:\n"
            + "\n".join(str(p) for p in usable)
        )

    return usable[0]


# ============================================================================
# 5. WEIGHT CALIBRATION
# ============================================================================

@dataclass(frozen=True)
class WeightCalibration:
    alpha: float
    beta: float
    full_p1: float
    full_p99: float
    pure_p1: float
    pure_p99: float


def compute_final_score(
    base_score: np.ndarray,
    spatial_score: np.ndarray,
    reliability: np.ndarray,
    *,
    alpha: float,
    beta: float,
    p1: float,
    p99: float,
) -> np.ndarray:
    refined_raw = (
        alpha * base_score
        + beta * spatial_score
    ).astype(
        np.float32,
        copy=False,
    )

    refined_norm = normalize_with_bounds(
        refined_raw,
        p1,
        p99,
    )

    final_score = (
        refined_norm * reliability
    ).astype(
        np.float32,
        copy=False,
    )

    if not np.isfinite(final_score).all():
        raise ValueError("Final score contains NaN or Inf.")

    return final_score


# ============================================================================
# 6. SUMMARY / EVALUATION HELPERS
# ============================================================================

def detection_result(
    score: np.ndarray,
    injected_indices: np.ndarray,
    budget: float,
) -> dict[str, float | int]:
    n = score.size
    k = int(math.ceil(budget * n))
    selected = exact_top_k_indices(score, k)

    injected_mask = np.zeros(n, dtype=np.uint8)
    injected_mask[injected_indices] = 1

    detected = int(injected_mask[selected].sum())
    total = int(injected_indices.size)

    return {
        "Budget_Fraction": float(budget),
        "Budget_Percent": float(100.0 * budget),
        "Selected_Count": int(k),
        "Injected_Count": total,
        "Detected_Injected": detected,
        "Detection_Rate_Percent": float(100.0 * detected / total),
    }


def write_summary_csv(
    path: Path,
    rows: list[dict[str, object]],
) -> None:
    if not rows:
        raise RuntimeError("No summary rows were generated.")

    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)


# ============================================================================
# 7. PREFLIGHT
# ============================================================================

def preflight() -> dict[tuple[str, str], Path]:
    log("=" * 80)
    log("PREFLIGHT")
    log("=" * 80)

    if not CLEAN_DIR.is_dir():
        raise FileNotFoundError(CLEAN_DIR)
    if not COMPARISON_DIR.is_dir():
        raise FileNotFoundError(COMPARISON_DIR)
    if not INJECTED_ROOT.is_dir():
        raise FileNotFoundError(INJECTED_ROOT)

    # Clean inputs.
    for filename in FULL_COMPONENT_FILES:
        require_file(CLEAN_DIR / filename)

    require_file(CLEAN_DIR / FULL_SPATIAL_FILE)
    require_file(CLEAN_DIR / PURE_SPATIAL_FILE)
    require_file(CLEAN_DIR / "full_pure_reliability_diagnostics.npz")

    # Verified clean reference outputs.
    for path in CLEAN_REFERENCE_SCORE_FILES.values():
        require_file(path)
    for path in CLEAN_REFERENCE_LABEL_FILES.values():
        require_file(path)

    # Injection-ratio specific indices + scenario folders.
    scenario_dirs: dict[tuple[str, str], Path] = {}

    for ratio_label, config in INJECTION_RATIO_CONFIG.items():
        ratio_dir = INJECTED_ROOT / str(config["folder"])

        if not ratio_dir.is_dir():
            raise FileNotFoundError(
                f"Missing injection-ratio folder:\n{ratio_dir}"
            )

        index_path = require_file(
            ratio_dir / "inject_indices.npy"
        )
        injected_indices = np.load(
            index_path,
            allow_pickle=False,
        ).astype(
            np.int64,
            copy=False,
        ).reshape(-1)

        if injected_indices.size == 0:
            raise ValueError(
                f"{ratio_label}: inject_indices.npy is empty."
            )

        if np.unique(injected_indices).size != injected_indices.size:
            raise ValueError(
                f"{ratio_label}: duplicate injected indices."
            )

        for scenario in SCENARIOS:
            folder = find_scenario_dir(
                ratio_dir,
                scenario,
            )
            scenario_dirs[(ratio_label, scenario)] = folder

        log(
            f"{ratio_label:>5s}: "
            f"{injected_indices.size:,} injected pixels | "
            f"5 scenario folders found"
        )

    if len(scenario_dirs) != 15:
        raise RuntimeError(
            f"Expected 15 injected scenario folders; "
            f"found {len(scenario_dirs)}."
        )

    log("\nClean inputs: OK")
    log("Injected inputs: 15/15 scenario folders OK")
    log("Existing source folders will be READ ONLY.")
    log(f"New output root:\n{OUTPUT_ROOT}")
    log("")

    return scenario_dirs


# ============================================================================
# 8. MAIN
# ============================================================================

def main() -> None:
    scenario_dirs = preflight()

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_rows: list[dict[str, object]] = []

    # ----------------------------------------------------------------------
    # A. Load clean source components ONCE.
    # ----------------------------------------------------------------------
    log("=" * 80)
    log("CLEAN WEIGHT CALIBRATION")
    log("=" * 80)

    full_base_clean, n_samples = build_full_base(
        CLEAN_DIR
    )

    pure_base_clean = load_vector(
        CLEAN_DIR / "ae_score_normalized.npy",
        expected_length=n_samples,
    )

    full_spatial_clean = load_vector(
        CLEAN_DIR / FULL_SPATIAL_FILE,
        expected_length=n_samples,
    )

    pure_spatial_clean = load_vector(
        CLEAN_DIR / PURE_SPATIAL_FILE,
        expected_length=n_samples,
    )

    (
        full_reliability_clean,
        pure_reliability_clean,
    ) = load_clean_reliability(
        n_samples
    )

    # Validate ratio-specific indices are in range and expected counts agree.
    for ratio_label, config in INJECTION_RATIO_CONFIG.items():
        ratio_dir = INJECTED_ROOT / str(config["folder"])
        injected_indices = np.load(
            ratio_dir / "inject_indices.npy",
            allow_pickle=False,
        ).astype(np.int64, copy=False).reshape(-1)

        if injected_indices.min() < 0 or injected_indices.max() >= n_samples:
            raise ValueError(
                f"{ratio_label}: injected index is outside [0, {n_samples - 1}]."
            )

        expected_injected = int(
            math.floor(float(config["ratio"]) * n_samples)
        )

        if injected_indices.size != expected_injected:
            raise ValueError(
                f"{ratio_label}: found {injected_indices.size:,} injected "
                f"indices; expected {expected_injected:,}."
            )

    # Reference clean files for strong validation.
    ref_full_score = load_vector(
        CLEAN_REFERENCE_SCORE_FILES["Full"],
        expected_length=n_samples,
    )
    ref_pure_score = load_vector(
        CLEAN_REFERENCE_SCORE_FILES["Pure"],
        expected_length=n_samples,
    )
    ref_full_labels = load_vector(
        CLEAN_REFERENCE_LABEL_FILES["Full"],
        expected_length=n_samples,
        dtype=np.uint8,
    )
    ref_pure_labels = load_vector(
        CLEAN_REFERENCE_LABEL_FILES["Pure"],
        expected_length=n_samples,
        dtype=np.uint8,
    )

    expected_top2 = int(
        math.ceil(TOP_LABEL_FRACTION * n_samples)
    )

    if int(ref_full_labels.sum()) != expected_top2:
        raise ValueError(
            "Existing Full reference mask does not contain exact top-2% count."
        )
    if int(ref_pure_labels.sum()) != expected_top2:
        raise ValueError(
            "Existing Pure reference mask does not contain exact top-2% count."
        )

    # ----------------------------------------------------------------------
    # B. Open one combined Full/Pure archive PER WEIGHT.
    # ----------------------------------------------------------------------
    calibrations: dict[tuple[float, float], WeightCalibration] = {}
    clean_labels_by_weight: dict[
        tuple[float, float],
        tuple[np.ndarray, np.ndarray],
    ] = {}

    writers: dict[tuple[float, float], IncrementalNPZWriter] = {}

    with ExitStack() as stack:
        for alpha, beta in WEIGHT_PAIRS:
            if not np.isclose(alpha + beta, 1.0, atol=1e-12):
                raise ValueError(
                    f"Weight pair does not sum to one: "
                    f"alpha={alpha}, beta={beta}"
                )

            out_dir = OUTPUT_ROOT / weight_tag(alpha, beta)
            archive_path = (
                out_dir
                / "Full_Pure_Anomaly_Results.npz"
            )

            writer = stack.enter_context(
                IncrementalNPZWriter(
                    archive_path
                )
            )
            writers[(alpha, beta)] = writer

        # ------------------------------------------------------------------
        # C. Clean calibration and clean result storage for every weight.
        # ------------------------------------------------------------------
        for alpha, beta in WEIGHT_PAIRS:
            full_refined_raw = (
                alpha * full_base_clean
                + beta * full_spatial_clean
            ).astype(
                np.float32,
                copy=False,
            )

            pure_refined_raw = (
                alpha * pure_base_clean
                + beta * pure_spatial_clean
            ).astype(
                np.float32,
                copy=False,
            )

            full_p1, full_p99 = robust_bounds(
                full_refined_raw
            )
            pure_p1, pure_p99 = robust_bounds(
                pure_refined_raw
            )

            calibration = WeightCalibration(
                alpha=alpha,
                beta=beta,
                full_p1=full_p1,
                full_p99=full_p99,
                pure_p1=pure_p1,
                pure_p99=pure_p99,
            )
            calibrations[(alpha, beta)] = calibration

            full_score = compute_final_score(
                full_base_clean,
                full_spatial_clean,
                full_reliability_clean,
                alpha=alpha,
                beta=beta,
                p1=full_p1,
                p99=full_p99,
            )

            pure_score = compute_final_score(
                pure_base_clean,
                pure_spatial_clean,
                pure_reliability_clean,
                alpha=alpha,
                beta=beta,
                p1=pure_p1,
                p99=pure_p99,
            )

            full_labels, _ = exact_top_fraction_labels(
                full_score
            )
            pure_labels, _ = exact_top_fraction_labels(
                pure_score
            )

            if int(full_labels.sum()) != expected_top2:
                raise RuntimeError(
                    f"Full clean exact top-2% count failed for "
                    f"{alpha}/{beta}"
                )
            if int(pure_labels.sum()) != expected_top2:
                raise RuntimeError(
                    f"Pure clean exact top-2% count failed for "
                    f"{alpha}/{beta}"
                )

            clean_labels_by_weight[
                (alpha, beta)
            ] = (
                np.array(full_labels, copy=True),
                np.array(pure_labels, copy=True),
            )

            writer = writers[(alpha, beta)]
            prefix = condition_key()

            writer.write_array(
                f"{prefix}__full_score",
                full_score,
            )
            writer.write_array(
                f"{prefix}__full_labels",
                full_labels,
            )
            writer.write_array(
                f"{prefix}__pure_score",
                pure_score,
            )
            writer.write_array(
                f"{prefix}__pure_labels",
                pure_labels,
            )

            # Validate the current 0.75/0.25 configuration exactly against
            # the previously verified clean results.
            full_ref_diff = ""
            pure_ref_diff = ""
            full_ref_mask_j = ""
            pure_ref_mask_j = ""

            if (
                np.isclose(alpha, REFERENCE_ALPHA)
                and np.isclose(beta, REFERENCE_BETA)
            ):
                full_diff = max_abs_difference(
                    full_score,
                    ref_full_score,
                )
                pure_diff = max_abs_difference(
                    pure_score,
                    ref_pure_score,
                )
                full_j = jaccard(
                    full_labels,
                    ref_full_labels,
                )
                pure_j = jaccard(
                    pure_labels,
                    ref_pure_labels,
                )

                if full_diff > REFERENCE_SCORE_ATOL:
                    raise RuntimeError(
                        "Clean Full reference validation failed at 0.75/0.25: "
                        f"max abs diff={full_diff:.9g}"
                    )
                if pure_diff > REFERENCE_SCORE_ATOL:
                    raise RuntimeError(
                        "Clean Pure reference validation failed at 0.75/0.25: "
                        f"max abs diff={pure_diff:.9g}"
                    )
                if not np.isclose(full_j, 1.0, atol=0.0):
                    raise RuntimeError(
                        "Clean Full 0.75/0.25 top-2% mask does not reproduce "
                        "the verified reference mask."
                    )
                if not np.isclose(pure_j, 1.0, atol=0.0):
                    raise RuntimeError(
                        "Clean Pure 0.75/0.25 top-2% mask does not reproduce "
                        "the verified reference mask."
                    )

                full_ref_diff = full_diff
                pure_ref_diff = pure_diff
                full_ref_mask_j = full_j
                pure_ref_mask_j = pure_j

                log(
                    "Reference clean validation 0.75/0.25: "
                    f"Full diff={full_diff:.3g}, Pure diff={pure_diff:.3g}, "
                    "both masks Jaccard=1.0"
                )

            full_pure_j = jaccard(
                full_labels,
                pure_labels,
            )

            for variant, score, labels in [
                ("Full", full_score, full_labels),
                ("Pure", pure_score, pure_labels),
            ]:
                summary_rows.append(
                    {
                        "Record_Type": "Clean",
                        "Alpha": alpha,
                        "Beta": beta,
                        "Weight_Tag": weight_tag(alpha, beta),
                        "Variant": variant,
                        "Injection_Ratio": "",
                        "Scenario": "",
                        "Budget_Fraction": TOP_LABEL_FRACTION,
                        "Budget_Percent": 100.0 * TOP_LABEL_FRACTION,
                        "Selected_Count": int(labels.sum()),
                        "Injected_Count": "",
                        "Detected_Injected": "",
                        "Detection_Rate_Percent": "",
                        "Clean_Fusion_P1": (
                            full_p1 if variant == "Full" else pure_p1
                        ),
                        "Clean_Fusion_P99": (
                            full_p99 if variant == "Full" else pure_p99
                        ),
                        "Mean_Final_Score": float(
                            np.mean(score, dtype=np.float64)
                        ),
                        "Median_Final_Score": float(
                            np.median(score)
                        ),
                        "Jaccard_vs_Reference_075": "",
                        "Full_Pure_Clean_Jaccard": full_pure_j,
                        "Reference_075_MaxAbsDiff": (
                            full_ref_diff
                            if variant == "Full"
                            else pure_ref_diff
                        ),
                        "Reference_075_Mask_Jaccard": (
                            full_ref_mask_j
                            if variant == "Full"
                            else pure_ref_mask_j
                        ),
                        "Result_Archive": str(
                            OUTPUT_ROOT
                            / weight_tag(alpha, beta)
                            / "Full_Pure_Anomaly_Results.npz"
                        ),
                        "Score_Key": (
                            f"{prefix}__{variant.lower()}_score"
                        ),
                        "Label_Key": (
                            f"{prefix}__{variant.lower()}_labels"
                        ),
                    }
                )

            del (
                full_refined_raw,
                pure_refined_raw,
                full_score,
                pure_score,
                full_labels,
                pure_labels,
            )
            gc.collect()

        # Clean Jaccard versus the current 0.75/0.25 configuration.
        """ reference_clean_full, reference_clean_pure = (
            clean_labels_by_weight[
                (REFERENCE_ALPHA, REFERENCE_BETA)
            ]
        ) """

        # The next two are for the second run
        reference_clean_full = np.asarray(
            ref_full_labels,
            dtype=np.uint8,
        )

        reference_clean_pure = np.asarray(
            ref_pure_labels,
            dtype=np.uint8,
        )

        clean_jaccard_lookup: dict[
            tuple[float, float, str],
            float,
        ] = {}

        for alpha, beta in WEIGHT_PAIRS:
            full_labels, pure_labels = clean_labels_by_weight[
                (alpha, beta)
            ]

            clean_jaccard_lookup[
                (alpha, beta, "Full")
            ] = jaccard(
                full_labels,
                reference_clean_full,
            )

            clean_jaccard_lookup[
                (alpha, beta, "Pure")
            ] = jaccard(
                pure_labels,
                reference_clean_pure,
            )

        for row in summary_rows:
            if row["Record_Type"] == "Clean":
                row["Jaccard_vs_Reference_075"] = clean_jaccard_lookup[
                    (
                        float(row["Alpha"]),
                        float(row["Beta"]),
                        str(row["Variant"]),
                    )
                ]

        # We no longer need all clean masks.
        del clean_labels_by_weight
        gc.collect()

        # ------------------------------------------------------------------
        # D. Process each injected scenario ONCE, then apply all weights.
        # ------------------------------------------------------------------
        log("")
        log("=" * 80)
        log("INJECTED WEIGHT SENSITIVITY")
        log("=" * 80)

        processed_conditions = 0

        for ratio_label, ratio_config in INJECTION_RATIO_CONFIG.items():
            ratio_dir = (
                INJECTED_ROOT
                / str(ratio_config["folder"])
            )

            injected_indices = np.load(
                ratio_dir / "inject_indices.npy",
                allow_pickle=False,
            ).astype(
                np.int64,
                copy=False,
            ).reshape(-1)

            ratio_key = str(
                ratio_config["key"]
            )

            for scenario in SCENARIOS:
                scenario_dir = scenario_dirs[
                    (ratio_label, scenario)
                ]

                log(
                    f"[{ratio_label:>5s}] "
                    f"{scenario:16s} | loading source artifacts"
                )

                full_base, scenario_n = build_full_base(
                    scenario_dir,
                    expected_length=n_samples,
                )

                if scenario_n != n_samples:
                    raise ValueError(
                        f"{scenario_dir}: sample count changed."
                    )

                pure_base = load_vector(
                    scenario_dir
                    / "ae_score_normalized.npy",
                    expected_length=n_samples,
                )

                full_spatial = load_vector(
                    scenario_dir
                    / FULL_SPATIAL_FILE,
                    expected_length=n_samples,
                )

                pure_spatial = load_vector(
                    scenario_dir
                    / PURE_SPATIAL_FILE,
                    expected_length=n_samples,
                )

                (
                    full_reliability,
                    pure_reliability,
                ) = load_injected_reliability(
                    scenario_dir,
                    n_samples,
                )

                for alpha, beta in WEIGHT_PAIRS:
                    calibration = calibrations[
                        (alpha, beta)
                    ]

                    full_score = compute_final_score(
                        full_base,
                        full_spatial,
                        full_reliability,
                        alpha=alpha,
                        beta=beta,
                        p1=calibration.full_p1,
                        p99=calibration.full_p99,
                    )

                    pure_score = compute_final_score(
                        pure_base,
                        pure_spatial,
                        pure_reliability,
                        alpha=alpha,
                        beta=beta,
                        p1=calibration.pure_p1,
                        p99=calibration.pure_p99,
                    )

                    full_labels, _ = exact_top_fraction_labels(
                        full_score
                    )
                    pure_labels, _ = exact_top_fraction_labels(
                        pure_score
                    )

                    if int(full_labels.sum()) != expected_top2:
                        raise RuntimeError(
                            "Injected Full exact top-2% count failed."
                        )
                    if int(pure_labels.sum()) != expected_top2:
                        raise RuntimeError(
                            "Injected Pure exact top-2% count failed."
                        )

                    prefix = condition_key(
                        ratio_key=ratio_key,
                        scenario=scenario,
                    )

                    writer = writers[
                        (alpha, beta)
                    ]

                    writer.write_array(
                        f"{prefix}__full_score",
                        full_score,
                    )
                    writer.write_array(
                        f"{prefix}__full_labels",
                        full_labels,
                    )
                    writer.write_array(
                        f"{prefix}__pure_score",
                        pure_score,
                    )
                    writer.write_array(
                        f"{prefix}__pure_labels",
                        pure_labels,
                    )

                    # Validate against the existing reference configuration.
                    reference_full_diff = ""
                    reference_pure_diff = ""

                    if (
                        np.isclose(alpha, REFERENCE_ALPHA)
                        and np.isclose(beta, REFERENCE_BETA)
                    ):
                        full_reference_path = (
                            scenario_dir
                            / INJECTED_REFERENCE_SCORE_FILES["Full"]
                        )
                        pure_reference_path = (
                            scenario_dir
                            / INJECTED_REFERENCE_SCORE_FILES["Pure"]
                        )

                        if full_reference_path.exists():
                            existing_full = load_vector(
                                full_reference_path,
                                expected_length=n_samples,
                            )
                            full_diff = max_abs_difference(
                                full_score,
                                existing_full,
                            )
                            reference_full_diff = full_diff

                            if full_diff > REFERENCE_SCORE_ATOL:
                                raise RuntimeError(
                                    f"{ratio_label}/{scenario}: "
                                    "Full injected 0.75/0.25 reference "
                                    f"validation failed; diff={full_diff:.9g}"
                                )

                            del existing_full

                        if pure_reference_path.exists():
                            existing_pure = load_vector(
                                pure_reference_path,
                                expected_length=n_samples,
                            )
                            pure_diff = max_abs_difference(
                                pure_score,
                                existing_pure,
                            )
                            reference_pure_diff = pure_diff

                            if pure_diff > REFERENCE_SCORE_ATOL:
                                raise RuntimeError(
                                    f"{ratio_label}/{scenario}: "
                                    "Pure injected 0.75/0.25 reference "
                                    f"validation failed; diff={pure_diff:.9g}"
                                )

                            del existing_pure

                    for variant, score in [
                        ("Full", full_score),
                        ("Pure", pure_score),
                    ]:
                        for budget in EVALUATION_BUDGETS:
                            result = detection_result(
                                score,
                                injected_indices,
                                budget,
                            )

                            summary_rows.append(
                                {
                                    "Record_Type": "Injected",
                                    "Alpha": alpha,
                                    "Beta": beta,
                                    "Weight_Tag": weight_tag(alpha, beta),
                                    "Variant": variant,
                                    "Injection_Ratio": ratio_label,
                                    "Scenario": SCENARIO_DISPLAY[scenario],
                                    "Scenario_Key": scenario,
                                    **result,
                                    "Clean_Fusion_P1": (
                                        calibration.full_p1
                                        if variant == "Full"
                                        else calibration.pure_p1
                                    ),
                                    "Clean_Fusion_P99": (
                                        calibration.full_p99
                                        if variant == "Full"
                                        else calibration.pure_p99
                                    ),
                                    "Mean_Final_Score": float(
                                        np.mean(
                                            score,
                                            dtype=np.float64,
                                        )
                                    ),
                                    "Median_Final_Score": "",
                                    "Jaccard_vs_Reference_075": "",
                                    "Full_Pure_Clean_Jaccard": "",
                                    "Reference_075_MaxAbsDiff": (
                                        reference_full_diff
                                        if variant == "Full"
                                        else reference_pure_diff
                                    ),
                                    "Reference_075_Mask_Jaccard": "",
                                    "Result_Archive": str(
                                        OUTPUT_ROOT
                                        / weight_tag(alpha, beta)
                                        / "Full_Pure_Anomaly_Results.npz"
                                    ),
                                    "Score_Key": (
                                        f"{prefix}__{variant.lower()}_score"
                                    ),
                                    "Label_Key": (
                                        f"{prefix}__{variant.lower()}_labels"
                                    ),
                                }
                            )

                    del (
                        full_score,
                        pure_score,
                        full_labels,
                        pure_labels,
                    )
                    gc.collect()

                del (
                    full_base,
                    pure_base,
                    full_spatial,
                    pure_spatial,
                    full_reliability,
                    pure_reliability,
                )
                gc.collect()

                processed_conditions += 1
                log(
                    f"[{ratio_label:>5s}] "
                    f"{scenario:16s} | all "
                    f"{len(WEIGHT_PAIRS)} weights completed"
                )

        if processed_conditions != 15:
            raise RuntimeError(
                f"Expected 15 injected conditions; "
                f"processed {processed_conditions}."
            )

        # Leaving ExitStack here closes every temporary archive and atomically
        # promotes it to the final .npz filename.

    # ----------------------------------------------------------------------
    # E. Write one global compact summary.
    # ----------------------------------------------------------------------
    # This is for the first run
    """ summary_path = (
        OUTPUT_ROOT
        / "Weight_Sensitivity_Summary.csv"
    ) """

    #This is for the second run
    summary_path = (
        OUTPUT_ROOT
        / "Weight_Sensitivity_Summary_Additional_Weights.csv"
    )

    write_summary_csv(
        summary_path,
        summary_rows,
    )

    # ----------------------------------------------------------------------
    # F. Final archive integrity check.
    # ----------------------------------------------------------------------
    log("")
    log("=" * 80)
    log("FINAL ARCHIVE CHECK")
    log("=" * 80)

    expected_condition_count = 1 + 15
    expected_keys_per_condition = 4
    expected_keys_per_archive = (
        expected_condition_count
        * expected_keys_per_condition
    )

    for alpha, beta in WEIGHT_PAIRS:
        archive_path = (
            OUTPUT_ROOT
            / weight_tag(alpha, beta)
            / "Full_Pure_Anomaly_Results.npz"
        )

        require_file(
            archive_path
        )

        with np.load(
            archive_path,
            allow_pickle=False,
        ) as z:
            if len(z.files) != expected_keys_per_archive:
                raise RuntimeError(
                    f"{archive_path}: found {len(z.files)} arrays; "
                    f"expected {expected_keys_per_archive}."
                )

            # Spot-check the clean arrays and one injected condition.
            for key in [
                "clean__full_score",
                "clean__full_labels",
                "clean__pure_score",
                "clean__pure_labels",
                "inj_0p1__extreme__full_score",
                "inj_0p1__extreme__pure_score",
            ]:
                if key not in z.files:
                    raise KeyError(
                        f"{archive_path}: missing key {key}"
                    )

            if z["clean__full_score"].size != n_samples:
                raise RuntimeError(
                    f"{archive_path}: clean Full score length mismatch."
                )
            if z["clean__pure_score"].size != n_samples:
                raise RuntimeError(
                    f"{archive_path}: clean Pure score length mismatch."
                )
            if int(z["clean__full_labels"].sum()) != expected_top2:
                raise RuntimeError(
                    f"{archive_path}: clean Full labels count mismatch."
                )
            if int(z["clean__pure_labels"].sum()) != expected_top2:
                raise RuntimeError(
                    f"{archive_path}: clean Pure labels count mismatch."
                )

        log(
            f"{weight_tag(alpha, beta)}: "
            f"{expected_keys_per_archive} arrays OK"
        )

    log("")
    log("=" * 80)
    log("WEIGHT SENSITIVITY COMPLETED")
    log("=" * 80)
    log(f"Dataset size: {n_samples:,}")
    log(f"Weight settings: {len(WEIGHT_PAIRS)}")
    log("Clean conditions per weight: 1")
    log("Injected conditions per weight: 15")
    log(
        "Each weight folder contains ONLY ONE combined Full/Pure NPZ archive."
    )
    log(
        "Each archive stores only final pre-clipping anomaly scores and "
        "exact top-2% labels."
    )
    log(f"Global summary:\n{summary_path}")
    log("")
    log("No model training, fitting, or detector rescoring was performed.")


if __name__ == "__main__":
    main()
