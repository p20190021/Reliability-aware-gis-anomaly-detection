"""
13_reliability_parameter_sensitivity_VALIDATED.py

Reliability-parameter sensitivity for the predefined Full and Pure variants.

WHY THIS VERSION
----------------
This version deliberately does NOT reconstruct the spatially refined score
from earlier component formulas. Instead, it treats the CURRENT saved
pre-final-clipping Full/Pure score and CURRENT saved reliability diagnostics
as authoritative, derives the fixed refined score once, and then varies only:

    q      : internal component-indication fraction
             {1%, 2%, 5%}

    lambda : agreement weight in reliability
             {0.0, 0.2, 0.4, 0.6, 0.8, 1.0}

This isolates reliability sensitivity exactly and avoids mixing the experiment
with spatial-fusion recalibration or legacy clean-mask differences.

No training or inference is performed.
No existing project result is modified.

Authoritative sources
---------------------
Continuous component signals:
    03_Proposed_Results_Seed_42
    03_Injected_Proposed_Results_Seed_42

Clean current Full/Pure pre-final-clipping reference:
    04_Baseline_Comparison_Results_Seed_42

    This is intentional:
      - Full remains the original/main Full architecture.
      - Corrected Pure is synchronized into this same folder by
        08_recompute_corrected_pure_outputs_FINAL_075_025.py.
      - No Full-noIF folder is read anywhere in this analysis.

Injected current Full/Pure pre-final-clipping reference:
    each scenario folder in
    03_Injected_Proposed_Results_Seed_42

New output folder
-----------------
    13_Reliability_Parameter_Sensitivity_Results_Seed_42_CORRECTED_v2
"""

from __future__ import annotations
import os

import argparse
import gc
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from openpyxl import load_workbook
from openpyxl.styles import Font, Alignment
from openpyxl.utils import get_column_letter


# ============================================================================
# 1. Configuration
# ============================================================================

DEFAULT_ROOT = Path(os.environ.get("GIS_ANOMALY_PROJECT_ROOT", Path.cwd())).resolve()

MAIN_Q = 0.02
MAIN_LAMBDA = 0.60

Q_VALUES = [0.01, 0.02, 0.05]
LAMBDA_VALUES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

FINAL_BUDGETS = [
    0.0010,
    0.0025,
    0.0050,
    0.0100,
    0.0200,
]

RATIO_CONFIG = {
    "0.05%": {
        "ratio": 0.0005,
        "folder": "Injected_Datasets_with_0.05%_Injection_Rate",
    },
    "0.1%": {
        "ratio": 0.0010,
        "folder": "Injected_Datasets_with_0.1%_Injection_Rate",
    },
    "0.5%": {
        "ratio": 0.0050,
        "folder": "Injected_Datasets_with_0.5%_Injection_Rate",
    },
}

SCENARIOS = [
    "extreme",
    "inconsistent",
    "noise_mild",
    "noise_moderate",
    "noise_strong",
]

FULL_SIGNAL_FILES = [
    "ae_score_normalized.npy",
    "if_feature_score_normalized.npy",
    "lof_feature_score_normalized.npy",
    "md_feature_score_normalized.npy",
    "if_embedding_score_normalized.npy",
    "lof_embedding_score_normalized.npy",
    "md_embedding_score_normalized.npy",
    "spatial_score_normalized.npy",
    "context_score_normalized.npy",
]

PURE_SIGNAL_FILES = [
    "ae_score_normalized.npy",
    "pure_spatial_score_normalized.npy",
    "context_score_normalized.npy",
]

FULL_SIGNAL_COUNT = 9
PURE_SIGNAL_COUNT = 3


# ============================================================================
# 2. Utilities
# ============================================================================

def resolve_root(cli_root: str | None) -> Path:
    if cli_root:
        return Path(cli_root).expanduser().resolve()

    try:
        here = Path(__file__).resolve().parent
    except NameError:
        here = Path.cwd().resolve()

    if (
        (here / "03_Proposed_Results_Seed_42").is_dir()
        and (here / "03_Injected_Proposed_Results_Seed_42").is_dir()
    ):
        return here

    return DEFAULT_ROOT


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Required file was not found:\n{path}")
    return path


def require_dir(path: Path) -> Path:
    if not path.is_dir():
        raise FileNotFoundError(f"Required folder was not found:\n{path}")
    return path


def load_vector(
    path: Path,
    expected_length: int | None = None,
    name: str | None = None,
) -> np.ndarray:
    require_file(path)

    arr = np.load(
        path,
        mmap_mode="r",
        allow_pickle=False,
    )
    arr = np.asarray(arr).reshape(-1)

    display = name or path.name

    if expected_length is not None and arr.size != expected_length:
        raise ValueError(
            f"{display}: found {arr.size:,} values; "
            f"expected {expected_length:,}."
        )

    if not np.issubdtype(arr.dtype, np.number):
        raise TypeError(f"{display} is not numeric.")

    if not np.isfinite(arr).all():
        raise ValueError(f"{display} contains NaN or Inf.")

    return arr


def load_mask(
    path: Path,
    expected_length: int | None = None,
    name: str | None = None,
) -> np.ndarray:
    """
    Load a binary anomaly mask safely.

    Masks may legitimately be stored as:
      - bool
      - uint8 / int with values {0,1}
      - float with values {0.0,1.0}

    This is intentionally separate from load_vector(), which remains strict
    for continuous score arrays.
    """
    require_file(path)

    arr = np.load(
        path,
        mmap_mode="r",
        allow_pickle=False,
    )
    arr = np.asarray(arr).reshape(-1)

    display = name or path.name

    if expected_length is not None and arr.size != expected_length:
        raise ValueError(
            f"{display}: found {arr.size:,} values; "
            f"expected {expected_length:,}."
        )

    if np.issubdtype(arr.dtype, np.bool_):
        return arr.astype(
            bool,
            copy=False,
        )

    if not np.issubdtype(arr.dtype, np.number):
        raise TypeError(
            f"{display} must be a boolean or numeric binary mask; "
            f"found dtype {arr.dtype}."
        )

    if not np.isfinite(arr).all():
        raise ValueError(
            f"{display} contains NaN or Inf."
        )

    unique_values = np.unique(arr)

    if not np.all(
        np.isin(
            unique_values,
            [0, 1, 0.0, 1.0],
        )
    ):
        raise ValueError(
            f"{display} is not binary. "
            f"Observed values include: {unique_values[:10]}"
        )

    return arr.astype(
        bool,
        copy=False,
    )


def exact_top_k_order(scores: np.ndarray, k: int) -> np.ndarray:
    """
    Deterministic exact top-k:
      1) all scores strictly above the boundary;
      2) lower original indices first for exact boundary ties;
      3) returned indices ordered by descending score, then index.
    """
    scores = np.asarray(scores).reshape(-1)

    if not (1 <= k <= scores.size):
        raise ValueError(
            f"Invalid k={k} for {scores.size:,} samples."
        )

    boundary_position = scores.size - k
    boundary = np.partition(
        scores,
        boundary_position,
    )[boundary_position]

    above = np.flatnonzero(scores > boundary)
    tied = np.flatnonzero(scores == boundary)

    need = k - above.size

    if need < 0 or tied.size < need:
        raise RuntimeError("Exact top-k boundary calculation failed.")

    chosen = np.concatenate(
        [
            above,
            tied[:need],
        ]
    )

    if chosen.size != k:
        raise RuntimeError(
            f"Exact top-k produced {chosen.size:,}; expected {k:,}."
        )

    order = np.lexsort(
        (
            chosen,
            -scores[chosen],
        )
    )

    return chosen[order]


def exact_top_fraction_order(
    scores: np.ndarray,
    fraction: float,
) -> np.ndarray:
    k = int(
        math.ceil(
            float(fraction)
            * np.asarray(scores).size
        )
    )
    return exact_top_k_order(scores, k)


def load_signals(
    folder: Path,
    filenames: list[str],
    expected_length: int | None = None,
) -> tuple[list[np.ndarray], int]:
    signals = []
    n = expected_length

    for filename in filenames:
        arr = load_vector(
            folder / filename,
            expected_length=n,
            name=f"{folder.name}/{filename}",
        ).astype(
            np.float32,
            copy=False,
        )

        if n is None:
            n = arr.size

        signals.append(arr)

    return signals, int(n)


def agreement_for_q(
    signals: list[np.ndarray],
    q: float,
) -> np.ndarray:
    n = signals[0].size

    agreement = np.zeros(
        n,
        dtype=np.uint8,
    )

    k = int(
        math.ceil(
            q * n
        )
    )

    for signal in signals:
        selected = exact_top_k_order(
            signal,
            k,
        )
        agreement[selected] += 1
        del selected

    return agreement


def reliability_from_parts(
    agreement_count: np.ndarray,
    signal_count: int,
    uncertainty_norm: np.ndarray,
    lambda_value: float,
) -> np.ndarray:
    agreement_fraction = (
        agreement_count.astype(
            np.float32,
            copy=False,
        )
        / float(signal_count)
    )

    reliability = (
        float(lambda_value)
        * agreement_fraction
        +
        (
            1.0
            - float(lambda_value)
        )
        * (
            1.0
            - np.clip(
                uncertainty_norm,
                0.0,
                1.0,
            )
        )
    ).astype(
        np.float32,
        copy=False,
    )

    return np.clip(
        reliability,
        0.0,
        1.0,
    ).astype(
        np.float32,
        copy=False,
    )


def jaccard_indices(
    a: np.ndarray,
    b: np.ndarray,
) -> tuple[int, int, float]:
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)

    intersection = int(
        np.intersect1d(
            a,
            b,
            assume_unique=True,
        ).size
    )

    union = int(
        a.size
        + b.size
        - intersection
    )

    value = (
        intersection / union
        if union > 0
        else 1.0
    )

    return intersection, union, float(value)


def get_npz_key(
    data: np.lib.npyio.NpzFile,
    candidates: list[str],
    label: str,
) -> str:
    for key in candidates:
        if key in data.files:
            return key

    raise KeyError(
        f"Could not find {label}. "
        f"Tried {candidates}. "
        f"Available keys: {data.files}"
    )


def load_clean_diagnostics(
    clean_dir: Path,
    variant: str,
    expected_length: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Path]:
    path = require_file(
        clean_dir
        / "full_pure_reliability_diagnostics.npz"
    )

    with np.load(path, allow_pickle=False) as data:
        if variant == "Full":
            agreement_key = get_npz_key(
                data,
                [
                    "full_agreement",
                    "full_agreement_count",
                ],
                "clean Full agreement",
            )
            uncertainty_key = get_npz_key(
                data,
                [
                    "full_uncertainty",
                    "full_uncertainty_normalized",
                ],
                "clean Full normalized uncertainty",
            )
            reliability_key = get_npz_key(
                data,
                [
                    "full_reliability",
                ],
                "clean Full reliability",
            )
        elif variant == "Pure":
            agreement_key = get_npz_key(
                data,
                [
                    "pure_agreement",
                    "pure_agreement_count",
                ],
                "clean Pure agreement",
            )
            uncertainty_key = get_npz_key(
                data,
                [
                    "pure_uncertainty",
                    "pure_uncertainty_normalized",
                ],
                "clean Pure normalized uncertainty",
            )
            reliability_key = get_npz_key(
                data,
                [
                    "pure_reliability",
                ],
                "clean Pure reliability",
            )
        else:
            raise ValueError(variant)

        agreement = np.asarray(
            data[agreement_key]
        ).reshape(-1)

        uncertainty = np.asarray(
            data[uncertainty_key]
        ).reshape(-1)

        reliability = np.asarray(
            data[reliability_key]
        ).reshape(-1)

    for name, arr in [
        ("agreement", agreement),
        ("uncertainty", uncertainty),
        ("reliability", reliability),
    ]:
        if arr.size != expected_length:
            raise ValueError(
                f"{variant} clean {name}: "
                f"{arr.size:,} values; expected {expected_length:,}."
            )

    return (
        agreement.astype(np.uint8, copy=False),
        uncertainty.astype(np.float32, copy=False),
        reliability.astype(np.float32, copy=False),
        path,
    )


def load_injected_diagnostics(
    folder: Path,
    variant: str,
    expected_length: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Path]:
    if variant == "Full":
        candidates = [
            folder / "full_reliability_diagnostics.npz",
            folder / "full_pure_reliability_diagnostics.npz",
        ]
    elif variant == "Pure":
        candidates = [
            folder / "pure_reliability_diagnostics.npz",
            folder / "full_pure_reliability_diagnostics.npz",
        ]
    else:
        raise ValueError(variant)

    path = next(
        (
            p
            for p in candidates
            if p.is_file()
        ),
        None,
    )

    if path is None:
        raise FileNotFoundError(
            f"No injected {variant} reliability diagnostic file "
            f"was found in:\n{folder}"
        )

    with np.load(path, allow_pickle=False) as data:
        prefix = variant.lower()

        agreement_key = get_npz_key(
            data,
            [
                "agreement",
                "agreement_count",
                f"{prefix}_agreement",
                f"{prefix}_agreement_count",
            ],
            f"injected {variant} agreement",
        )

        uncertainty_key = get_npz_key(
            data,
            [
                "uncertainty_normalized",
                "uncertainty",
                f"{prefix}_uncertainty",
                f"{prefix}_uncertainty_normalized",
            ],
            f"injected {variant} normalized uncertainty",
        )

        reliability_key = get_npz_key(
            data,
            [
                "reliability",
                f"{prefix}_reliability",
            ],
            f"injected {variant} reliability",
        )

        agreement = np.asarray(
            data[agreement_key]
        ).reshape(-1)

        uncertainty = np.asarray(
            data[uncertainty_key]
        ).reshape(-1)

        reliability = np.asarray(
            data[reliability_key]
        ).reshape(-1)

    for name, arr in [
        ("agreement", agreement),
        ("uncertainty", uncertainty),
        ("reliability", reliability),
    ]:
        if arr.size != expected_length:
            raise ValueError(
                f"{variant} injected {name}: "
                f"{arr.size:,} values; expected {expected_length:,}."
            )

    return (
        agreement.astype(np.uint8, copy=False),
        uncertainty.astype(np.float32, copy=False),
        reliability.astype(np.float32, copy=False),
        path,
    )


def validate_saved_reliability(
    variant: str,
    saved_agreement: np.ndarray,
    signal_count: int,
    saved_uncertainty: np.ndarray,
    saved_reliability: np.ndarray,
) -> float:
    reconstructed = reliability_from_parts(
        agreement_count=saved_agreement,
        signal_count=signal_count,
        uncertainty_norm=saved_uncertainty,
        lambda_value=MAIN_LAMBDA,
    )

    max_abs = float(
        np.max(
            np.abs(
                reconstructed.astype(np.float64)
                -
                saved_reliability.astype(np.float64)
            )
        )
    )

    if max_abs > 2e-6:
        raise ValueError(
            f"{variant}: saved reliability is inconsistent with "
            f"q=2%, lambda=0.60 diagnostics. "
            f"Maximum absolute difference = {max_abs:.9g}"
        )

    return max_abs


def derive_fixed_refined_score(
    saved_final_raw: np.ndarray,
    saved_reliability: np.ndarray,
    variant: str,
) -> tuple[np.ndarray, float]:
    """
    Since:
        final_raw = refined_norm * reliability

    derive the fixed refined_norm directly from the CURRENT saved result.
    This isolates reliability sensitivity from spatial-fusion reconstruction.
    """
    saved_final_raw = np.asarray(
        saved_final_raw,
        dtype=np.float32,
    )
    saved_reliability = np.asarray(
        saved_reliability,
        dtype=np.float32,
    )

    tiny = 1e-12

    zero_reliability = (
        saved_reliability <= tiny
    )

    impossible = (
        zero_reliability
        & (
            np.abs(saved_final_raw)
            > 1e-8
        )
    )

    if impossible.any():
        raise ValueError(
            f"{variant}: non-zero saved final score occurs where "
            "saved reliability is zero."
        )

    refined = np.zeros_like(
        saved_final_raw,
        dtype=np.float32,
    )

    valid = ~zero_reliability

    refined[valid] = (
        saved_final_raw[valid]
        / saved_reliability[valid]
    )

    # Numerical tolerance only; refined score is theoretically in [0,1].
    if (
        float(refined.min()) < -1e-5
        or float(refined.max()) > 1.0 + 1e-4
    ):
        raise ValueError(
            f"{variant}: derived refined score lies outside the expected "
            f"[0,1] range: {float(refined.min())} to "
            f"{float(refined.max())}."
        )

    refined = np.clip(
        refined,
        0.0,
        1.0,
    ).astype(
        np.float32,
        copy=False,
    )

    reconstructed_main = (
        refined
        * saved_reliability
    ).astype(
        np.float32,
        copy=False,
    )

    max_abs = float(
        np.max(
            np.abs(
                reconstructed_main.astype(np.float64)
                -
                saved_final_raw.astype(np.float64)
            )
        )
    )

    if max_abs > 2e-6:
        raise ValueError(
            f"{variant}: fixed-refined-score reconstruction failed. "
            f"Maximum absolute difference = {max_abs:.9g}"
        )

    return refined, max_abs


def find_clean_reference_dir(root: Path) -> Path:
    """
    Return the ONLY clean comparison folder used by the current main paper.

    IMPORTANT:
    ---------
    Do NOT use any Full-noIF comparison folder here.

    The corrected Pure recomputation script explicitly synchronizes the
    corrected Pure clean score/mask/index into:
        04_Baseline_Comparison_Results_Seed_42

    It explicitly does NOT modify any Full score or Full mask. Therefore this
    folder contains the correct current pair:
        - original/main Full
        - corrected Pure
    """
    folder = (
        root
        / "04_Baseline_Comparison_Results_Seed_42"
    )

    required_names = [
        "Full_final_fused_score_preclip.npy",
        "Pure_final_fused_score_preclip.npy",
        "Full_top2pct_anomaly_mask.npy",
        "Pure_top2pct_anomaly_mask.npy",
    ]

    if not folder.is_dir():
        raise FileNotFoundError(
            "The current main clean comparison folder was not found:\n"
            f"{folder}"
        )

    missing = [
        name
        for name in required_names
        if not (folder / name).is_file()
    ]

    if missing:
        raise FileNotFoundError(
            "The current main clean comparison folder is incomplete:\n"
            f"{folder}\n\nMissing files:\n"
            + "\n".join(missing)
        )

    return folder


def format_workbook(path: Path) -> None:
    wb = load_workbook(path)

    for ws in wb.worksheets:
        ws.freeze_panes = "A2"
        ws.auto_filter.ref = ws.dimensions

        for cell in ws[1]:
            cell.font = Font(bold=True)
            cell.alignment = Alignment(
                horizontal="center",
                vertical="center",
                wrap_text=True,
            )

        for column in ws.columns:
            max_len = max(
                len(str(cell.value))
                if cell.value is not None
                else 0
                for cell in column
            )
            ws.column_dimensions[
                get_column_letter(
                    column[0].column
                )
            ].width = min(max_len + 2, 60)

    wb.save(path)


def heatmap(
    df: pd.DataFrame,
    title: str,
    colorbar_label: str,
    output: Path,
    fmt: str,
) -> None:
    values = df.to_numpy(dtype=float)

    fig, ax = plt.subplots(
        figsize=(8.0, 4.8)
    )

    im = ax.imshow(
        values,
        aspect="auto",
    )

    ax.set_xticks(
        np.arange(len(df.columns))
    )
    ax.set_xticklabels(
        [
            f"{float(v):.1f}"
            for v in df.columns
        ]
    )

    ax.set_yticks(
        np.arange(len(df.index))
    )
    ax.set_yticklabels(
        [
            f"{float(v):.0f}%"
            for v in df.index
        ]
    )

    ax.set_xlabel(
        "Agreement weight λ"
    )
    ax.set_ylabel(
        "Internal indication fraction q"
    )
    ax.set_title(title)

    cb = fig.colorbar(im, ax=ax)
    cb.set_label(colorbar_label)

    for r in range(values.shape[0]):
        for c in range(values.shape[1]):
            ax.text(
                c,
                r,
                format(values[r, c], fmt),
                ha="center",
                va="center",
            )

    fig.tight_layout()
    fig.savefig(
        output,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


# ============================================================================
# 3. Preflight
# ============================================================================

def preflight(
    root: Path,
    clean_source: Path,
    injected_root: Path,
    clean_reference: Path,
    output_dir: Path,
) -> pd.DataFrame:
    rows = []

    def record(
        item: str,
        path: Path,
        required: bool = True,
    ) -> None:
        exists = path.exists()

        rows.append(
            {
                "Item": item,
                "Path": str(path),
                "Required": required,
                "Exists": bool(exists),
            }
        )

        if required and not exists:
            raise FileNotFoundError(
                f"Preflight failed: {item}\n{path}"
            )

    record(
        "Clean proposed source",
        clean_source,
    )

    record(
        "Injected proposed source",
        injected_root,
    )

    record(
        "Clean current comparison reference",
        clean_reference,
    )

    # Main clean continuous signals:
    #   Full -> original/main Full signals from 03_Proposed_Results_Seed_42
    #   Pure -> corrected Pure uses pure_spatial_score_normalized.npy from the
    #           same folder, as generated from AE only by the Pure correction.
    for filename in (
        FULL_SIGNAL_FILES
        + [
            "pure_spatial_score_normalized.npy",
            "pure_final_anomaly_score_raw.npy",
            "pure_final_anomaly_labels.npy",
            "full_pure_reliability_diagnostics.npz",
        ]
    ):
        record(
            f"Clean source: {filename}",
            clean_source / filename,
        )

    for filename in [
        "Full_final_fused_score_preclip.npy",
        "Pure_final_fused_score_preclip.npy",
        "Full_top2pct_anomaly_mask.npy",
        "Pure_top2pct_anomaly_mask.npy",
    ]:
        record(
            f"Clean reference: {filename}",
            clean_reference / filename,
        )

    for ratio_label, config in RATIO_CONFIG.items():
        ratio_dir = (
            injected_root
            / config["folder"]
        )

        record(
            f"{ratio_label} ratio folder",
            ratio_dir,
        )

        record(
            f"{ratio_label} inject_indices.npy",
            ratio_dir / "inject_indices.npy",
        )

        for scenario in SCENARIOS:
            folder = ratio_dir / scenario

            record(
                f"{ratio_label}/{scenario}",
                folder,
            )

            for filename in (
                FULL_SIGNAL_FILES
                + [
                    "pure_spatial_score_normalized.npy",
                    "full_final_anomaly_score_raw.npy",
                    "pure_final_anomaly_score_raw.npy",
                    "full_reliability_diagnostics.npz",
                    "pure_reliability_diagnostics.npz",
                ]
            ):
                record(
                    f"{ratio_label}/{scenario}/{filename}",
                    folder / filename,
                )

    rows.append(
        {
            "Item": "New output folder",
            "Path": str(output_dir),
            "Required": False,
            "Exists": bool(output_dir.exists()),
        }
    )

    return pd.DataFrame(rows)


# ============================================================================
# 4. One variant / one dataset
# ============================================================================

def prepare_dataset_variant(
    folder: Path,
    variant: str,
    signal_files: list[str],
    signal_count: int,
    current_raw_path: Path,
    expected_length: int | None,
    clean: bool,
) -> dict:
    signals, n = load_signals(
        folder=folder,
        filenames=signal_files,
        expected_length=expected_length,
    )

    current_raw = load_vector(
        current_raw_path,
        expected_length=n,
        name=f"{variant} current pre-final score",
    ).astype(
        np.float32,
        copy=False,
    )

    if clean:
        (
            saved_agreement,
            saved_uncertainty,
            saved_reliability,
            diag_path,
        ) = load_clean_diagnostics(
            folder,
            variant,
            n,
        )
    else:
        (
            saved_agreement,
            saved_uncertainty,
            saved_reliability,
            diag_path,
        ) = load_injected_diagnostics(
            folder,
            variant,
            n,
        )

    reliability_diff = validate_saved_reliability(
        variant=variant,
        saved_agreement=saved_agreement,
        signal_count=signal_count,
        saved_uncertainty=saved_uncertainty,
        saved_reliability=saved_reliability,
    )

    fixed_refined, final_diff = derive_fixed_refined_score(
        saved_final_raw=current_raw,
        saved_reliability=saved_reliability,
        variant=variant,
    )

    # Recompute q=2 only as an audit. It is NOT used to replace the saved
    # q=2 agreement, because the saved agreement is the authoritative current
    # predefined configuration.
    recomputed_q2 = agreement_for_q(
        signals,
        MAIN_Q,
    )

    q2_mismatch_pixels = int(
        np.count_nonzero(
            recomputed_q2
            != saved_agreement
        )
    )

    agreements = {
        MAIN_Q: saved_agreement,
    }

    for q in Q_VALUES:
        if math.isclose(q, MAIN_Q):
            continue

        agreements[float(q)] = agreement_for_q(
            signals,
            float(q),
        )

    del recomputed_q2
    gc.collect()

    return {
        "n": n,
        "signals": signals,
        "current_raw": current_raw,
        "saved_agreement": saved_agreement,
        "saved_uncertainty": saved_uncertainty,
        "saved_reliability": saved_reliability,
        "fixed_refined": fixed_refined,
        "agreements": agreements,
        "diag_path": diag_path,
        "reliability_reconstruction_max_abs": reliability_diff,
        "final_reconstruction_max_abs": final_diff,
        "q2_recomputed_vs_saved_mismatch_pixels": q2_mismatch_pixels,
    }


def sensitivity_scores(
    prepared: dict,
    signal_count: int,
) -> dict[tuple[float, float], np.ndarray]:
    result = {}

    for q in Q_VALUES:
        agreement = prepared[
            "agreements"
        ][
            float(q)
        ]

        for lambda_value in LAMBDA_VALUES:
            # Preserve the exact existing main score for the predefined point.
            if (
                math.isclose(q, MAIN_Q)
                and math.isclose(
                    lambda_value,
                    MAIN_LAMBDA,
                )
            ):
                final_raw = prepared[
                    "current_raw"
                ].astype(
                    np.float32,
                    copy=False,
                )
            else:
                reliability = reliability_from_parts(
                    agreement_count=agreement,
                    signal_count=signal_count,
                    uncertainty_norm=prepared[
                        "saved_uncertainty"
                    ],
                    lambda_value=lambda_value,
                )

                final_raw = (
                    prepared[
                        "fixed_refined"
                    ]
                    * reliability
                ).astype(
                    np.float32,
                    copy=False,
                )

            result[
                (
                    float(q),
                    float(lambda_value),
                )
            ] = final_raw

    return result


# ============================================================================
# 5. Main
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help=(
            "Optional Outlier Paper/Code root. "
            "Normally unnecessary when this script is placed in Code."
        ),
    )

    parser.add_argument(
        "--preflight-only",
        action="store_true",
    )

    args = parser.parse_args()

    root = resolve_root(args.root)

    clean_source = (
        root
        / "03_Proposed_Results_Seed_42"
    )

    injected_root = (
        root
        / "03_Injected_Proposed_Results_Seed_42"
    )

    output_dir = (
        root
        / "13_Reliability_Parameter_Sensitivity_Results_Seed_42"
    )

    require_dir(root)
    require_dir(clean_source)
    require_dir(injected_root)

    clean_reference = find_clean_reference_dir(
        root
    )

    # Hard safety check: this analysis must NEVER use a Full-noIF folder.
    if "Full_Without_IF" in str(clean_reference) or "Full-noIF" in str(clean_reference):
        raise RuntimeError(
            "Incorrect clean reference selected: a Full-noIF folder "
            "must not be used for reliability sensitivity."
        )

    preflight_df = preflight(
        root=root,
        clean_source=clean_source,
        injected_root=injected_root,
        clean_reference=clean_reference,
        output_dir=output_dir,
    )

    print("=" * 80)
    print("RELIABILITY PARAMETER SENSITIVITY — CORRECT FULL + CORRECTED PURE (v2)")
    print("=" * 80)
    print("Project root:")
    print(root)
    print("\nClean continuous-signal source:")
    print(clean_source)
    print("\nClean current ranking reference:")
    print(clean_reference)
    print("\nInjected source:")
    print(injected_root)
    print("\nNew output folder:")
    print(output_dir)

    if args.preflight_only:
        print("\nPRECHECK PASSED.")
        print(
            preflight_df[
                [
                    "Item",
                    "Exists",
                ]
            ].to_string(
                index=False
            )
        )
        return

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    validation_rows = []
    source_rows = []
    clean_rows = []
    injected_rows = []

    # ------------------------------------------------------------------------
    # CLEAN
    # ------------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("CLEAN CURRENT-PIPELINE CALIBRATION")
    print("=" * 80)

    clean_config = [
        (
            "Full",
            FULL_SIGNAL_FILES,
            FULL_SIGNAL_COUNT,
            clean_reference
            / "Full_final_fused_score_preclip.npy",
            clean_reference
            / "Full_top2pct_anomaly_mask.npy",
        ),
        (
            "Pure",
            PURE_SIGNAL_FILES,
            PURE_SIGNAL_COUNT,
            clean_reference
            / "Pure_final_fused_score_preclip.npy",
            clean_reference
            / "Pure_top2pct_anomaly_mask.npy",
        ),
    ]

    # --------------------------------------------------------------
    # Corrected-Pure synchronization validation.
    #
    # 08_recompute_corrected_pure_outputs_FINAL_075_025.py writes the
    # corrected Pure raw score/mask into 03_Proposed_Results_Seed_42 and
    # also synchronizes the same Pure score/mask into
    # 04_Baseline_Comparison_Results_Seed_42.
    #
    # These two copies must therefore agree before sensitivity analysis.
    # --------------------------------------------------------------
    pure_raw_03 = load_vector(
        clean_source
        / "pure_final_anomaly_score_raw.npy",
        name="Corrected Pure raw score in 03_Proposed_Results_Seed_42",
    ).astype(
        np.float32,
        copy=False,
    )

    pure_raw_04 = load_vector(
        clean_reference
        / "Pure_final_fused_score_preclip.npy",
        expected_length=pure_raw_03.size,
        name="Corrected Pure raw score in 04_Baseline_Comparison_Results_Seed_42",
    ).astype(
        np.float32,
        copy=False,
    )

    pure_score_sync_max_abs = float(
        np.max(
            np.abs(
                pure_raw_03.astype(np.float64)
                -
                pure_raw_04.astype(np.float64)
            )
        )
    )

    if pure_score_sync_max_abs > 2e-6:
        raise ValueError(
            "Corrected Pure clean score is not synchronized between "
            "03_Proposed_Results_Seed_42 and "
            "04_Baseline_Comparison_Results_Seed_42. "
            f"Maximum absolute difference = {pure_score_sync_max_abs:.9g}"
        )

    pure_mask_03 = load_mask(
        clean_source
        / "pure_final_anomaly_labels.npy",
        expected_length=pure_raw_03.size,
        name="Corrected Pure clean mask in 03_Proposed_Results_Seed_42",
    )

    pure_mask_04 = load_mask(
        clean_reference
        / "Pure_top2pct_anomaly_mask.npy",
        expected_length=pure_raw_03.size,
        name="Corrected Pure clean mask in 04_Baseline_Comparison_Results_Seed_42",
    )

    if not np.array_equal(
        pure_mask_03,
        pure_mask_04,
    ):
        raise ValueError(
            "Corrected Pure clean mask is not synchronized between "
            "03_Proposed_Results_Seed_42 and "
            "04_Baseline_Comparison_Results_Seed_42."
        )

    print(
        "Corrected Pure synchronization check passed: "
        "03_Proposed_Results_Seed_42 == "
        "04_Baseline_Comparison_Results_Seed_42."
    )

    del pure_raw_03
    del pure_raw_04
    del pure_mask_03
    del pure_mask_04
    gc.collect()

    n_samples = None

    clean_prepared = {}

    for (
        variant,
        signal_files,
        signal_count,
        current_raw_path,
        current_mask_path,
    ) in clean_config:

        prepared = prepare_dataset_variant(
            folder=clean_source,
            variant=variant,
            signal_files=signal_files,
            signal_count=signal_count,
            current_raw_path=current_raw_path,
            expected_length=n_samples,
            clean=True,
        )

        if n_samples is None:
            n_samples = prepared["n"]

        current_mask = load_mask(
            current_mask_path,
            expected_length=n_samples,
            name=f"{variant} current clean mask",
        )

        current_indices = np.flatnonzero(
            current_mask
        )

        expected_clean_selected = int(
            math.ceil(
                0.02 * n_samples
            )
        )

        if current_indices.size != expected_clean_selected:
            raise ValueError(
                f"{variant} current clean mask selects "
                f"{current_indices.size:,} pixels; "
                f"expected exactly {expected_clean_selected:,} "
                "for the top-2% operating point."
            )

        current_raw_indices = exact_top_fraction_order(
            prepared[
                "current_raw"
            ],
            0.02,
        )

        mask_match = bool(
            np.array_equal(
                np.sort(current_indices),
                np.sort(current_raw_indices),
            )
        )

        validation_rows.append(
            {
                "Scope": "Clean",
                "Variant": variant,
                "Check": (
                    "Current pre-final score exact top-2% "
                    "matches current clean comparison mask"
                ),
                "Passed": mask_match,
                "Value": int(
                    np.intersect1d(
                        current_indices,
                        current_raw_indices,
                        assume_unique=True,
                    ).size
                ),
                "Expected": int(
                    current_indices.size
                ),
            }
        )

        validation_rows.append(
            {
                "Scope": "Clean",
                "Variant": variant,
                "Check": (
                    "Saved reliability reconstructed from "
                    "saved q=2% agreement and uncertainty"
                ),
                "Passed": bool(
                    prepared[
                        "reliability_reconstruction_max_abs"
                    ]
                    <= 2e-6
                ),
                "Value": prepared[
                    "reliability_reconstruction_max_abs"
                ],
                "Expected": "<= 2e-6",
            }
        )

        validation_rows.append(
            {
                "Scope": "Clean",
                "Variant": variant,
                "Check": (
                    "Current raw final score reconstructed from "
                    "fixed refined score x saved reliability"
                ),
                "Passed": bool(
                    prepared[
                        "final_reconstruction_max_abs"
                    ]
                    <= 2e-6
                ),
                "Value": prepared[
                    "final_reconstruction_max_abs"
                ],
                "Expected": "<= 2e-6",
            }
        )

        source_rows.append(
            {
                "Scope": "Clean",
                "Variant": variant,
                "Current_Raw_Score": str(
                    current_raw_path
                ),
                "Current_Mask": str(
                    current_mask_path
                ),
                "Diagnostics": str(
                    prepared[
                        "diag_path"
                    ]
                ),
                "q2_Recomputed_vs_Saved_Agreement_Mismatch_Pixels": (
                    prepared[
                        "q2_recomputed_vs_saved_mismatch_pixels"
                    ]
                ),
            }
        )

        score_grid = sensitivity_scores(
            prepared,
            signal_count,
        )

        for (
            q,
            lambda_value,
        ), final_raw in score_grid.items():

            selected = exact_top_fraction_order(
                final_raw,
                0.02,
            )

            intersection, union, jac = (
                jaccard_indices(
                    selected,
                    current_raw_indices,
                )
            )

            clean_rows.append(
                {
                    "Variant": variant,
                    "Agreement_Fraction_q": q,
                    "Agreement_Percent_q": q * 100.0,
                    "Lambda_Agreement_Weight": lambda_value,
                    "Certainty_Weight": 1.0 - lambda_value,
                    "Selected_Count": int(selected.size),
                    "Intersection_with_Main": intersection,
                    "Union_with_Main": union,
                    "Jaccard_with_Main": jac,
                    "Replacement_vs_Main_Percent": (
                        100.0
                        * (
                            selected.size
                            - intersection
                        )
                        / selected.size
                    ),
                }
            )

        clean_prepared[
            variant
        ] = prepared

        del score_grid
        del current_mask
        del current_indices
        del current_raw_indices
        gc.collect()

    clean_validation_df = pd.DataFrame(
        validation_rows
    )

    if not bool(
        clean_validation_df[
            "Passed"
        ].all()
    ):
        clean_validation_df.to_csv(
            output_dir
            / "VALIDATION_FAILED_CLEAN.csv",
            index=False,
        )

        print("\nCLEAN VALIDATION DETAILS:")
        print(
            clean_validation_df.to_string(
                index=False
            )
        )

        raise ValueError(
            "Clean validation failed. "
            "See VALIDATION_FAILED_CLEAN.csv."
        )

    print(
        "Clean current Full/Pure pipeline validation passed."
    )

    # Release clean continuous signals; only clean metadata/results remain.
    for variant in list(
        clean_prepared.keys()
    ):
        clean_prepared[
            variant
        ][
            "signals"
        ] = None

    gc.collect()

    # ------------------------------------------------------------------------
    # INJECTED
    # ------------------------------------------------------------------------

    for ratio_label, config in RATIO_CONFIG.items():
        ratio_dir = (
            injected_root
            / config[
                "folder"
            ]
        )

        injected_indices = np.load(
            ratio_dir
            / "inject_indices.npy",
            allow_pickle=False,
        ).astype(
            np.int64,
            copy=False,
        ).reshape(-1)

        expected_injected = int(
            math.floor(
                config[
                    "ratio"
                ]
                * n_samples
            )
        )

        if injected_indices.size != expected_injected:
            raise ValueError(
                f"{ratio_label}: found {injected_indices.size:,} "
                f"injected indices; expected {expected_injected:,}."
            )

        print("\n" + "=" * 80)
        print(
            f"INJECTION RATIO {ratio_label}"
        )
        print("=" * 80)

        for scenario in SCENARIOS:
            folder = (
                ratio_dir
                / scenario
            )

            print(
                f"Processing {ratio_label} / {scenario}"
            )

            for (
                variant,
                signal_files,
                signal_count,
                raw_filename,
            ) in [
                (
                    "Full",
                    FULL_SIGNAL_FILES,
                    FULL_SIGNAL_COUNT,
                    "full_final_anomaly_score_raw.npy",
                ),
                (
                    "Pure",
                    PURE_SIGNAL_FILES,
                    PURE_SIGNAL_COUNT,
                    "pure_final_anomaly_score_raw.npy",
                ),
            ]:
                prepared = prepare_dataset_variant(
                    folder=folder,
                    variant=variant,
                    signal_files=signal_files,
                    signal_count=signal_count,
                    current_raw_path=(
                        folder
                        / raw_filename
                    ),
                    expected_length=n_samples,
                    clean=False,
                )

                validation_rows.append(
                    {
                        "Scope": (
                            f"{ratio_label}/{scenario}"
                        ),
                        "Variant": variant,
                        "Check": (
                            "Saved reliability reconstructed from "
                            "saved q=2% agreement and uncertainty"
                        ),
                        "Passed": bool(
                            prepared[
                                "reliability_reconstruction_max_abs"
                            ]
                            <= 2e-6
                        ),
                        "Value": prepared[
                            "reliability_reconstruction_max_abs"
                        ],
                        "Expected": "<= 2e-6",
                    }
                )

                validation_rows.append(
                    {
                        "Scope": (
                            f"{ratio_label}/{scenario}"
                        ),
                        "Variant": variant,
                        "Check": (
                            "Current raw final score reconstructed from "
                            "fixed refined score x saved reliability"
                        ),
                        "Passed": bool(
                            prepared[
                                "final_reconstruction_max_abs"
                            ]
                            <= 2e-6
                        ),
                        "Value": prepared[
                            "final_reconstruction_max_abs"
                        ],
                        "Expected": "<= 2e-6",
                    }
                )

                source_rows.append(
                    {
                        "Scope": (
                            f"{ratio_label}/{scenario}"
                        ),
                        "Variant": variant,
                        "Current_Raw_Score": str(
                            folder
                            / raw_filename
                        ),
                        "Current_Mask": "",
                        "Diagnostics": str(
                            prepared[
                                "diag_path"
                            ]
                        ),
                        "q2_Recomputed_vs_Saved_Agreement_Mismatch_Pixels": (
                            prepared[
                                "q2_recomputed_vs_saved_mismatch_pixels"
                            ]
                        ),
                    }
                )

                score_grid = sensitivity_scores(
                    prepared,
                    signal_count,
                )

                max_budget = max(
                    FINAL_BUDGETS
                )

                max_k = int(
                    math.ceil(
                        max_budget
                        * n_samples
                    )
                )

                for (
                    q,
                    lambda_value,
                ), final_raw in score_grid.items():

                    ranking = exact_top_k_order(
                        final_raw,
                        max_k,
                    )

                    for budget in FINAL_BUDGETS:
                        k = int(
                            math.ceil(
                                budget
                                * n_samples
                            )
                        )

                        selected = ranking[
                            :k
                        ]

                        detected = int(
                            np.isin(
                                injected_indices,
                                selected,
                                assume_unique=True,
                            ).sum()
                        )

                        total = int(
                            injected_indices.size
                        )

                        injected_rows.append(
                            {
                                "Injection_Ratio": ratio_label,
                                "Scenario": scenario,
                                "Variant": variant,
                                "Agreement_Fraction_q": q,
                                "Agreement_Percent_q": q * 100.0,
                                "Lambda_Agreement_Weight": lambda_value,
                                "Certainty_Weight": 1.0 - lambda_value,
                                "Budget_Fraction": budget,
                                "Budget_Percent": budget * 100.0,
                                "Selected_Count": k,
                                "Total_Injected": total,
                                "Detected_Injected": detected,
                                "Detection_Rate_Percent": (
                                    100.0
                                    * detected
                                    / total
                                ),
                                "Raw_Final_Score_Threshold": float(
                                    final_raw[
                                        selected[-1]
                                    ]
                                ),
                            }
                        )

                    del ranking

                del score_grid
                del prepared
                gc.collect()

    validation_df = pd.DataFrame(
        validation_rows
    )

    if not bool(
        validation_df[
            "Passed"
        ].all()
    ):
        validation_df.to_csv(
            output_dir
            / "VALIDATION_FAILED_ALL.csv",
            index=False,
        )

        print("\nVALIDATION FAILURES:")
        print(
            validation_df[
                ~validation_df[
                    "Passed"
                ]
            ].to_string(
                index=False
            )
        )

        raise ValueError(
            "Injected/current-pipeline validation failed. "
            "See VALIDATION_FAILED_ALL.csv."
        )

    # ------------------------------------------------------------------------
    # Aggregate
    # ------------------------------------------------------------------------

    clean_df = pd.DataFrame(
        clean_rows
    )

    injected_df = pd.DataFrame(
        injected_rows
    )

    source_df = pd.DataFrame(
        source_rows
    )

    all15_df = (
        injected_df.groupby(
            [
                "Variant",
                "Agreement_Fraction_q",
                "Agreement_Percent_q",
                "Lambda_Agreement_Weight",
                "Certainty_Weight",
                "Budget_Fraction",
                "Budget_Percent",
            ],
            as_index=False,
        )
        .agg(
            Mean_Detection_Rate_Percent=(
                "Detection_Rate_Percent",
                "mean",
            ),
            Standard_Deviation=(
                "Detection_Rate_Percent",
                "std",
            ),
            Minimum_Detection_Rate_Percent=(
                "Detection_Rate_Percent",
                "min",
            ),
            Maximum_Detection_Rate_Percent=(
                "Detection_Rate_Percent",
                "max",
            ),
        )
    )

    reference_ratio_df = (
        injected_df[
            injected_df[
                "Injection_Ratio"
            ]
            == "0.1%"
        ]
        .groupby(
            [
                "Variant",
                "Scenario",
                "Agreement_Fraction_q",
                "Agreement_Percent_q",
                "Lambda_Agreement_Weight",
                "Certainty_Weight",
                "Budget_Fraction",
                "Budget_Percent",
            ],
            as_index=False,
        )
        .agg(
            Detection_Rate_Percent=(
                "Detection_Rate_Percent",
                "first",
            )
        )
    )

    top2_all15 = all15_df[
        np.isclose(
            all15_df[
                "Budget_Fraction"
            ],
            0.02,
        )
    ].copy()

    scenario_top2 = (
        injected_df[
            np.isclose(
                injected_df[
                    "Budget_Fraction"
                ],
                0.02,
            )
        ]
        .groupby(
            [
                "Variant",
                "Scenario",
                "Agreement_Fraction_q",
                "Agreement_Percent_q",
                "Lambda_Agreement_Weight",
                "Certainty_Weight",
            ],
            as_index=False,
        )
        .agg(
            Mean_Detection_Rate_Percent=(
                "Detection_Rate_Percent",
                "mean",
            )
        )
    )

    matrices = {}

    for variant in [
        "Full",
        "Pure",
    ]:
        top2_matrix = (
            top2_all15[
                top2_all15[
                    "Variant"
                ]
                == variant
            ]
            .pivot(
                index="Agreement_Percent_q",
                columns="Lambda_Agreement_Weight",
                values="Mean_Detection_Rate_Percent",
            )
            .sort_index()
        )

        clean_matrix = (
            clean_df[
                clean_df[
                    "Variant"
                ]
                == variant
            ]
            .pivot(
                index="Agreement_Percent_q",
                columns="Lambda_Agreement_Weight",
                values="Jaccard_with_Main",
            )
            .sort_index()
        )

        matrices[
            (
                variant,
                "top2",
            )
        ] = top2_matrix

        matrices[
            (
                variant,
                "clean",
            )
        ] = clean_matrix

        heatmap(
            top2_matrix,
            title=(
                f"{variant}: reliability sensitivity "
                "of mean top-2% injected recovery"
            ),
            colorbar_label=(
                "Mean detection rate (%)"
            ),
            output=(
                output_dir
                / (
                    f"{variant}_Reliability_Sensitivity_"
                    "Top2_Mean.png"
                )
            ),
            fmt=".2f",
        )

        heatmap(
            clean_matrix,
            title=(
                f"{variant}: clean top-2% sensitivity "
                "relative to the predefined reliability setting"
            ),
            colorbar_label=(
                "Jaccard similarity"
            ),
            output=(
                output_dir
                / (
                    f"{variant}_Reliability_Sensitivity_"
                    "Clean_Jaccard.png"
                )
            ),
            fmt=".3f",
        )

    # Compact paper summary.
    paper_rows = []

    for variant in [
        "Full",
        "Pure",
    ]:
        main = all15_df[
            (
                all15_df[
                    "Variant"
                ]
                == variant
            )
            &
            np.isclose(
                all15_df[
                    "Agreement_Fraction_q"
                ],
                MAIN_Q,
            )
            &
            np.isclose(
                all15_df[
                    "Lambda_Agreement_Weight"
                ],
                MAIN_LAMBDA,
            )
        ]

        top2_values = (
            top2_all15[
                top2_all15[
                    "Variant"
                ]
                == variant
            ][
                "Mean_Detection_Rate_Percent"
            ]
        )

        clean_values = (
            clean_df[
                clean_df[
                    "Variant"
                ]
                == variant
            ][
                "Jaccard_with_Main"
            ]
        )

        paper_rows.append(
            {
                "Variant": variant,
                "Main_q_Percent": 2.0,
                "Main_Lambda": 0.60,
                "Main_All15_Top0.1_Mean_DR": float(
                    main[
                        np.isclose(
                            main[
                                "Budget_Fraction"
                            ],
                            0.001,
                        )
                    ][
                        "Mean_Detection_Rate_Percent"
                    ].iloc[0]
                ),
                "Main_All15_Top2_Mean_DR": float(
                    main[
                        np.isclose(
                            main[
                                "Budget_Fraction"
                            ],
                            0.02,
                        )
                    ][
                        "Mean_Detection_Rate_Percent"
                    ].iloc[0]
                ),
                "Top2_Sensitivity_Min": float(
                    top2_values.min()
                ),
                "Top2_Sensitivity_Max": float(
                    top2_values.max()
                ),
                "Top2_Sensitivity_Range_pp": float(
                    top2_values.max()
                    - top2_values.min()
                ),
                "Clean_Jaccard_Min": float(
                    clean_values.min()
                ),
                "Clean_Jaccard_Main": float(
                    clean_df[
                        (
                            clean_df[
                                "Variant"
                            ]
                            == variant
                        )
                        &
                        np.isclose(
                            clean_df[
                                "Agreement_Fraction_q"
                            ],
                            MAIN_Q,
                        )
                        &
                        np.isclose(
                            clean_df[
                                "Lambda_Agreement_Weight"
                            ],
                            MAIN_LAMBDA,
                        )
                    ][
                        "Jaccard_with_Main"
                    ].iloc[0]
                ),
            }
        )

    paper_summary_df = pd.DataFrame(
        paper_rows
    )

    settings_df = pd.DataFrame(
        {
            "Setting": [
                "Clean signal source",
                "Clean current ranking reference",
                "Clean Full definition",
                "Clean Pure definition",
                "Injected source",
                "New output folder",
                "Variants",
                "q values",
                "lambda values",
                "Predefined q",
                "Predefined lambda",
                "Final budgets",
                "Main reliability reconstruction",
                "Spatial/refined score treatment",
                "Uncertainty treatment",
                "q=2% agreement treatment",
                "Model training",
            ],
            "Value": [
                str(clean_source),
                str(clean_reference),
                (
                    "Original/main Full from 03_Proposed_Results_Seed_42; "
                    "clean pre-final score/mask from "
                    "04_Baseline_Comparison_Results_Seed_42"
                ),
                (
                    "Corrected Pure: AE + Pure-specific spatial from "
                    "03_Proposed_Results_Seed_42; corrected clean "
                    "pre-final score/mask synchronized into "
                    "04_Baseline_Comparison_Results_Seed_42"
                ),
                str(injected_root),
                str(output_dir),
                "Full, Pure",
                "1%, 2%, 5%",
                "0.0, 0.2, 0.4, 0.6, 0.8, 1.0",
                "2%",
                "0.60",
                "0.10%, 0.25%, 0.50%, 1.00%, 2.00%",
                (
                    "Validated directly from saved current "
                    "agreement + uncertainty"
                ),
                (
                    "Held fixed; derived once from current "
                    "pre-final score / current saved reliability"
                ),
                (
                    "Current saved normalized uncertainty is "
                    "held fixed for each dataset/variant"
                ),
                (
                    "Current saved q=2% agreement is authoritative; "
                    "recomputed q=2% is recorded only as an audit. "
                    "q=1% and 5% use deterministic exact top-k."
                ),
                "None",
            ],
        }
    )

    # ------------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------------

    clean_df.to_csv(
        output_dir
        / "Reliability_Sensitivity_Clean_Detailed.csv",
        index=False,
    )

    injected_df.to_csv(
        output_dir
        / "Reliability_Sensitivity_Injected_Detailed.csv",
        index=False,
    )

    all15_df.to_csv(
        output_dir
        / "Reliability_Sensitivity_All15_Means.csv",
        index=False,
    )

    scenario_top2.to_csv(
        output_dir
        / "Reliability_Sensitivity_Scenario_Top2.csv",
        index=False,
    )

    validation_df.to_csv(
        output_dir
        / "Reliability_Sensitivity_Validation.csv",
        index=False,
    )

    source_df.to_csv(
        output_dir
        / "Reliability_Sensitivity_Source_Audit.csv",
        index=False,
    )

    output_excel = (
        output_dir
        / "Reliability_Parameter_Sensitivity_Results.xlsx"
    )

    with pd.ExcelWriter(
        output_excel,
        engine="openpyxl",
    ) as writer:
        settings_df.to_excel(
            writer,
            sheet_name="Settings",
            index=False,
        )

        preflight_df.to_excel(
            writer,
            sheet_name="Preflight",
            index=False,
        )

        source_df.to_excel(
            writer,
            sheet_name="Source Audit",
            index=False,
        )

        validation_df.to_excel(
            writer,
            sheet_name="Validation",
            index=False,
        )

        paper_summary_df.to_excel(
            writer,
            sheet_name="Paper Summary",
            index=False,
        )

        clean_df.to_excel(
            writer,
            sheet_name="Clean Detailed",
            index=False,
        )

        all15_df.to_excel(
            writer,
            sheet_name="All15 Means",
            index=False,
        )

        reference_ratio_df.to_excel(
            writer,
            sheet_name="Ref 0.1pct Detailed",
            index=False,
        )

        scenario_top2.to_excel(
            writer,
            sheet_name="Scenario Top2",
            index=False,
        )

        for variant in [
            "Full",
            "Pure",
        ]:
            matrices[
                (
                    variant,
                    "top2",
                )
            ].reset_index().to_excel(
                writer,
                sheet_name=(
                    f"{variant} Top2 Matrix"
                ),
                index=False,
            )

            matrices[
                (
                    variant,
                    "clean",
                )
            ].reset_index().to_excel(
                writer,
                sheet_name=(
                    f"{variant} Clean Jaccard"
                ),
                index=False,
            )

    format_workbook(
        output_excel
    )

    print("\n" + "=" * 80)
    print("RELIABILITY SENSITIVITY COMPLETED")
    print("=" * 80)

    print("\nAll current-pipeline validation checks passed.")

    print("\nImportant audit:")
    print(
        source_df[
            [
                "Scope",
                "Variant",
                "q2_Recomputed_vs_Saved_Agreement_Mismatch_Pixels",
            ]
        ].to_string(
            index=False
        )
    )

    print("\nOutput folder:")
    print(output_dir)

    print("\nMain workbook:")
    print(output_excel)


if __name__ == "__main__":
    main()
