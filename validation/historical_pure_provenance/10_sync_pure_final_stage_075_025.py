import os
from pathlib import Path
import math
import numpy as np

# ============================================================
# TARGETED PURE FINAL-STAGE SYNCHRONIZATION
# ------------------------------------------------------------
# Purpose:
#   Synchronize ONLY the downstream Pure fusion/reliability/final files
#   after locking Pure fusion to 0.75 AE + 0.25 Pure-specific spatial.
#
# Assumption:
#   pure_spatial_score_normalized.npy has already been generated correctly
#   from AE only by the earlier Pure-spatial correction.
#
# This script does NOT:
#   - retrain any model
#   - recompute the Pure spatial raster/neighborhood calculation
#   - touch Full
#   - touch baselines
#
# It DOES:
#   - recompute Pure spatial fusion with 0.75/0.25
#   - recompute agreement / uncertainty / reliability
#   - recompute final Pure raw and normalized scores
#   - recompute exact top-2% Pure labels / indices
#   - update only Pure keys in the existing parameter/diagnostic archives
#   - apply clean calibration to all 15 injected scenarios
#   - recompute the same downstream Pure stage independently for all
#     five clean stability-seed folders
# ============================================================

ROOT_DIR = Path(os.environ.get("GIS_ANOMALY_PROJECT_ROOT", Path.cwd())).resolve()

MAIN_DIR = ROOT_DIR / "03_Proposed_Results_Seed_42"
INJECTED_ROOT = ROOT_DIR / "03_Injected_Proposed_Results_Seed_42"

STABILITY_SEEDS = [42, 7, 123, 2024, 2025]

RATIO_FOLDERS = {
    "0.05%": "Injected_Datasets_with_0.05%_Injection_Rate",
    "0.1%": "Injected_Datasets_with_0.1%_Injection_Rate",
    "0.5%": "Injected_Datasets_with_0.5%_Injection_Rate",
}

SCENARIOS = [
    "extreme",
    "noise_strong",
    "noise_moderate",
    "noise_mild",
    "inconsistent",
]

PURE_BASE_WEIGHT = 0.75
PURE_SPATIAL_WEIGHT = 0.25
AGREEMENT_WEIGHT = 0.60
CERTAINTY_WEIGHT = 0.40
TOP_FRACTION = 0.02


def load_vector(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    x = np.load(path, allow_pickle=False)
    x = np.asarray(x).reshape(-1)
    if not np.isfinite(x).all():
        raise ValueError(f"Non-finite values in {path}")
    return x.astype(np.float32, copy=False)


def percentile_bounds(x: np.ndarray) -> tuple[float, float]:
    p1, p99 = np.percentile(x, [1.0, 99.0])
    p1 = float(p1)
    p99 = float(p99)
    if p99 <= p1:
        raise ValueError(f"Invalid bounds: {p1}, {p99}")
    return p1, p99


def normalize_bounds(x: np.ndarray, p1: float, p99: float) -> np.ndarray:
    return ((np.clip(x, p1, p99) - p1) / (p99 - p1)).astype(np.float32, copy=False)


def normalize_minmax(x: np.ndarray, xmin: float, xmax: float) -> np.ndarray:
    if xmax <= xmin:
        return np.zeros_like(x, dtype=np.float32)
    y = (x - xmin) / (xmax - xmin)
    return np.clip(y, 0.0, 1.0).astype(np.float32, copy=False)


def exact_top_fraction(scores: np.ndarray, fraction: float = TOP_FRACTION):
    scores = np.asarray(scores).reshape(-1)
    n = scores.size
    k = int(math.ceil(fraction * n))
    partition_position = n - k
    threshold = np.partition(scores, partition_position)[partition_position]

    above = np.flatnonzero(scores > threshold)
    tied = np.flatnonzero(scores == threshold)
    need = k - above.size
    selected = np.concatenate([above, tied[:need]])

    if selected.size != k:
        raise RuntimeError(f"Exact top-k failed: {selected.size} != {k}")

    # Deterministic ordering: descending score, then ascending index.
    order = np.lexsort((selected, -scores[selected]))
    selected = selected[order].astype(np.int64, copy=False)

    mask = np.zeros(n, dtype=np.uint8)
    mask[selected] = 1
    return mask, selected, float(threshold)


def npz_dict(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        return {}
    with np.load(path, allow_pickle=False) as z:
        return {k: np.array(z[k], copy=True) for k in z.files}


def update_npz(path: Path, updates: dict[str, object]) -> None:
    payload = npz_dict(path)
    for key, value in updates.items():
        payload[key] = np.asarray(value)
    np.savez_compressed(path, **payload)


def agreement_uncertainty_reliability(
    ae: np.ndarray,
    spatial: np.ndarray,
    context: np.ndarray,
    *,
    uncertainty_bounds: tuple[float, float] | None,
):
    ae_flag, _, _ = exact_top_fraction(ae)
    spatial_flag, _, _ = exact_top_fraction(spatial)
    context_flag, _, _ = exact_top_fraction(context)

    agreement_count = (
        ae_flag.astype(np.uint8)
        + spatial_flag.astype(np.uint8)
        + context_flag.astype(np.uint8)
    ).astype(np.uint8, copy=False)

    agreement_fraction = agreement_count.astype(np.float32) / 3.0

    stack = np.stack([ae, spatial, context], axis=0).astype(np.float32, copy=False)
    uncertainty_raw = np.std(stack, axis=0).astype(np.float32, copy=False)

    if uncertainty_bounds is None:
        umin = float(uncertainty_raw.min())
        umax = float(uncertainty_raw.max())
    else:
        umin, umax = map(float, uncertainty_bounds)

    uncertainty_norm = normalize_minmax(uncertainty_raw, umin, umax)

    reliability = (
        AGREEMENT_WEIGHT * agreement_fraction
        + CERTAINTY_WEIGHT * (1.0 - uncertainty_norm)
    ).astype(np.float32, copy=False)
    reliability = np.clip(reliability, 0.0, 1.0).astype(np.float32, copy=False)

    return agreement_count, uncertainty_raw, uncertainty_norm, reliability, umin, umax


def save_final_outputs(
    folder: Path,
    final_raw: np.ndarray,
    final_norm: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
) -> None:
    np.save(folder / "pure_final_anomaly_score_raw.npy", final_raw)
    np.save(folder / "pure_final_fused_score_preclip.npy", final_raw)
    np.save(folder / "pure_final_anomaly_score.npy", final_norm)
    np.save(folder / "pure_final_anomaly_labels.npy", labels.astype(np.uint8, copy=False))
    np.save(folder / "Pure_top2pct_anomaly_mask.npy", labels.astype(np.uint8, copy=False))
    np.save(folder / "Pure_top2pct_selected_indices.npy", indices.astype(np.int64, copy=False))


def compute_clean_calibration(folder: Path, update_main_archives: bool = True):
    print(f"\n[CLEAN] {folder}")
    ae = load_vector(folder / "ae_score_normalized.npy")
    spatial = load_vector(folder / "pure_spatial_score_normalized.npy")
    context = load_vector(folder / "context_score_normalized.npy")

    if not (ae.size == spatial.size == context.size):
        raise ValueError("Pure signal lengths do not match.")

    fused_raw = (
        PURE_BASE_WEIGHT * ae
        + PURE_SPATIAL_WEIGHT * spatial
    ).astype(np.float32, copy=False)

    fused_p1, fused_p99 = percentile_bounds(fused_raw)
    fused_norm = normalize_bounds(fused_raw, fused_p1, fused_p99)

    (
        agreement,
        uncertainty_raw,
        uncertainty_norm,
        reliability,
        uncertainty_min,
        uncertainty_max,
    ) = agreement_uncertainty_reliability(
        ae,
        spatial,
        context,
        uncertainty_bounds=None,
    )

    final_raw = (fused_norm * reliability).astype(np.float32, copy=False)
    final_p1, final_p99 = percentile_bounds(final_raw)
    final_norm = normalize_bounds(final_raw, final_p1, final_p99)
    labels, indices, threshold = exact_top_fraction(final_raw)

    save_final_outputs(folder, final_raw, final_norm, labels, indices)

    if update_main_archives:
        update_npz(
            folder / "full_pure_reliability_diagnostics.npz",
            {
                "pure_agreement": agreement,
                "pure_uncertainty": uncertainty_norm,
                "pure_uncertainty_raw": uncertainty_raw,
                "pure_reliability": reliability,
            },
        )

        update_npz(
            folder / "full_pure_final_normalization_params.npz",
            {
                # Existing key name 'pure_spatial_p1/p99' means the
                # Pure spatial-fusion normalization bounds.
                "pure_spatial_p1": np.float32(fused_p1),
                "pure_spatial_p99": np.float32(fused_p99),
                "pure_uncertainty_min": np.float32(uncertainty_min),
                "pure_uncertainty_max": np.float32(uncertainty_max),
                "pure_final_p1": np.float32(final_p1),
                "pure_final_p99": np.float32(final_p99),
                "pure_base_weight": np.float32(PURE_BASE_WEIGHT),
                "pure_spatial_weight": np.float32(PURE_SPATIAL_WEIGHT),
                "consensus_weight": np.float32(AGREEMENT_WEIGHT),
                "certainty_weight": np.float32(CERTAINTY_WEIGHT),
            },
        )

    # Internal integrity check.
    reconstructed = normalize_bounds(final_raw, final_p1, final_p99)
    max_diff = float(np.max(np.abs(reconstructed - final_norm)))
    if not np.allclose(reconstructed, final_norm, rtol=1e-7, atol=1e-7):
        raise RuntimeError(f"Clean Pure normalization self-check failed: {max_diff}")

    print(f"  weights: {PURE_BASE_WEIGHT:.2f} / {PURE_SPATIAL_WEIGHT:.2f}")
    print(f"  fused p1/p99: {fused_p1:.9g} / {fused_p99:.9g}")
    print(f"  uncertainty min/max: {uncertainty_min:.9g} / {uncertainty_max:.9g}")
    print(f"  final p1/p99: {final_p1:.9g} / {final_p99:.9g}")
    print(f"  selected: {int(labels.sum()):,}")
    print(f"  top-2% boundary: {threshold:.9g}")
    print(f"  normalized self-check max diff: {max_diff:.3g}")

    return {
        "fused_p1": fused_p1,
        "fused_p99": fused_p99,
        "uncertainty_min": uncertainty_min,
        "uncertainty_max": uncertainty_max,
        "final_p1": final_p1,
        "final_p99": final_p99,
    }


def apply_clean_calibration_to_injected(folder: Path, calibration: dict[str, float]):
    ae = load_vector(folder / "ae_score_normalized.npy")
    spatial = load_vector(folder / "pure_spatial_score_normalized.npy")
    context = load_vector(folder / "context_score_normalized.npy")

    fused_raw = (
        PURE_BASE_WEIGHT * ae
        + PURE_SPATIAL_WEIGHT * spatial
    ).astype(np.float32, copy=False)
    fused_norm = normalize_bounds(fused_raw, calibration["fused_p1"], calibration["fused_p99"])

    agreement, uncertainty_raw, uncertainty_norm, reliability, _, _ = (
        agreement_uncertainty_reliability(
            ae,
            spatial,
            context,
            uncertainty_bounds=(
                calibration["uncertainty_min"],
                calibration["uncertainty_max"],
            ),
        )
    )

    final_raw = (fused_norm * reliability).astype(np.float32, copy=False)
    final_norm = normalize_bounds(final_raw, calibration["final_p1"], calibration["final_p99"])
    labels, indices, _ = exact_top_fraction(final_raw)

    save_final_outputs(folder, final_raw, final_norm, labels, indices)

    np.savez_compressed(
        folder / "pure_reliability_diagnostics.npz",
        agreement=agreement.astype(np.uint8, copy=False),
        uncertainty_raw=uncertainty_raw,
        uncertainty_normalized=uncertainty_norm,
        reliability=reliability,
    )


# ---------------------------------------------------------------------------
# 1. Synchronize the clean reference Pure stage and obtain clean calibration.
# ---------------------------------------------------------------------------
if not MAIN_DIR.exists():
    raise FileNotFoundError(MAIN_DIR)

main_cal = compute_clean_calibration(MAIN_DIR, update_main_archives=True)

# ---------------------------------------------------------------------------
# 2. Apply the corrected clean calibration to all 15 injected scenarios.
# ---------------------------------------------------------------------------
processed_injected = 0
for ratio_label, ratio_folder_name in RATIO_FOLDERS.items():
    ratio_dir = INJECTED_ROOT / ratio_folder_name
    if not ratio_dir.exists():
        raise FileNotFoundError(ratio_dir)

    for scenario in SCENARIOS:
        direct = ratio_dir / scenario
        candidates = []
        if direct.is_dir():
            candidates.append(direct)
        candidates.extend(p for p in ratio_dir.rglob(scenario) if p.is_dir() and p != direct)
        candidates = sorted(set(candidates), key=lambda p: str(p).lower())
        usable = [
            p for p in candidates
            if (p / "ae_score_normalized.npy").exists()
            and (p / "pure_spatial_score_normalized.npy").exists()
            and (p / "context_score_normalized.npy").exists()
        ]
        if len(usable) != 1:
            raise RuntimeError(
                f"{ratio_label} / {scenario}: expected exactly one usable folder, found {len(usable)}: {usable}"
            )

        folder = usable[0]
        apply_clean_calibration_to_injected(folder, main_cal)
        processed_injected += 1
        print(f"[INJECTED] {ratio_label:>5s} | {scenario:16s} | updated")

if processed_injected != 15:
    raise RuntimeError(f"Expected 15 injected scenarios, updated {processed_injected}")

# ---------------------------------------------------------------------------
# 3. Synchronize each clean stability seed independently.
# ---------------------------------------------------------------------------
processed_seeds = []
for seed in STABILITY_SEEDS:
    folder = ROOT_DIR / f"07_Proposed_Results_Seed_{seed}"
    if not folder.exists():
        raise FileNotFoundError(folder)
    compute_clean_calibration(folder, update_main_archives=True)
    processed_seeds.append(seed)

print("\n" + "=" * 78)
print("PURE FINAL-STAGE SYNCHRONIZATION COMPLETED")
print("=" * 78)
print("Main clean folder updated.")
print(f"Injected scenarios updated: {processed_injected}/15")
print(f"Stability seeds updated: {processed_seeds}")
print("Full and all baselines were untouched.")
print("Locked Pure fusion: 0.75 AE + 0.25 Pure-specific spatial.")
print("\nNEXT:")
print("1. Rerun 09_refresh_injected_proposed_evaluation_only.py")
print("2. Restart kernel and rerun notebook 04 from the beginning")
print("3. Then rerun 05, 06, and 07")
