#!/usr/bin/env python3
"""Core Full/Pure scoring logic for the proposed GIS anomaly-detection method.

This module is the cleaned, publication-facing implementation of the final
post-training Full/Pure scoring stage used for the reported manuscript results.
It contains no machine-specific paths and performs no file I/O.

Important implementation detail
--------------------------------
The Full and Pure variants use different local-dispersion constructions because
that is what was used to generate the reported results:

* Full: local dispersion is the standard deviation of the Full base score in
  the fixed neighborhood.
* Pure: the MVAE reconstruction score is locally centered at every pixel first;
  local dispersion at pixel i is the RMS of those locally centered residuals
  within i's neighborhood.

Thus Pure remains independent of IF/LOF/MD and this module matches the actual
reported experimental pipeline rather than the superseded draft equation.

The clean-data calibration is frozen and then reused for injected datasets.
Final anomaly sets rank the pre-final-clipping product (refined score times
reliability), with deterministic exact top-k selection and lower original index
breaking boundary ties.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Iterable, Mapping

import numpy as np
from scipy.ndimage import uniform_filter


# -----------------------------------------------------------------------------
# Locked final configuration used for reported Full/Pure results
# -----------------------------------------------------------------------------
TOP_FRACTION = 0.02
ALPHA = 0.75
BETA = 0.25
LAMBDA_AGREEMENT = 0.60
LAMBDA_CERTAINTY = 0.40
SPATIAL_WINDOW = 5
EPS = 1e-6

FULL_COMPONENT_ORDER = (
    "AE",
    "IF_feature",
    "LOF_feature",
    "MD_feature",
    "IF_embedding",
    "LOF_embedding",
    "MD_embedding",
)


@dataclass(frozen=True)
class VariantCalibration:
    """Clean-data calibration frozen for one scoring variant."""

    spatial_p1: float
    spatial_p99: float
    refined_p1: float
    refined_p99: float
    uncertainty_min: float
    uncertainty_max: float
    final_p1: float
    final_p99: float


@dataclass(frozen=True)
class CalibrationBundle:
    """Frozen clean-data calibration for Full and Pure."""

    full: VariantCalibration
    pure: VariantCalibration
    n_samples: int
    top_fraction: float = TOP_FRACTION
    alpha: float = ALPHA
    beta: float = BETA
    lambda_agreement: float = LAMBDA_AGREEMENT
    lambda_certainty: float = LAMBDA_CERTAINTY
    spatial_window: int = SPATIAL_WINDOW

    def to_dict(self) -> dict:
        return asdict(self)


def _as_float32_vector(x: np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(x).reshape(-1)
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains NaN/Inf")
    return arr.astype(np.float32, copy=False)


def _validate_component_scores(
    components: Mapping[str, np.ndarray],
    context: np.ndarray,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    missing = [name for name in FULL_COMPONENT_ORDER if name not in components]
    if missing:
        raise KeyError(f"Missing Full component(s): {missing}")

    comps = {
        name: _as_float32_vector(components[name], name=name)
        for name in FULL_COMPONENT_ORDER
    }
    ctx = _as_float32_vector(context, name="context")

    lengths = {arr.size for arr in [*comps.values(), ctx]}
    if len(lengths) != 1:
        raise ValueError(f"Component length mismatch: {sorted(lengths)}")
    return comps, ctx


def robust_bounds(x: np.ndarray) -> tuple[float, float]:
    """Return the clean 1st/99th-percentile bounds used by the pipeline."""
    p1, p99 = np.percentile(np.asarray(x), [1.0, 99.0])
    p1, p99 = float(p1), float(p99)
    if not np.isfinite(p1) or not np.isfinite(p99) or p99 <= p1:
        raise ValueError(f"Invalid robust bounds: p1={p1}, p99={p99}")
    return p1, p99


def normalize_robust(x: np.ndarray, p1: float, p99: float) -> np.ndarray:
    """Clip to frozen percentile bounds and map to [0, 1]."""
    if p99 <= p1:
        raise ValueError("p99 must exceed p1")
    y = (np.clip(x, p1, p99) - p1) / (p99 - p1)
    return np.asarray(y, dtype=np.float32)


def normalize_minmax(x: np.ndarray, xmin: float, xmax: float) -> np.ndarray:
    """Frozen min-max normalization used for reliability dispersion."""
    if xmax <= xmin:
        return np.zeros(np.asarray(x).shape, dtype=np.float32)
    y = (x - xmin) / (xmax - xmin)
    return np.clip(y, 0.0, 1.0).astype(np.float32, copy=False)


def exact_top_k(
    scores: np.ndarray,
    *,
    fraction: float | None = None,
    k: int | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Deterministic exact top-k selection.

    Scores strictly above the boundary are selected. If the boundary is tied,
    the remaining observations with the smallest original indices are selected.
    The returned mask is uint8 and contains exactly k selected entries.
    """
    s = np.asarray(scores).reshape(-1)
    if not np.isfinite(s).all():
        raise ValueError("Scores contain NaN/Inf")

    n = s.size
    if k is None:
        if fraction is None or not (0.0 < fraction <= 1.0):
            raise ValueError("Supply k or a fraction in (0, 1]")
        k = int(math.ceil(float(fraction) * n))
    if not (1 <= int(k) <= n):
        raise ValueError(f"Invalid k={k} for n={n}")
    k = int(k)

    threshold = float(np.partition(s, n - k)[n - k])
    selected = s > threshold
    need = k - int(selected.sum())
    if need > 0:
        ties = np.flatnonzero(s == threshold)
        if ties.size < need:
            raise RuntimeError("Insufficient boundary ties")
        selected[ties[:need]] = True

    idx = np.flatnonzero(selected).astype(np.int64, copy=False)
    if idx.size != k:
        raise RuntimeError(f"Exact top-k failed: expected {k}, got {idx.size}")
    return selected.astype(np.uint8, copy=False), idx, threshold


def average_signals(signals: Iterable[np.ndarray]) -> np.ndarray:
    seq = [np.asarray(s, dtype=np.float32) for s in signals]
    if not seq:
        raise ValueError("No signals supplied")
    shape = seq[0].shape
    if any(s.shape != shape for s in seq):
        raise ValueError("Signal shape mismatch")

    out = np.zeros(shape, dtype=np.float32)
    for signal in seq:
        out += signal
    out /= np.float32(len(seq))
    return out


def _validate_mask(base: np.ndarray, valid_mask: np.ndarray, *, variant: str) -> np.ndarray:
    mask = np.asarray(valid_mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError(f"{variant}: valid_mask must be 2-D")
    n_valid = int(mask.sum())
    if np.asarray(base).size != n_valid:
        raise ValueError(
            f"{variant}: base has {np.asarray(base).size:,} scores but "
            f"valid_mask has {n_valid:,} valid cells"
        )
    return mask


def spatial_full(
    base: np.ndarray,
    valid_mask: np.ndarray,
    *,
    window_size: int = SPATIAL_WINDOW,
    eps: float = EPS,
) -> np.ndarray:
    """Full local spatial inconsistency used for the reported results.

    The denominator is the ordinary standard deviation of the Full base score
    over valid pixels in the fixed neighborhood.
    """
    base = _as_float32_vector(base, name="full_base")
    mask = _validate_mask(base, valid_mask, variant="Full")

    raster = np.full(mask.shape, np.nan, dtype=np.float32)
    raster[mask] = base
    valid = np.isfinite(raster)
    area = float(window_size * window_size)

    def nan_local_mean(arr: np.ndarray) -> np.ndarray:
        ok = np.isfinite(arr)
        filled = np.where(ok, arr, 0.0).astype(np.float32)
        local_sum = uniform_filter(
            filled, size=window_size, mode="nearest"
        ) * area
        local_count = uniform_filter(
            ok.astype(np.float32), size=window_size, mode="nearest"
        ) * area
        return local_sum / np.maximum(local_count, 1.0)

    local_mean = nan_local_mean(raster)
    second = np.where(valid, raster ** 2, np.nan)
    local_second = nan_local_mean(second)
    local_var = np.maximum(local_second - local_mean ** 2, 0.0)
    local_std = np.sqrt(local_var)

    score = np.abs(raster - local_mean) / (local_std + eps)
    out = score[mask].astype(np.float32, copy=False)
    if not np.isfinite(out).all():
        raise ValueError("Full spatial score contains NaN/Inf")
    return out


def spatial_pure(
    ae_score: np.ndarray,
    valid_mask: np.ndarray,
    *,
    window_size: int = SPATIAL_WINDOW,
    eps: float = EPS,
) -> np.ndarray:
    """Pure-specific local spatial inconsistency used for reported results.

    This function intentionally reproduces the final reported implementation.
    It does *not* reuse the Full spatial field.

    Let m(j) be the local mean of the MVAE reconstruction score around pixel j
    and r(j) = AE(j) - m(j). For pixel i, the denominator is the RMS of r(j)
    over valid pixels j in i's neighborhood. The numerator is |r(i)|.

    This is distinct from taking the ordinary standard deviation of AE directly
    inside i's neighborhood, and the distinction is documented in the revised
    manuscript method description.
    """
    ae = _as_float32_vector(ae_score, name="AE")
    mask = _validate_mask(ae, valid_mask, variant="Pure")

    raster = np.full(mask.shape, np.nan, dtype=np.float32)
    raster[mask] = ae

    valid_float = mask.astype(np.float32, copy=False)
    filled = np.where(mask, raster, 0.0).astype(np.float32, copy=False)
    area = float(window_size * window_size)

    local_sum = uniform_filter(
        filled, size=window_size, mode="nearest"
    ) * area
    local_count = uniform_filter(
        valid_float, size=window_size, mode="nearest"
    ) * area
    local_mean = local_sum / np.maximum(local_count, 1.0)

    residual_sq = (raster - local_mean) ** 2
    residual_sq = np.where(np.isnan(residual_sq), 0.0, residual_sq).astype(
        np.float32, copy=False
    )

    local_residual_sq_sum = uniform_filter(
        residual_sq, size=window_size, mode="nearest"
    ) * area
    local_residual_variance = local_residual_sq_sum / np.maximum(local_count, 1.0)
    local_residual_rms = np.sqrt(
        np.maximum(local_residual_variance, 0.0)
    ).astype(np.float32, copy=False)

    score = np.abs(raster - local_mean) / (local_residual_rms + eps)
    out = score[mask].astype(np.float32, copy=False)
    if not np.isfinite(out).all():
        raise ValueError("Pure spatial score contains NaN/Inf")
    return out


def _reliability_clean(
    signals: list[np.ndarray],
    *,
    top_fraction: float = TOP_FRACTION,
    lambda_agreement: float = LAMBDA_AGREEMENT,
    lambda_certainty: float = LAMBDA_CERTAINTY,
) -> tuple[np.ndarray, float, float, np.ndarray, np.ndarray]:
    if not signals:
        raise ValueError("No reliability signals")
    n = signals[0].size
    if any(np.asarray(s).size != n for s in signals):
        raise ValueError("Reliability signal length mismatch")

    agreement = np.zeros(n, dtype=np.uint8)
    for signal in signals:
        flag, _, _ = exact_top_k(signal, fraction=top_fraction)
        agreement += flag
    agreement_fraction = agreement.astype(np.float32) / np.float32(len(signals))

    stack = np.stack(signals, axis=0).astype(np.float32, copy=False)
    uncertainty_raw = np.std(stack, axis=0, dtype=np.float32).astype(
        np.float32, copy=False
    )
    umin = float(uncertainty_raw.min())
    umax = float(uncertainty_raw.max())
    uncertainty_norm = normalize_minmax(uncertainty_raw, umin, umax)

    reliability = (
        np.float32(lambda_agreement) * agreement_fraction
        + np.float32(lambda_certainty)
        * (np.float32(1.0) - uncertainty_norm)
    )
    reliability = np.clip(reliability, 0.0, 1.0).astype(np.float32, copy=False)
    return reliability, umin, umax, agreement, uncertainty_norm


def _reliability_apply(
    signals: list[np.ndarray],
    umin: float,
    umax: float,
    *,
    top_fraction: float = TOP_FRACTION,
    lambda_agreement: float = LAMBDA_AGREEMENT,
    lambda_certainty: float = LAMBDA_CERTAINTY,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not signals:
        raise ValueError("No reliability signals")
    n = signals[0].size
    if any(np.asarray(s).size != n for s in signals):
        raise ValueError("Reliability signal length mismatch")

    agreement = np.zeros(n, dtype=np.uint8)
    for signal in signals:
        flag, _, _ = exact_top_k(signal, fraction=top_fraction)
        agreement += flag
    agreement_fraction = agreement.astype(np.float32) / np.float32(len(signals))

    stack = np.stack(signals, axis=0).astype(np.float32, copy=False)
    uncertainty_raw = np.std(stack, axis=0, dtype=np.float32).astype(
        np.float32, copy=False
    )
    uncertainty_norm = normalize_minmax(uncertainty_raw, umin, umax)

    reliability = (
        np.float32(lambda_agreement) * agreement_fraction
        + np.float32(lambda_certainty)
        * (np.float32(1.0) - uncertainty_norm)
    )
    reliability = np.clip(reliability, 0.0, 1.0).astype(np.float32, copy=False)
    return reliability, agreement, uncertainty_norm


def score_clean(
    components: Mapping[str, np.ndarray],
    context: np.ndarray,
    valid_mask: np.ndarray,
) -> tuple[CalibrationBundle, dict[str, np.ndarray]]:
    """Fit clean calibration and compute final Full/Pure clean scores.

    Parameters
    ----------
    components:
        Mapping containing the seven normalized component signals named by
        ``FULL_COMPONENT_ORDER``. ``AE`` is the normalized MVAE reconstruction
        score and is also the Pure base score.
    context:
        Normalized context-autoencoder score.
    valid_mask:
        Original 2-D raster-validity mask. ``valid_mask.sum()`` must equal the
        score-vector length.

    Returns
    -------
    calibration, outputs
        The calibration is frozen and must be passed unchanged to
        :func:`score_injected`.
    """
    comps, ctx = _validate_component_scores(components, context)
    ae = comps["AE"]
    n = ae.size
    mask = _validate_mask(ae, valid_mask, variant="Clean")

    # ----- Full -----
    full_base = average_signals(comps[name] for name in FULL_COMPONENT_ORDER)
    full_spatial_raw = spatial_full(full_base, mask)
    fs1, fs99 = robust_bounds(full_spatial_raw)
    full_spatial_norm = normalize_robust(full_spatial_raw, fs1, fs99)

    full_refined_raw = (
        np.float32(ALPHA) * full_base
        + np.float32(BETA) * full_spatial_norm
    ).astype(np.float32)
    fr1, fr99 = robust_bounds(full_refined_raw)
    full_refined_norm = normalize_robust(full_refined_raw, fr1, fr99)

    full_reliability_signals = [
        *(comps[name] for name in FULL_COMPONENT_ORDER),
        full_spatial_norm,
        ctx,
    ]
    (
        full_reliability,
        fu_min,
        fu_max,
        full_agreement,
        full_uncertainty,
    ) = _reliability_clean(full_reliability_signals)

    full_final_raw = (full_refined_norm * full_reliability).astype(
        np.float32, copy=False
    )
    ff1, ff99 = robust_bounds(full_final_raw)
    full_final_normalized = normalize_robust(full_final_raw, ff1, ff99)
    full_mask, full_indices, full_threshold = exact_top_k(
        full_final_raw, fraction=TOP_FRACTION
    )

    # ----- Pure -----
    pure_base = ae
    pure_spatial_raw = spatial_pure(pure_base, mask)
    ps1, ps99 = robust_bounds(pure_spatial_raw)
    pure_spatial_norm = normalize_robust(pure_spatial_raw, ps1, ps99)

    pure_refined_raw = (
        np.float32(ALPHA) * pure_base
        + np.float32(BETA) * pure_spatial_norm
    ).astype(np.float32)
    pr1, pr99 = robust_bounds(pure_refined_raw)
    pure_refined_norm = normalize_robust(pure_refined_raw, pr1, pr99)

    pure_reliability_signals = [pure_base, pure_spatial_norm, ctx]
    (
        pure_reliability,
        pu_min,
        pu_max,
        pure_agreement,
        pure_uncertainty,
    ) = _reliability_clean(pure_reliability_signals)

    pure_final_raw = (pure_refined_norm * pure_reliability).astype(
        np.float32, copy=False
    )
    pf1, pf99 = robust_bounds(pure_final_raw)
    pure_final_normalized = normalize_robust(pure_final_raw, pf1, pf99)
    pure_mask, pure_indices, pure_threshold = exact_top_k(
        pure_final_raw, fraction=TOP_FRACTION
    )

    calibration = CalibrationBundle(
        full=VariantCalibration(
            fs1, fs99, fr1, fr99, fu_min, fu_max, ff1, ff99
        ),
        pure=VariantCalibration(
            ps1, ps99, pr1, pr99, pu_min, pu_max, pf1, pf99
        ),
        n_samples=n,
    )

    outputs = {
        "full_base": full_base,
        "full_spatial_raw": full_spatial_raw,
        "full_spatial": full_spatial_norm,
        "full_refined_raw": full_refined_raw,
        "full_refined": full_refined_norm,
        "full_reliability": full_reliability,
        "full_agreement": full_agreement,
        "full_uncertainty": full_uncertainty,
        "full_final_raw": full_final_raw,
        "full_final_normalized": full_final_normalized,
        "full_mask": full_mask,
        "full_indices": full_indices,
        "full_threshold": np.asarray(full_threshold),
        "pure_base": pure_base,
        "pure_spatial_raw": pure_spatial_raw,
        "pure_spatial": pure_spatial_norm,
        "pure_refined_raw": pure_refined_raw,
        "pure_refined": pure_refined_norm,
        "pure_reliability": pure_reliability,
        "pure_agreement": pure_agreement,
        "pure_uncertainty": pure_uncertainty,
        "pure_final_raw": pure_final_raw,
        "pure_final_normalized": pure_final_normalized,
        "pure_mask": pure_mask,
        "pure_indices": pure_indices,
        "pure_threshold": np.asarray(pure_threshold),
    }
    return calibration, outputs


def score_injected(
    components: Mapping[str, np.ndarray],
    context: np.ndarray,
    valid_mask: np.ndarray,
    calibration: CalibrationBundle,
) -> dict[str, np.ndarray]:
    """Score one injected dataset using frozen clean-data calibration."""
    comps, ctx = _validate_component_scores(components, context)
    ae = comps["AE"]
    if ae.size != calibration.n_samples:
        raise ValueError(
            f"Injected score length {ae.size:,} != clean calibration "
            f"length {calibration.n_samples:,}"
        )
    mask = _validate_mask(ae, valid_mask, variant="Injected")

    # Full.
    full_base = average_signals(comps[name] for name in FULL_COMPONENT_ORDER)
    full_spatial_raw = spatial_full(full_base, mask)
    full_spatial_norm = normalize_robust(
        full_spatial_raw,
        calibration.full.spatial_p1,
        calibration.full.spatial_p99,
    )
    full_refined_raw = (
        np.float32(ALPHA) * full_base
        + np.float32(BETA) * full_spatial_norm
    ).astype(np.float32)
    full_refined_norm = normalize_robust(
        full_refined_raw,
        calibration.full.refined_p1,
        calibration.full.refined_p99,
    )
    full_reliability, full_agreement, full_uncertainty = _reliability_apply(
        [
            *(comps[name] for name in FULL_COMPONENT_ORDER),
            full_spatial_norm,
            ctx,
        ],
        calibration.full.uncertainty_min,
        calibration.full.uncertainty_max,
    )
    full_final_raw = (full_refined_norm * full_reliability).astype(
        np.float32, copy=False
    )
    full_final_normalized = normalize_robust(
        full_final_raw,
        calibration.full.final_p1,
        calibration.full.final_p99,
    )

    # Pure.
    pure_spatial_raw = spatial_pure(ae, mask)
    pure_spatial_norm = normalize_robust(
        pure_spatial_raw,
        calibration.pure.spatial_p1,
        calibration.pure.spatial_p99,
    )
    pure_refined_raw = (
        np.float32(ALPHA) * ae
        + np.float32(BETA) * pure_spatial_norm
    ).astype(np.float32)
    pure_refined_norm = normalize_robust(
        pure_refined_raw,
        calibration.pure.refined_p1,
        calibration.pure.refined_p99,
    )
    pure_reliability, pure_agreement, pure_uncertainty = _reliability_apply(
        [ae, pure_spatial_norm, ctx],
        calibration.pure.uncertainty_min,
        calibration.pure.uncertainty_max,
    )
    pure_final_raw = (pure_refined_norm * pure_reliability).astype(
        np.float32, copy=False
    )
    pure_final_normalized = normalize_robust(
        pure_final_raw,
        calibration.pure.final_p1,
        calibration.pure.final_p99,
    )

    return {
        "full_base": full_base,
        "full_spatial_raw": full_spatial_raw,
        "full_spatial": full_spatial_norm,
        "full_refined_raw": full_refined_raw,
        "full_refined": full_refined_norm,
        "full_reliability": full_reliability,
        "full_agreement": full_agreement,
        "full_uncertainty": full_uncertainty,
        "full_final_raw": full_final_raw,
        "full_final_normalized": full_final_normalized,
        "pure_base": ae,
        "pure_spatial_raw": pure_spatial_raw,
        "pure_spatial": pure_spatial_norm,
        "pure_refined_raw": pure_refined_raw,
        "pure_refined": pure_refined_norm,
        "pure_reliability": pure_reliability,
        "pure_agreement": pure_agreement,
        "pure_uncertainty": pure_uncertainty,
        "pure_final_raw": pure_final_raw,
        "pure_final_normalized": pure_final_normalized,
    }
