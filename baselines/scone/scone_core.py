"""Core SCoNE implementation for the geospatial baseline experiment.

Independent reimplementation of Xu et al., AAAI 2026, Eqs. (2)-(5).
Higher returned score means more anomalous: S = 1 - Cbar.
"""
from __future__ import annotations

import gc
import math
from dataclasses import dataclass
from typing import Dict, Mapping, Sequence

import numpy as np

VIEW_ORDER = ["terrain", "hydrological", "spatial", "surface", "infrastructure", "landuse"]
VIEW_DIMS = {"terrain": 5, "hydrological": 3, "spatial": 4, "surface": 3, "infrastructure": 1, "landuse": 7}


def exact_top_k_order(scores: np.ndarray, k: int) -> np.ndarray:
    """Deterministic exact top-k. Boundary ties -> lower original index."""
    scores = np.asarray(scores).reshape(-1)
    if not np.isfinite(scores).all():
        raise ValueError("Scores contain NaN/Inf.")
    if not 1 <= k <= scores.size:
        raise ValueError(f"Invalid k={k} for n={scores.size}.")
    boundary = np.partition(scores, scores.size - k)[scores.size - k]
    above = np.flatnonzero(scores > boundary)
    tied = np.flatnonzero(scores == boundary)
    need = k - above.size
    if need < 0 or need > tied.size:
        raise RuntimeError("Exact top-k boundary logic failed.")
    selected = np.concatenate([above, tied[:need]])
    if selected.size != k:
        raise RuntimeError(f"Selected {selected.size}; expected {k}.")
    order = np.lexsort((selected, -scores[selected]))
    return selected[order]


def exact_top_fraction_order(scores: np.ndarray, fraction: float) -> np.ndarray:
    return exact_top_k_order(scores, int(math.ceil(fraction * np.asarray(scores).size)))


@dataclass
class SCoNEModel:
    psi: int
    t: int
    seed: int
    center_global_indices: np.ndarray
    centers: Dict[str, np.ndarray]      # view -> (t, psi, d_v)
    radii_sq: Dict[str, np.ndarray]     # view -> (t, psi)

    def validate(self) -> None:
        if self.center_global_indices.shape != (self.t, self.psi):
            raise ValueError("center_global_indices shape mismatch.")
        for name in VIEW_ORDER:
            c = self.centers[name]
            r = self.radii_sq[name]
            if c.shape != (self.t, self.psi, VIEW_DIMS[name]):
                raise ValueError(f"{name} center shape mismatch: {c.shape}")
            if r.shape != (self.t, self.psi):
                raise ValueError(f"{name} radius shape mismatch: {r.shape}")
            if not np.isfinite(c).all() or not np.isfinite(r).all() or np.any(r < 0):
                raise ValueError(f"Invalid {name} model arrays.")


def pairwise_center_radii_sq(centers: np.ndarray) -> np.ndarray:
    """SCoNE Eq. (2): nearest-other-center squared radius per ensemble/view."""
    centers = np.asarray(centers, dtype=np.float32)
    norms = np.sum(centers * centers, axis=2, dtype=np.float32)
    gram = np.matmul(centers, np.swapaxes(centers, 1, 2))
    dist_sq = (norms[:, :, None] + norms[:, None, :] - 2.0 * gram).astype(np.float32, copy=False)
    np.maximum(dist_sq, 0.0, out=dist_sq)
    diag = np.arange(centers.shape[1])
    dist_sq[:, diag, diag] = np.inf
    return np.min(dist_sq, axis=2).astype(np.float32, copy=False)


def deterministic_center_indices(reference_pool_indices: np.ndarray, psi: int, t: int, seed: int) -> np.ndarray:
    """Draw t independent SCoNE subsamples from a fixed clean reference pool."""
    pool = np.asarray(reference_pool_indices, dtype=np.int64).reshape(-1)
    if np.unique(pool).size != pool.size:
        raise ValueError("Reference-pool indices are not unique.")
    if psi < 2 or psi > pool.size:
        raise ValueError(f"Invalid psi={psi} for pool size {pool.size}.")
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(psi), 2601]))
    out = np.empty((t, psi), dtype=np.int64)
    for e in range(t):
        out[e] = pool[rng.choice(pool.size, size=psi, replace=False)]
    return out


def build_model(raw_X: np.ndarray, preprocessor, reference_pool_indices: np.ndarray, psi: int, t: int, seed: int) -> SCoNEModel:
    center_idx = deterministic_center_indices(reference_pool_indices, psi, t, seed)
    raw_centers = np.asarray(raw_X[center_idx.reshape(-1)], dtype=np.float32)
    flat_views = preprocessor.transform(raw_centers)
    centers, radii = {}, {}
    for name in VIEW_ORDER:
        arr = flat_views[name].reshape(t, psi, VIEW_DIMS[name])
        centers[name] = arr
        radii[name] = pairwise_center_radii_sq(arr)
    model = SCoNEModel(psi, t, seed, center_idx, centers, radii)
    model.validate()
    del raw_centers, flat_views
    gc.collect()
    return model


def _dist_sq(X: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Squared Euclidean distances: (B,d) vs (E,psi,d) -> (B,E,psi)."""
    X = np.asarray(X, dtype=np.float32)
    centers = np.asarray(centers, dtype=np.float32)
    B, d = X.shape
    E, psi, dc = centers.shape
    if d != dc:
        raise ValueError(f"Distance dimension mismatch {d} vs {dc}.")
    C = centers.reshape(E * psi, d)
    xn = np.sum(X * X, axis=1, dtype=np.float32)[:, None]
    cn = np.sum(C * C, axis=1, dtype=np.float32)[None, :]
    D = (xn + cn - 2.0 * np.matmul(X, C.T)).astype(np.float32, copy=False)
    np.maximum(D, 0.0, out=D)
    return D.reshape(B, E, psi)


def score_views_multi_k(
    views: Mapping[str, np.ndarray],
    model: SCoNEModel,
    k_values: Sequence[int],
    *,
    batch_size: int = 8000,
    target_distance_elements: int = 4_000_000,
    progress_label: str = "",
) -> Dict[int, np.ndarray]:
    """Optimized exact SCoNE Eqs. (3)-(5), with deterministic stable kNN ties."""
    model.validate()
    ks = sorted(set(int(k) for k in k_values))
    if not ks or min(ks) < 1 or max(ks) > model.psi:
        raise ValueError(f"Invalid k values {ks} for psi={model.psi}.")

    n = None
    for name in VIEW_ORDER:
        arr = np.asarray(views[name])
        if arr.ndim != 2 or arr.shape[1] != VIEW_DIMS[name]:
            raise ValueError(f"Invalid {name} query shape {arr.shape}.")
        n = arr.shape[0] if n is None else n
        if arr.shape[0] != n:
            raise ValueError("Query views have inconsistent sample counts.")
    out = {k: np.empty(n, dtype=np.float32) for k in ks}
    n_batches = math.ceil(n / batch_size)

    for bi, b0 in enumerate(range(0, n, batch_size), 1):
        b1 = min(b0 + batch_size, n)
        B = b1 - b0
        counts = {k: np.zeros(B, dtype=np.uint32) for k in ks}
        e_block = max(1, min(model.t, target_distance_elements // max(1, B * model.psi)))

        for e0 in range(0, model.t, e_block):
            e1 = min(e0 + e_block, model.t)
            E = e1 - e0
            consistent = {k: np.ones((B, E, model.psi), dtype=bool) for k in ks}

            for name in VIEW_ORDER:
                D = _dist_sq(np.asarray(views[name][b0:b1], dtype=np.float32), model.centers[name][e0:e1])
                inside = D <= model.radii_sq[name][e0:e1][None, :, :]
                # Stable sort is deliberately used: the categorical one-hot land-use
                # view can produce exact distance ties, and reproducibility matters.
                order = np.argsort(D, axis=2, kind="stable")
                rank = np.empty(order.shape, dtype=np.uint16)
                np.put_along_axis(rank, order, np.arange(model.psi, dtype=np.uint16)[None, None, :], axis=2)
                for k in ks:
                    consistent[k] &= inside & (rank < k)
                del D, inside, order, rank

            for k in ks:
                counts[k] += np.sum(consistent[k], axis=(1, 2), dtype=np.uint32)
                del consistent[k]

        denom = float(model.psi * model.t)
        for k in ks:
            out[k][b0:b1] = 1.0 - counts[k].astype(np.float32) / denom

        if progress_label and (bi == 1 or bi == n_batches or bi % max(1, n_batches // 10) == 0):
            print(f"{progress_label}: batch {bi}/{n_batches}, rows {b1:,}/{n:,}")
        del counts
        gc.collect()

    for k in ks:
        if not np.isfinite(out[k]).all():
            raise ValueError(f"Invalid SCoNE scores for k={k}.")
        np.clip(out[k], 0.0, 1.0, out=out[k])
    return out


def reference_scores(views: Mapping[str, np.ndarray], model: SCoNEModel, k: int) -> np.ndarray:
    """Literal loop version of Eqs. (3)-(5), used only by the self-test."""
    n = next(iter(views.values())).shape[0]
    out = np.empty(n, dtype=np.float32)
    for x_idx in range(n):
        total = 0
        for e in range(model.t):
            common = set(range(model.psi))
            for name in VIEW_ORDER:
                x = views[name][x_idx]
                C = model.centers[name][e]
                d2 = np.sum((C - x[None, :]) ** 2, axis=1)
                order = np.argsort(d2, kind="stable")
                knn = set(order[:k].tolist())
                valid = {i for i in range(model.psi) if i in knn and d2[i] <= model.radii_sq[name][e, i]}
                common &= valid
            total += len(common)
        out[x_idx] = 1.0 - total / float(model.psi * model.t)
    return out


def self_test() -> None:
    """Exact optimized-vs-reference parity check, including one-hot ties."""
    rng = np.random.default_rng(20260812)
    n, psi, t = 37, 5, 4
    views, centers, radii = {}, {}, {}
    for name in VIEW_ORDER:
        d = VIEW_DIMS[name]
        X = rng.normal(size=(n, d)).astype(np.float32)
        if name == "landuse":
            cls = rng.integers(0, d, size=n)
            X[:] = 0.0
            X[np.arange(n), cls] = 1.0
        views[name] = X
        C = np.empty((t, psi, d), dtype=np.float32)
        for e in range(t):
            C[e] = X[rng.choice(n, size=psi, replace=False)]
        centers[name] = C
        radii[name] = pairwise_center_radii_sq(C)
    model = SCoNEModel(psi, t, 999, np.arange(t * psi).reshape(t, psi), centers, radii)
    model.validate()
    optimized = score_views_multi_k(views, model, [1, 3, 5], batch_size=11, target_distance_elements=50_000)
    for k in (1, 3, 5):
        ref = reference_scores(views, model, k)
        if not np.array_equal(optimized[k], ref):
            raise AssertionError(f"SCoNE optimized/reference mismatch for k={k}; max_abs={np.max(np.abs(optimized[k]-ref))}")
    print("SCoNE CORE SELF-TEST PASSED: optimized scorer exactly matches literal Eqs. (3)-(5).")
