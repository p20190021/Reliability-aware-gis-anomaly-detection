"""FINAL full-clean-center SCoNE baseline pipeline for the geospatial anomaly-detection paper.

Stages
------
preflight      Validate exact project inputs, preprocessing artifacts, full-clean center protocol, and test injections.
self-test      Verify optimized SCoNE code against a literal reference implementation.
benchmark      Estimate resource cost before the expensive grid search.
select-params  Build disjoint validation anomalies and choose one (psi,k) by mean validation ROC-AUC.
final          Freeze selected parameters, score clean + all existing test injections, export comparable results.

This is the reviewer-safe FINAL SCoNE protocol after the dedicated
normalization and cross-view-consistency diagnostics.

Frozen protocol before parameter search
---------------------------------------
* all six physical views;
* project robust preprocessing;
* representative centers sampled from the FULL CLEAN 4,290,958-row dataset;
* native SCoNE score S = 1 - Cbar;
* t = 200;
* exact stable-k nearest-neighbor rule;
* disjoint validation only for (psi,k) selection;
* no final-test labels used for selection.

All outputs are NEW. The completed 14_, 15_, and 16_ SCoNE result folders are
never write targets.

Place this file and scone_core.py directly in the project's Code folder.
"""
from __future__ import annotations
import os

import argparse
import gc
import json
import math
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from scone_core import (
    VIEW_DIMS,
    VIEW_ORDER,
    SCoNEModel,
    build_model,
    pairwise_center_radii_sq,
    exact_top_fraction_order,
    exact_top_k_order,
    score_views_multi_k,
    self_test,
)

# -----------------------------------------------------------------------------
# Project constants
# -----------------------------------------------------------------------------
DEFAULT_ROOT = Path(os.environ.get("GIS_ANOMALY_PROJECT_ROOT", Path.cwd())).resolve()
EXPECTED_N = 4_290_958
EXPECTED_D = 23
BASELINE_FIT_SIZE = 300_000
EXPECTED_BASELINE_SUBSET_SEED = 7
MODEL_SEED = 42
VALIDATION_SEED = 12_345
VALIDATION_NEGATIVE_SEED = 12_346
VALIDATION_RATIO = 0.0010
FINAL_T = 200
PSI_GRID = [2, 4, 8, 16, 32, 64, 128, 256]
K_GRID = [1, 3, 5, 7, 11, 21, 51, 101]
SCENARIOS = ["extreme", "noise_strong", "noise_moderate", "noise_mild", "inconsistent"]
FINAL_BUDGETS = [0.0010, 0.0025, 0.0050, 0.0100, 0.0200]
RATIO_CONFIG = {
    "0.05%": {"ratio": 0.0005, "folder": "Injected_Datasets_with_0.05%_Injection_Rate"},
    "0.1%":  {"ratio": 0.0010, "folder": "Injected_Datasets_with_0.1%_Injection_Rate"},
    "0.5%":  {"ratio": 0.0050, "folder": "Injected_Datasets_with_0.5%_Injection_Rate"},
}
SCENARIO_FILES = {
    "extreme": "X_injected_extreme.npy",
    "noise_strong": "X_injected_noise_strong.npy",
    "noise_moderate": "X_injected_noise_moderate.npy",
    "noise_mild": "X_injected_noise_mild.npy",
    "inconsistent": "X_injected_inconsistent.npy",
}
FEATURE_NAMES = [
    "elevation", "slope", "flow_acc", "profile_curv", "plan_curv",
    "depression_depth", "local_std", "elev_percentile", "elev_contrast",
    "slope_variability", "flow_contrast", "road_dist", "river_dist",
    "ndvi", "ndwi", "ndbi",
    "landuse_1", "landuse_2", "landuse_3", "landuse_4", "landuse_5", "landuse_6", "landuse_7",
]
VIEW_INDICES = {
    "terrain": [0, 1, 3, 4, 5],
    "hydrological": [2, 10, 12],
    "spatial": [6, 7, 8, 9],
    "surface": [13, 14, 15],
    "infrastructure": [11],
    "landuse": list(range(16, 23)),
}

# Prior completed folders are read-only context for this final pipeline.
PRIOR_SCoNE_FOLDERS = [
    "14_SCoNE_Baseline_Results_Seed_42",
    "15_SCoNE_Verification_Seed_42",
    "16_SCoNE_Raw_Consistency_Diagnostics_Seed_42",
]
REFERENCE_CENTER_POPULATION = "full_clean_4_290_958"


# -----------------------------------------------------------------------------
# Paths / serialization
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class Paths:
    root: Path
    clean: Path
    baseline: Path
    injected: Path
    output: Path
    validation: Path
    final: Path

    @classmethod
    def make(cls, root: Path) -> "Paths":
        output = root / "17_SCoNE_Final_FullClean_Seed_42"
        return cls(
            root=root,
            clean=root / "03_Proposed_Results_Seed_42",
            baseline=root / "02_Baseline_Results_Seed_42",
            injected=root / "03_Injected_Proposed_Results_Seed_42",
            output=output,
            validation=output / "validation",
            final=output / "final",
        )


def resolve_root(cli: str | None) -> Path:
    if cli:
        return Path(cli).expanduser().resolve()
    here = Path(__file__).resolve().parent
    return here if (here / "X.npy").is_file() else DEFAULT_ROOT


def require_file(path: Path, label: str = "") -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label or 'required file'}:\n{path}")
    return path


def require_dir(path: Path, label: str = "") -> Path:
    if not path.is_dir():
        raise FileNotFoundError(f"Missing {label or 'required folder'}:\n{path}")
    return path


def save_json(path: Path, obj: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True)


def load_json(path: Path) -> dict:
    with require_file(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def load_X(paths: Paths) -> np.ndarray:
    X = np.load(require_file(paths.root / "X.npy", "X.npy"), mmap_mode="r", allow_pickle=False)
    if X.shape != (EXPECTED_N, EXPECTED_D):
        raise ValueError(f"X.npy shape {X.shape}; expected {(EXPECTED_N, EXPECTED_D)}.")
    return X


# -----------------------------------------------------------------------------
# Exact current six-view preprocessing; no fitting occurs here.
# -----------------------------------------------------------------------------
@dataclass
class ViewPreprocessor:
    terrain_scaler: object
    hydro_scaler: object
    spatial_scaler: object
    surface_scaler: object
    infra_scaler: object
    terrain_bounds: dict
    spatial_bounds: dict

    @classmethod
    def load(cls, clean: Path) -> "ViewPreprocessor":
        files = {
            "terrain_scaler": clean / "terrain_scaler.pkl",
            "hydro_scaler": clean / "hydro_scaler.pkl",
            "spatial_scaler": clean / "spatial_scaler.pkl",
            "surface_scaler": clean / "surface_scaler.pkl",
            "infra_scaler": clean / "infra_scaler.pkl",
            "terrain_bounds": clean / "terrain_clip_bounds.pkl",
            "spatial_bounds": clean / "spatial_clip_bounds.pkl",
        }
        for name, path in files.items():
            require_file(path, name)
        return cls(
            joblib.load(files["terrain_scaler"]), joblib.load(files["hydro_scaler"]),
            joblib.load(files["spatial_scaler"]), joblib.load(files["surface_scaler"]),
            joblib.load(files["infra_scaler"]), joblib.load(files["terrain_bounds"]),
            joblib.load(files["spatial_bounds"]),
        )

    def transform(self, raw: np.ndarray) -> Dict[str, np.ndarray]:
        raw = np.asarray(raw)
        if raw.ndim != 2 or raw.shape[1] != 23 or not np.isfinite(raw).all():
            raise ValueError(f"Invalid raw feature block {raw.shape}.")

        terrain = raw[:, VIEW_INDICES["terrain"]].astype(np.float32, copy=True)
        for fname, pos in {"profile_curv": 2, "plan_curv": 3}.items():
            b = self.terrain_bounds[fname]
            terrain[:, pos] = np.tanh(np.clip(terrain[:, pos], float(b["lower"]), float(b["upper"])))
        terrain[:, 4] = np.log1p(np.clip(terrain[:, 4], 0.0, None))
        terrain = self.terrain_scaler.transform(terrain).astype(np.float32, copy=False)

        hydro = self.hydro_scaler.transform(raw[:, VIEW_INDICES["hydrological"]]).astype(np.float32, copy=False)

        spatial = raw[:, VIEW_INDICES["spatial"]].astype(np.float32, copy=True)
        for pos, fname in enumerate(["local_std", "elev_percentile", "elev_contrast", "slope_variability"]):
            b = self.spatial_bounds[fname]
            spatial[:, pos] = np.clip(spatial[:, pos], float(b["lower"]), float(b["upper"]))
        spatial = self.spatial_scaler.transform(spatial).astype(np.float32, copy=False)

        surface = self.surface_scaler.transform(raw[:, VIEW_INDICES["surface"]]).astype(np.float32, copy=False)
        infra = self.infra_scaler.transform(raw[:, VIEW_INDICES["infrastructure"]]).astype(np.float32, copy=False)
        landuse = raw[:, VIEW_INDICES["landuse"]].astype(np.float32, copy=True)

        views = {
            "terrain": terrain, "hydrological": hydro, "spatial": spatial,
            "surface": surface, "infrastructure": infra, "landuse": landuse,
        }
        n = raw.shape[0]
        for name in VIEW_ORDER:
            if views[name].shape != (n, VIEW_DIMS[name]) or not np.isfinite(views[name]).all():
                raise ValueError(f"Invalid processed {name} view {views[name].shape}.")
        return views



# -----------------------------------------------------------------------------
# Full-clean SCoNE center sampling.
# -----------------------------------------------------------------------------
def full_clean_center_indices(n: int, psi: int, t: int, seed: int) -> np.ndarray:
    """Draw t independent SCoNE subsamples directly from the full clean dataset.

    This is mathematically equivalent to sampling from pool=np.arange(n), but
    avoids constructing/sorting a 4.29-million-entry reference-pool array.
    """
    if psi < 2 or psi > n:
        raise ValueError(f"Invalid psi={psi} for n={n}.")
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(psi), 2601]))
    out = np.empty((t, psi), dtype=np.int64)
    for e in range(t):
        out[e] = rng.choice(n, size=psi, replace=False)
    return out


def build_model_full_clean(
    raw_X: np.ndarray,
    preprocessor,
    psi: int,
    t: int,
    seed: int,
) -> SCoNEModel:
    """Build exact SCoNE model with centers sampled from all clean rows."""
    center_idx = full_clean_center_indices(raw_X.shape[0], psi, t, seed)
    raw_centers = np.asarray(raw_X[center_idx.reshape(-1)], dtype=np.float32)
    flat_views = preprocessor.transform(raw_centers)

    centers, radii = {}, {}
    for name in VIEW_ORDER:
        arr = flat_views[name].reshape(t, psi, VIEW_DIMS[name])
        centers[name] = arr
        radii[name] = pairwise_center_radii_sq(arr)

    model = SCoNEModel(
        psi=psi,
        t=t,
        seed=seed,
        center_global_indices=center_idx,
        centers=centers,
        radii_sq=radii,
    )
    model.validate()
    del raw_centers, flat_views
    gc.collect()
    return model


def full_clean_reference_audit(n: int) -> dict:
    return {
        "reference_population": REFERENCE_CENTER_POPULATION,
        "reference_population_count": int(n),
        "sampling_without_replacement_within_each_ensemble": True,
        "clean_only": True,
        "model_seed": MODEL_SEED,
    }

# -----------------------------------------------------------------------------
# Baseline fitting-pool audit
# -----------------------------------------------------------------------------
def baseline_pool(paths: Paths, n: int) -> Tuple[np.ndarray, dict]:
    path = require_file(paths.baseline / "Baseline_sample_indices.npy", "baseline fitting indices")
    saved = np.load(path, allow_pickle=False).astype(np.int64, copy=False).reshape(-1)
    if saved.size != BASELINE_FIT_SIZE or np.unique(saved).size != saved.size:
        raise ValueError("Baseline fitting-index file is not the expected unique 300,000-row pool.")
    if saved.min() < 0 or saved.max() >= n:
        raise IndexError("Baseline fitting indices are out of range.")

    match = {}
    for seed in (7, 42):
        regen = np.random.default_rng(seed).choice(n, size=BASELINE_FIT_SIZE, replace=False).astype(np.int64)
        match[seed] = bool(np.array_equal(saved, regen))
    audit = {
        "file": str(path), "count": int(saved.size),
        "matches_seed_7": match[7], "matches_seed_42": match[42],
        "expected_manuscript_subset_seed": EXPECTED_BASELINE_SUBSET_SEED,
    }
    if not match[EXPECTED_BASELINE_SUBSET_SEED]:
        raise ValueError(
            "BASELINE FITTING-POOL AUDIT FAILED. The saved Baseline_sample_indices.npy "
            f"does not match the manuscript-reported subset seed {EXPECTED_BASELINE_SUBSET_SEED}.\n"
            f"Audit: {audit}\nDo not bypass this: first confirm the authoritative current baseline pool."
        )
    return saved, audit


# -----------------------------------------------------------------------------
# Preflight
# -----------------------------------------------------------------------------
def preflight(paths: Paths) -> dict:
    # Hard safety: this final script may never write into any prior SCoNE folder.
    if paths.output.name in PRIOR_SCoNE_FOLDERS:
        raise RuntimeError("Unsafe output path: prior SCoNE results would be overwritten.")
    require_dir(paths.root, "project root")
    require_dir(paths.clean, "clean proposed results")
    require_dir(paths.baseline, "baseline results")
    require_dir(paths.injected, "injected proposed root")
    X = load_X(paths)
    _ = ViewPreprocessor.load(paths.clean)

    flat = [i for name in VIEW_ORDER for i in VIEW_INDICES[name]]
    if sorted(flat) != list(range(23)) or len(set(flat)) != 23:
        raise RuntimeError("Six-view feature partition is not disjoint/complete.")

    pool, audit = baseline_pool(paths, X.shape[0])
    test_meta = []
    for ratio_label, cfg in RATIO_CONFIG.items():
        rdir = require_dir(paths.injected / cfg["folder"], ratio_label)
        idx = np.load(require_file(rdir / "inject_indices.npy"), allow_pickle=False).reshape(-1)
        expected = int(math.floor(cfg["ratio"] * X.shape[0]))
        if idx.size != expected:
            raise ValueError(f"{ratio_label}: injected count {idx.size}, expected {expected}.")
        for scenario in SCENARIOS:
            arr = np.load(require_file(rdir / SCENARIO_FILES[scenario]), mmap_mode="r", allow_pickle=False)
            if arr.shape != X.shape:
                raise ValueError(f"{ratio_label}/{scenario} shape {arr.shape}, expected {X.shape}.")
        test_meta.append({"ratio": ratio_label, "count": int(idx.size), "folder": str(rdir)})

    info = {
        "n": EXPECTED_N, "d": EXPECTED_D, "features": FEATURE_NAMES,
        "view_indices": VIEW_INDICES, "view_dims": VIEW_DIMS,
        "legacy_conventional_baseline_pool_audit": audit,
        "scone_center_sampling_population": REFERENCE_CENTER_POPULATION,
        "scone_center_sampling_count": int(X.shape[0]),
        "test_injections": test_meta,
        "paper_grid": {"psi": PSI_GRID, "k": K_GRID, "t_final": FINAL_T},
        "final_center_population": REFERENCE_CENTER_POPULATION,
        "prior_scone_folders_read_only": PRIOR_SCoNE_FOLDERS,
        "validation_seed": VALIDATION_SEED, "validation_ratio": VALIDATION_RATIO,
        "python": platform.python_version(), "platform": platform.platform(),
    }
    paths.output.mkdir(parents=True, exist_ok=True)
    save_json(paths.output / "SCoNE_Preflight_Audit.json", info)
    print("=" * 80)
    print("SCoNE PREFLIGHT PASSED")
    print("=" * 80)
    print(f"N={EXPECTED_N:,}, D={EXPECTED_D}")
    print(f"SCoNE center-sampling population: FULL CLEAN dataset ({EXPECTED_N:,} rows)")
    print(f"Legacy conventional-baseline fitting pool audited separately: {pool.size:,} rows")
    print(json.dumps(audit, indent=2))
    print("Views:")
    for name in VIEW_ORDER:
        print(f"  {name:15s}: {VIEW_DIMS[name]} features -> {VIEW_INDICES[name]}")
    return info


# -----------------------------------------------------------------------------
# Validation definition: disjoint from ALL existing final-test injected pixels.
# -----------------------------------------------------------------------------
@dataclass
class ValidationDef:
    inject: np.ndarray
    negatives: np.ndarray
    eval_idx: np.ndarray
    labels: np.ndarray
    strong: np.ndarray
    moderate: np.ndarray
    mild: np.ndarray
    donor: np.ndarray


def test_index_union(paths: Paths) -> np.ndarray:
    chunks = []
    for cfg in RATIO_CONFIG.values():
        chunks.append(np.load(paths.injected / cfg["folder"] / "inject_indices.npy", allow_pickle=False).astype(np.int64))
    return np.unique(np.concatenate(chunks))


def validation_definition(paths: Paths, negative_count: int) -> ValidationDef:
    paths.validation.mkdir(parents=True, exist_ok=True)
    npz = paths.validation / "validation_definition.npz"
    meta_path = paths.validation / "validation_definition.json"
    if npz.is_file() and meta_path.is_file():
        meta = load_json(meta_path)
        if int(meta["negative_count"]) != int(negative_count):
            raise ValueError(
                f"Existing SCoNE validation definition has {meta['negative_count']} negatives, "
                f"but this run requests {negative_count}. Keep the same value for reproducibility."
            )
        d = np.load(npz, allow_pickle=False)
        return ValidationDef(d["inject"], d["negatives"], d["eval_idx"], d["labels"], d["strong"], d["moderate"], d["mild"], d["donor"])

    n = EXPECTED_N
    excluded_test = test_index_union(paths)
    allowed_mask = np.ones(n, dtype=bool); allowed_mask[excluded_test] = False
    allowed = np.flatnonzero(allowed_mask); del allowed_mask
    n_inject = int(math.floor(VALIDATION_RATIO * n))

    rng = np.random.default_rng(VALIDATION_SEED)
    inject = np.sort(rng.choice(allowed, size=n_inject, replace=False)).astype(np.int64)
    excluded = np.union1d(excluded_test, inject)
    neg_mask = np.ones(n, dtype=bool); neg_mask[excluded] = False
    neg_pool = np.flatnonzero(neg_mask); del neg_mask
    nrng = np.random.default_rng(VALIDATION_NEGATIVE_SEED)
    negatives = np.sort(nrng.choice(neg_pool, size=negative_count, replace=False)).astype(np.int64)
    eval_idx = np.sort(np.concatenate([inject, negatives])).astype(np.int64)
    labels = np.isin(eval_idx, inject, assume_unique=True).astype(np.uint8)

    # Same numeric mechanisms as the current test-injection notebook.
    strong = rng.normal(5.0, 2.0, size=(n_inject, 16)).astype(np.float32)
    moderate = rng.normal(0.0, 0.5, size=(n_inject, 16)).astype(np.float32)
    mild = rng.normal(0.0, 0.2, size=(n_inject, 16)).astype(np.float32)
    donor = rng.permutation(inject).astype(np.int64)

    if np.intersect1d(inject, excluded_test).size:
        raise RuntimeError("Validation injections overlap final-test injections.")
    np.savez_compressed(npz, inject=inject, negatives=negatives, eval_idx=eval_idx, labels=labels,
                        strong=strong, moderate=moderate, mild=mild, donor=donor)
    save_json(meta_path, {
        "validation_seed": VALIDATION_SEED,
        "validation_negative_seed": VALIDATION_NEGATIVE_SEED,
        "validation_ratio": VALIDATION_RATIO,
        "injected_count": int(n_inject), "negative_count": int(negative_count),
        "eval_count": int(eval_idx.size), "disjoint_from_all_final_test_injections": True,
        "scenarios": SCENARIOS,
    })
    return ValidationDef(inject, negatives, eval_idx, labels, strong, moderate, mild, donor)


def validation_scenario_rows(X: np.ndarray, vd: ValidationDef, scenario: str) -> np.ndarray:
    raw = np.asarray(X[vd.eval_idx], dtype=np.float32).copy()
    loc = np.searchsorted(vd.eval_idx, vd.inject)
    if not np.array_equal(vd.eval_idx[loc], vd.inject):
        raise RuntimeError("Validation index mapping failed.")
    if scenario == "extreme":
        raw[loc, 0] *= 3.0; raw[loc, 1] *= 4.0; raw[loc, 2] *= 5.0; raw[loc, 10] *= 3.0
    elif scenario == "noise_strong":
        raw[np.ix_(loc, np.arange(16))] += vd.strong
    elif scenario == "noise_moderate":
        raw[np.ix_(loc, np.arange(16))] += vd.moderate
    elif scenario == "noise_mild":
        raw[np.ix_(loc, np.arange(16))] += vd.mild
    elif scenario == "inconsistent":
        donor_raw = np.asarray(X[vd.donor], dtype=np.float32)
        for cols in ([0,1,3,4,5], [2,10], [11,12]):
            raw[np.ix_(loc, cols)] = donor_raw[:, cols]
        raw[loc, 13] = donor_raw[:, 15]  # donor NDBI -> injected NDVI
    else:
        raise ValueError(scenario)
    if not np.isfinite(raw).all():
        raise ValueError(f"Validation scenario {scenario} produced invalid data.")
    return raw


# -----------------------------------------------------------------------------
# Memory-safe full-matrix scoring
# -----------------------------------------------------------------------------
def score_raw_matrix(
    raw_matrix: np.ndarray,
    prep: ViewPreprocessor,
    model: SCoNEModel,
    k: int,
    *,
    preprocess_batch: int,
    score_batch: int,
    target_elements: int,
    label: str,
) -> np.ndarray:
    n = raw_matrix.shape[0]
    out = np.empty(n, dtype=np.float32)
    total_blocks = math.ceil(n / preprocess_batch)
    t0 = time.perf_counter()
    for bi, p0 in enumerate(range(0, n, preprocess_batch), 1):
        p1 = min(p0 + preprocess_batch, n)
        raw = np.asarray(raw_matrix[p0:p1], dtype=np.float32)
        views = prep.transform(raw)
        scores = score_views_multi_k(
            views, model, [k], batch_size=min(score_batch, p1-p0),
            target_distance_elements=target_elements,
            progress_label=f"{label} block {bi}/{total_blocks}",
        )[k]
        out[p0:p1] = scores
        elapsed = time.perf_counter() - t0
        print(f"{label}: raw block {bi}/{total_blocks}; {p1:,}/{n:,}; {p1/max(elapsed,1e-9):,.0f} rows/s")
        del raw, views, scores
        gc.collect()
    return out


def score_diag(scores: np.ndarray) -> dict:
    sel = exact_top_fraction_order(scores, 0.02)
    boundary = float(scores[sel[-1]])
    return {
        "n": int(scores.size), "min": float(scores.min()), "p01": float(np.percentile(scores,1)),
        "median": float(np.median(scores)), "mean": float(np.mean(scores, dtype=np.float64)),
        "p99": float(np.percentile(scores,99)), "max": float(scores.max()),
        "fraction_score_1": float(np.mean(scores == 1.0)), "fraction_score_0": float(np.mean(scores == 0.0)),
        "unique_scores": int(np.unique(scores).size), "top2_count": int(sel.size),
        "top2_boundary": boundary, "top2_global_boundary_ties": int(np.count_nonzero(scores == boundary)),
    }


# -----------------------------------------------------------------------------
# Benchmark
# -----------------------------------------------------------------------------
def benchmark(paths: Paths, rows: int, t_bench: int, score_batch: int, target_elements: int) -> None:
    preflight(paths); self_test()
    X = load_X(paths); prep = ViewPreprocessor.load(paths.clean); _, _ = baseline_pool(paths, X.shape[0])
    qidx = np.sort(np.random.default_rng(260826).choice(X.shape[0], size=rows, replace=False))
    views = prep.transform(np.asarray(X[qidx], dtype=np.float32))
    result = []
    for psi in (4, 16, 64, 256):
        k = max(v for v in K_GRID if v <= psi)
        model = build_model_full_clean(X, prep, psi, t_bench, MODEL_SEED)
        start = time.perf_counter()
        s = score_views_multi_k(views, model, [k], batch_size=score_batch, target_distance_elements=target_elements,
                                progress_label=f"benchmark psi={psi}")[k]
        sec = time.perf_counter() - start
        rps = rows / sec
        est = EXPECTED_N / rps * (FINAL_T / t_bench)
        row = {"psi":psi,"k":k,"benchmark_t":t_bench,"rows":rows,"seconds":sec,
               "rows_per_second":rps,"estimated_full_hours_t200":est/3600.0,"unique_scores":int(np.unique(s).size)}
        print(json.dumps(row, indent=2)); result.append(row)
        del model, s; gc.collect()
    pd.DataFrame(result).to_csv(paths.output / "SCoNE_Runtime_Benchmark.csv", index=False)
    print("Benchmark complete. Review SCoNE_Runtime_Benchmark.csv before parameter search.")


# -----------------------------------------------------------------------------
# Parameter selection: t=200, five scenarios, disjoint validation labels only.
# Checkpointed by psi.
# -----------------------------------------------------------------------------
def valid_ks(psi: int) -> List[int]:
    return [k for k in K_GRID if k <= psi]


def select_params(paths: Paths, negative_count: int, score_batch: int, target_elements: int) -> None:
    preflight(paths); self_test()
    X = load_X(paths); prep = ViewPreprocessor.load(paths.clean); _, baseline_pool_audit = baseline_pool(paths, X.shape[0]); reference_audit = full_clean_reference_audit(X.shape[0])
    vd = validation_definition(paths, negative_count)

    # Audit that the newly generated validation definition is exactly the same
    # disjoint validation split used in the completed SCoNE work, when those
    # prior files are available.
    prior_val = paths.root / "14_SCoNE_Baseline_Results_Seed_42" / "validation" / "validation_definition.npz"
    prior_validation_match = None
    if prior_val.is_file():
        old = np.load(prior_val, allow_pickle=False)
        prior_validation_match = bool(
            np.array_equal(vd.inject, old["inject"]) and
            np.array_equal(vd.negatives, old["negatives"]) and
            np.array_equal(vd.eval_idx, old["eval_idx"]) and
            np.array_equal(vd.labels, old["labels"])
        )
        if not prior_validation_match:
            raise RuntimeError(
                "The new validation definition does not exactly match the "
                "completed disjoint SCoNE validation split. Stop and inspect."
            )
        print("Validation audit: exact match to completed 14_ validation split.")
    checkpoints = paths.validation / "parameter_search_checkpoints"; checkpoints.mkdir(parents=True, exist_ok=True)
    print(f"Validation positives={vd.inject.size:,}, negatives={vd.negatives.size:,}; final test labels are untouched.")
    all_rows = []

    for psi in PSI_GRID:
        cp = checkpoints / f"psi_{psi:03d}.csv"
        if cp.is_file():
            print(f"psi={psi}: loading checkpoint")
            all_rows.extend(pd.read_csv(cp).to_dict("records")); continue
        ks = valid_ks(psi)
        print("\n" + "="*80 + f"\nPARAMETER SEARCH psi={psi}, ks={ks}, t={FINAL_T}\n" + "="*80)
        model = build_model_full_clean(X, prep, psi, FINAL_T, MODEL_SEED)
        aucs = {k:{} for k in ks}; diagnostics = {k:{} for k in ks}
        for scenario in SCENARIOS:
            raw = validation_scenario_rows(X, vd, scenario)
            views = prep.transform(raw)
            scores = score_views_multi_k(views, model, ks, batch_size=score_batch,
                                          target_distance_elements=target_elements,
                                          progress_label=f"psi={psi}/{scenario}")
            for k in ks:
                auc = float(roc_auc_score(vd.labels, scores[k]))
                aucs[k][scenario] = auc
                diagnostics[k][scenario] = {"score1_fraction":float(np.mean(scores[k]==1.0)),
                                             "unique_scores":int(np.unique(scores[k]).size)}
                print(f"psi={psi:3d} k={k:3d} {scenario:16s} AUC={auc:.6f} score1={diagnostics[k][scenario]['score1_fraction']:.4f}")
            del raw, views, scores; gc.collect()
        rows = []
        for k in ks:
            vals = [aucs[k][s] for s in SCENARIOS]
            row = {"psi":psi,"k":k,"t":FINAL_T,"model_seed":MODEL_SEED,"validation_seed":VALIDATION_SEED,
                   "mean_auc":float(np.mean(vals)),"min_auc":float(np.min(vals)),"max_auc":float(np.max(vals)),"std_auc":float(np.std(vals)),
                   "validation_positives":int(vd.inject.size),"validation_negatives":int(vd.negatives.size)}
            for s in SCENARIOS:
                row[f"auc_{s}"] = aucs[k][s]
                row[f"score1_fraction_{s}"] = diagnostics[k][s]["score1_fraction"]
                row[f"unique_scores_{s}"] = diagnostics[k][s]["unique_scores"]
            rows.append(row)
        pdf = pd.DataFrame(rows); pdf.to_csv(cp, index=False); all_rows.extend(rows)
        print(f"Checkpoint saved: {cp}")
        del model; gc.collect()

    df = pd.DataFrame(all_rows).sort_values(["psi","k"]).reset_index(drop=True)
    ranked = df.sort_values(["mean_auc","min_auc","psi","k"], ascending=[False,False,True,True], kind="mergesort").reset_index(drop=True)
    best = ranked.iloc[0]
    selected = {
        "psi":int(best.psi), "k":int(best.k), "t":FINAL_T, "model_seed":MODEL_SEED,
        "mean_validation_auc":float(best.mean_auc), "min_validation_auc":float(best.min_auc),
        "selection_metric":"mean ROC-AUC across five disjoint validation corruption scenarios",
        "tie_break":"higher minimum scenario AUC, then smaller psi, then smaller k",
        "validation_seed":VALIDATION_SEED,"validation_ratio":VALIDATION_RATIO,
        "validation_positive_count":int(vd.inject.size),"validation_negative_count":int(vd.negatives.size),
        "final_test_labels_used_for_selection":False,
        "preprocessing":"project robust preprocessing",
        "all_six_views_retained":True,
        "knn_tie_rule":"stable exact-k",
        "ranking_score":"native SCoNE 1-Cbar",
        "center_sampling_population":REFERENCE_CENTER_POPULATION,
        "reference_population_audit":reference_audit,
        "legacy_baseline_pool_audit_for_comparability":baseline_pool_audit,
        "validation_definition_exact_match_to_completed_14_run":prior_validation_match,
    }
    df.to_csv(paths.validation / "SCoNE_Parameter_Search_All.csv", index=False)
    ranked.to_csv(paths.validation / "SCoNE_Parameter_Search_Ranked.csv", index=False)
    save_json(paths.validation / "SCoNE_Selected_Parameters.json", selected)
    print("\nSELECTED SCoNE PARAMETERS\n" + json.dumps(selected, indent=2))
    print("\nTop 10:\n" + ranked[["psi","k","mean_auc","min_auc","std_auc"]].head(10).to_string(index=False))


# -----------------------------------------------------------------------------
# Final test evaluation
# -----------------------------------------------------------------------------
def tie_aware_selection_metrics(
    scores: np.ndarray,
    budget_fraction: float,
    injected: np.ndarray | None = None,
) -> dict:
    """
    Characterize an exact top-k selection when the kth score lies in a tie.

    SCoNE can produce a highly discrete score distribution.  A deterministic
    index-based exact top-k is retained for reproducibility/compatibility, but
    when only part of a boundary-tie group can be selected it is not uniquely
    implied by SCoNE itself.

    For injected data, this function therefore also reports:
      * minimum possible injected recovery across all valid tie resolutions,
      * maximum possible injected recovery,
      * expected recovery under uniform random selection within the tie group.

    These quantities depend only on the SCoNE scores and the known evaluation
    labels; no secondary detector or post-hoc ranking signal is introduced.
    """
    scores = np.asarray(scores).reshape(-1)
    n = scores.size
    k = int(math.ceil(float(budget_fraction) * n))

    if not 1 <= k <= n:
        raise ValueError(f"Invalid top-k={k} for n={n:,}.")

    # kth largest score, without imposing an arbitrary order among equals.
    boundary = float(np.partition(scores, n - k)[n - k])

    above_mask = scores > boundary
    tie_mask = scores == boundary

    n_above = int(np.count_nonzero(above_mask))
    n_tie = int(np.count_nonzero(tie_mask))
    need_from_tie = int(k - n_above)

    if not (0 <= need_from_tie <= n_tie):
        raise RuntimeError(
            "Tie-aware top-k accounting failed: "
            f"k={k}, above={n_above}, tie={n_tie}, need={need_from_tie}."
        )

    result = {
        "Selected_Count": k,
        "Raw_Score_Threshold": boundary,
        "Strictly_Above_Threshold_Count": n_above,
        "Boundary_Tie_Count": n_tie,
        "Need_From_Boundary_Tie": need_from_tie,
        "Boundary_Tie_Selection_Fraction": (
            float(need_from_tie / n_tie) if n_tie else 0.0
        ),
        "Boundary_Tie_Affects_Selection": bool(
            0 < need_from_tie < n_tie
        ),
        "Selection_Unique_From_SCoNE_Score": bool(
            need_from_tie == 0 or need_from_tie == n_tie
        ),
    }

    if injected is not None:
        injected = np.asarray(injected, dtype=np.int64).reshape(-1)

        injected_scores = scores[injected]
        inj_above = int(np.count_nonzero(injected_scores > boundary))
        inj_tie = int(np.count_nonzero(injected_scores == boundary))

        noninj_tie = n_tie - inj_tie

        # Worst case: choose as many non-injected tied points as possible.
        min_from_tie = max(
            0,
            need_from_tie - noninj_tie,
        )

        # Best case: choose as many injected tied points as possible.
        max_from_tie = min(
            need_from_tie,
            inj_tie,
        )

        # Expected value if the required boundary positions are drawn uniformly
        # from all tied observations.
        expected_from_tie = (
            need_from_tie * inj_tie / n_tie
            if n_tie > 0
            else 0.0
        )

        m = int(injected.size)

        result.update(
            {
                "Injected_Count": m,
                "Injected_Strictly_Above_Threshold": inj_above,
                "Injected_In_Boundary_Tie": inj_tie,
                "TieAware_Min_Detected_Injected": int(
                    inj_above + min_from_tie
                ),
                "TieAware_Max_Detected_Injected": int(
                    inj_above + max_from_tie
                ),
                "TieAware_Expected_Detected_Injected": float(
                    inj_above + expected_from_tie
                ),
                "TieAware_Min_Detection_Rate_Percent": float(
                    100.0 * (inj_above + min_from_tie) / m
                ),
                "TieAware_Max_Detection_Rate_Percent": float(
                    100.0 * (inj_above + max_from_tie) / m
                ),
                "TieAware_Expected_Detection_Rate_Percent": float(
                    100.0 * (inj_above + expected_from_tie) / m
                ),
            }
        )

    return result


def injected_budget_rows(
    scores: np.ndarray,
    injected: np.ndarray,
    ratio: str,
    scenario: str,
) -> List[dict]:
    """
    Produce both the legacy deterministic exact-top-k result and tie-aware
    recovery metrics at every paper budget.
    """
    ranking = exact_top_k_order(
        scores,
        int(math.ceil(max(FINAL_BUDGETS) * scores.size)),
    )

    rows = []

    for b in FINAL_BUDGETS:
        k = int(math.ceil(b * scores.size))
        sel = ranking[:k]

        deterministic_detected = int(
            np.isin(
                injected,
                sel,
                assume_unique=True,
            ).sum()
        )

        tie_metrics = tie_aware_selection_metrics(
            scores=scores,
            budget_fraction=b,
            injected=injected,
        )

        row = {
            "Injection_Ratio": ratio,
            "Scenario": scenario,
            "Budget_Fraction": b,
            "Budget_Percent": 100.0 * b,
            "Detected_Injected": deterministic_detected,
            "Detection_Rate_Percent": (
                100.0
                * deterministic_detected
                / injected.size
            ),
            # Explicit aliases make it impossible to confuse deterministic
            # tie-breaking with the method-implied expected recovery.
            "Deterministic_Detected_Injected": deterministic_detected,
            "Deterministic_Detection_Rate_Percent": (
                100.0
                * deterministic_detected
                / injected.size
            ),
        }

        row.update(tie_metrics)
        rows.append(row)

    return rows


def final_run(paths: Paths, preprocess_batch: int, score_batch: int, target_elements: int) -> None:
    preflight(paths); self_test()
    selected_path = paths.validation / "SCoNE_Selected_Parameters.json"
    if not selected_path.is_file():
        raise FileNotFoundError("Frozen SCoNE parameters are missing. Run --stage select-params first.")
    selected = load_json(selected_path)
    psi, k, t = int(selected["psi"]), int(selected["k"]), int(selected["t"])
    if t != FINAL_T:
        raise ValueError(f"Final t must be {FINAL_T}; found {t}.")

    X = load_X(paths); prep = ViewPreprocessor.load(paths.clean); _, baseline_pool_audit = baseline_pool(paths, X.shape[0]); reference_audit = full_clean_reference_audit(X.shape[0])
    model = build_model_full_clean(X, prep, psi, t, MODEL_SEED)
    paths.final.mkdir(parents=True, exist_ok=True)
    np.save(paths.final / "SCoNE_center_global_indices.npy", model.center_global_indices)
    radius_diag = {name:{"zero_radius_fraction":float(np.mean(model.radii_sq[name]==0.0)),
                         "median_radius_sq":float(np.median(model.radii_sq[name])),
                         "max_radius_sq":float(np.max(model.radii_sq[name]))} for name in VIEW_ORDER}
    save_json(paths.final / "SCoNE_Final_Model_Audit.json", {
        "psi":psi,"k":k,"t":t,"model_seed":MODEL_SEED,"reference_population_audit":reference_audit, "legacy_baseline_pool_audit_for_comparability":baseline_pool_audit,
        "all_centers_are_valid_clean_global_indices":bool((model.center_global_indices >= 0).all() and (model.center_global_indices < X.shape[0]).all()),
        "radius_diagnostics":radius_diag,
    })

    print(f"\nFINAL CLEAN SCoNE: psi={psi}, k={k}, t={t}")
    clean_scores = score_raw_matrix(X, prep, model, k, preprocess_batch=preprocess_batch, score_batch=score_batch,
                                    target_elements=target_elements, label="clean")
    cdir = paths.final / "clean"; cdir.mkdir(parents=True, exist_ok=True)
    np.save(cdir / "SCoNE_scores_raw.npy", clean_scores)
    p1, p99 = float(np.percentile(clean_scores,1)), float(np.percentile(clean_scores,99))
    if p99 <= p1:
        raise ValueError("Clean SCoNE p1/p99 are degenerate; inspect score diagnostics/parameters.")
    clean_norm = ((np.clip(clean_scores,p1,p99)-p1)/(p99-p1)).astype(np.float32)
    np.save(cdir / "SCoNE_scores_norm.npy", clean_norm)
    np.savez(cdir / "SCoNE_normalization_params.npz", p1=np.float64(p1), p99=np.float64(p99))
    top2 = exact_top_fraction_order(clean_scores,0.02); mask = np.zeros(clean_scores.size,dtype=bool); mask[top2]=True
    np.save(cdir / "SCoNE_selected_indices_top2.npy", top2.astype(np.int64)); np.save(cdir / "SCoNE_anomaly_labels_top2.npy",mask)
    cdiag = score_diag(clean_scores); cdiag.update({"psi":psi,"k":k,"t":t,"ranking":"native raw 1-Cbar","normalized_used_for_ranking":False})

    clean_top2_tie = tie_aware_selection_metrics(
        scores=clean_scores,
        budget_fraction=0.02,
        injected=None,
    )
    cdiag["top2_tie_diagnostics"] = clean_top2_tie

    save_json(cdir / "SCoNE_clean_score_diagnostics.json",cdiag)
    save_json(cdir / "SCoNE_clean_top2_tie_diagnostics.json", clean_top2_tie)
    print(json.dumps(cdiag,indent=2))

    if clean_top2_tie["Boundary_Tie_Affects_Selection"]:
        print(
            "\nIMPORTANT: the clean top-2% set cuts through a SCoNE score tie. "
            "The saved deterministic indices are retained only for exact "
            "reproducibility. Do not interpret clean-set Jaccard, spatial "
            "morphology, or flood-context statistics from that one arbitrary "
            "tie resolution until tie handling is explicitly accounted for."
        )

    del clean_norm, mask; gc.collect()

    eval_rows, diag_rows, auc_rows = [], [], []
    for ratio_label,cfg in RATIO_CONFIG.items():
        rdir = paths.injected / cfg["folder"]
        inject_idx = np.load(rdir / "inject_indices.npy",allow_pickle=False).astype(np.int64)
        for scenario in SCENARIOS:
            print("\n" + "="*80 + f"\nFINAL TEST {ratio_label}/{scenario}\n" + "="*80)
            raw = np.load(rdir / SCENARIO_FILES[scenario],mmap_mode="r",allow_pickle=False)
            scores = score_raw_matrix(raw,prep,model,k,preprocess_batch=preprocess_batch,score_batch=score_batch,
                                      target_elements=target_elements,label=f"{ratio_label}/{scenario}")
            odir = paths.final / "injected" / cfg["folder"] / scenario; odir.mkdir(parents=True,exist_ok=True)
            np.save(odir / "SCoNE_scores_raw.npy",scores)
            sel2 = exact_top_fraction_order(scores,0.02); np.save(odir / "SCoNE_selected_indices_top2.npy",sel2.astype(np.int64))
            norm = ((np.clip(scores,p1,p99)-p1)/(p99-p1)).astype(np.float32)
            np.save(odir / "SCoNE_scores_norm_clean_bounds.npy",norm)
            eval_rows.extend(
                injected_budget_rows(
                    scores,
                    inject_idx,
                    ratio_label,
                    scenario,
                )
            )

            # ROC-AUC is retained as a tie-robust diagnostic because the
            # original SCoNE study evaluates AUC and SCoNE may generate very
            # coarse score distributions on these six heterogeneous views.
            auc_labels = np.zeros(scores.size, dtype=np.uint8)
            auc_labels[inject_idx] = 1
            full_auc = float(roc_auc_score(auc_labels, scores))
            del auc_labels

            auc_rows.append(
                {
                    "Injection_Ratio": ratio_label,
                    "Scenario": scenario,
                    "Injected_Count": int(inject_idx.size),
                    "ROC_AUC": full_auc,
                }
            )

            d = score_diag(scores)
            d.update(
                {
                    "Injection_Ratio": ratio_label,
                    "Scenario": scenario,
                    "psi": psi,
                    "k": k,
                    "t": t,
                    "ROC_AUC": full_auc,
                }
            )
            diag_rows.append(d)
            save_json(odir / "SCoNE_score_diagnostics.json",d)
            del raw,scores,sel2,norm; gc.collect()

    df = pd.DataFrame(eval_rows)
    dg = pd.DataFrame(diag_rows)
    auc_df = pd.DataFrame(auc_rows)

    df.to_csv(
        paths.final / "SCoNE_Injected_Detection_Results.csv",
        index=False,
    )
    dg.to_csv(
        paths.final / "SCoNE_Injected_Score_Diagnostics.csv",
        index=False,
    )
    auc_df.to_csv(
        paths.final / "SCoNE_Injected_AUC_Results.csv",
        index=False,
    )

    ref = df[df.Injection_Ratio=="0.1%"].copy()
    ref.to_csv(
        paths.final / "SCoNE_Reference_0.1pct_Results.csv",
        index=False,
    )

    means = df.groupby(
        ["Injection_Ratio","Budget_Fraction","Budget_Percent"],
        as_index=False,
    ).agg(
        Mean_Detection_Rate_Percent=(
            "Detection_Rate_Percent",
            "mean",
        ),
        Min_Detection_Rate_Percent=(
            "Detection_Rate_Percent",
            "min",
        ),
        Max_Detection_Rate_Percent=(
            "Detection_Rate_Percent",
            "max",
        ),
        Mean_TieAware_Expected_Detection_Rate_Percent=(
            "TieAware_Expected_Detection_Rate_Percent",
            "mean",
        ),
        Mean_TieAware_Min_Detection_Rate_Percent=(
            "TieAware_Min_Detection_Rate_Percent",
            "mean",
        ),
        Mean_TieAware_Max_Detection_Rate_Percent=(
            "TieAware_Max_Detection_Rate_Percent",
            "mean",
        ),
        Mean_Boundary_Tie_Count=(
            "Boundary_Tie_Count",
            "mean",
        ),
    )

    means.to_csv(
        paths.final / "SCoNE_Injected_Mean_Results.csv",
        index=False,
    )
    save_json(paths.final / "SCoNE_Final_Evaluation_Audit.json", {
        "baseline":"SCoNE","implementation":"independent Eqs. (2)-(5) reimplementation",
        "final_protocol_version":"full-clean-center reviewer-safe protocol",
        "center_sampling_population":REFERENCE_CENTER_POPULATION,
        "psi":psi,"k":k,"t":t,"model_seed":MODEL_SEED,
        "hyperparameters_selected_on_disjoint_validation":True,"final_test_labels_used_for_selection":False,
        "input_views":VIEW_ORDER,
        "all_six_views_retained":True,
        "preprocessing":"project robust preprocessing frozen from clean proposed run",
        "center_sampling_population":REFERENCE_CENTER_POPULATION,
        "proposed_internal_scores_used":False,
        "raw_native_score_used_for_ranking":True,"score_orientation":"higher anomalous; S=1-Cbar",
        "tie_handling":{
            "deterministic_exact_top_k_saved_for_reproducibility":True,
            "deterministic_tie_break":"existing exact-top-k lower-index boundary resolution",
            "primary_interpretation_when_boundary_is_nonunique":"tie-aware expected/min/max recovery",
            "tie_aware_expected_rule":"uniform random selection within the kth-score boundary tie",
            "secondary_ranking_signal_added":False,
        },
        "budgets":FINAL_BUDGETS,"test_ratios":list(RATIO_CONFIG),"scenarios":SCENARIOS,"clean_diagnostics":cdiag,
    })
    print("\nSCoNE FINAL FULL-CLEAN-CENTER EVALUATION COMPLETE")
    print("Frozen parameters:\n" + json.dumps(selected,indent=2))
    print(
        "Reference 0.1% results "
        "(deterministic and tie-aware expected):\n"
        + ref[
            [
                "Scenario",
                "Budget_Percent",
                "Deterministic_Detection_Rate_Percent",
                "TieAware_Expected_Detection_Rate_Percent",
                "TieAware_Min_Detection_Rate_Percent",
                "TieAware_Max_Detection_Rate_Percent",
                "Boundary_Tie_Count",
            ]
        ].to_string(index=False)
    )
    print("\nFull-test AUC diagnostics:\n" + auc_df.to_string(index=False))
    print(f"Outputs: {paths.final}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parser() -> argparse.ArgumentParser:
    p=argparse.ArgumentParser(description="SCoNE AAAI-2026 baseline pipeline")
    p.add_argument("--root",default=None)
    p.add_argument("--stage",required=True,choices=["preflight","self-test","benchmark","select-params","final"])
    p.add_argument("--validation-negatives",type=int,default=20_000,
                   help="Fixed clean negative sample for validation AUC; all 4,290 validation positives are included.")
    p.add_argument("--score-batch-size",type=int,default=8_000)
    p.add_argument("--raw-preprocess-batch-size",type=int,default=100_000)
    p.add_argument("--target-distance-elements",type=int,default=4_000_000,
                   help="Memory control for distance tensors. Lower if RAM pressure occurs.")
    p.add_argument("--benchmark-rows",type=int,default=5_000)
    p.add_argument("--benchmark-t",type=int,default=10)
    return p


def main() -> None:
    a=parser().parse_args(); paths=Paths.make(resolve_root(a.root))
    if a.stage=="preflight": preflight(paths)
    elif a.stage=="self-test": self_test()
    elif a.stage=="benchmark": benchmark(paths,a.benchmark_rows,a.benchmark_t,a.score_batch_size,a.target_distance_elements)
    elif a.stage=="select-params": select_params(paths,a.validation_negatives,a.score_batch_size,a.target_distance_elements)
    elif a.stage=="final": final_run(paths,a.raw_preprocess_batch_size,a.score_batch_size,a.target_distance_elements)

if __name__=="__main__":
    main()
