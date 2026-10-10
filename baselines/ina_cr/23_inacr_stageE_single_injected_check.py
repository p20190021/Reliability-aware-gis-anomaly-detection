"""INA-CR Stage E — single frozen injected-evaluation check.

Runs ONE already-generated final-test case only:
    injection ratio: 0.1%
    scenario: Extreme

Purpose
-------
Validate the final patched-score evaluation path before running all 15
ratio/scenario combinations.

This script DOES NOT:
* retrain INA-CR;
* refit any scaler, min/max range, latent mean, or component range;
* regenerate injections;
* alter any hyperparameter;
* normalize final INA-CR scores before ranking.

It reuses the completed full-clean score vector and re-scores ONLY the rows in
the authoritative inject_indices.npy.  For injected rows that belong to the
100k clean reference population, the corresponding clean reference row is
excluded from that row's kNN query, matching the leave-self-out identity rule
used by the completed full-clean audit.

Final candidate budgets use the project's established rule:
    k = ceil(budget * 4,290,958)
giving 4,291; 10,728; 21,455; 42,910; 85,820.
"""
from __future__ import annotations
import os

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors

DEFAULT_ROOT = Path(os.environ.get("GIS_ANOMALY_PROJECT_ROOT", Path.cwd())).resolve()

EXPECTED_N = 4_290_958
EXPECTED_D = 23
REFERENCE_N = 100_000
K_NEIGHBORS = 6
LATENT_DIM = 32
RELEASED_BATCH_SIZE_COEFFICIENT = 100
QUERY_BATCH = 2048
NUMERICAL_TOLERANCE = 1e-5

RATIO_LABEL = "0.1%"
RATIO_VALUE = 0.0010
RATIO_FOLDER = "Injected_Datasets_with_0.1%_Injection_Rate"
SCENARIO = "extreme"
SCENARIO_FILE = "X_injected_extreme.npy"

BUDGETS = [0.0010, 0.0025, 0.0050, 0.0100, 0.0200]
WEIGHTS = (1.0, 0.25, 1.0, 0.25)

VIEW_ORDER = [
    "terrain", "hydrological", "spatial",
    "surface", "infrastructure", "landuse"
]
VIEW_INDICES = {
    "terrain": [0, 1, 3, 4, 5],
    "hydrological": [2, 10, 12],
    "spatial": [6, 7, 8, 9],
    "surface": [13, 14, 15],
    "infrastructure": [11],
    "landuse": list(range(16, 23)),
}
VIEW_DIMS = {k: len(v) for k, v in VIEW_INDICES.items()}

VIEW_SLICES = {}
_cursor = 0
for _v in VIEW_ORDER:
    VIEW_SLICES[_v] = slice(_cursor, _cursor + VIEW_DIMS[_v])
    _cursor += VIEW_DIMS[_v]
assert _cursor == EXPECTED_D

COMPONENTS = ["reconstruction", "neighborhood", "prediction", "instance"]


def require_file(path: Path, label: str = "") -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label or 'required file'}:\n{path}")
    return path


def resolve_root(cli_root: str | None) -> Path:
    if cli_root:
        return Path(cli_root).expanduser().resolve()
    try:
        here = Path(__file__).resolve().parent
    except NameError:
        here = Path.cwd().resolve()
    if (here / "X.npy").is_file():
        return here
    if (here.parent / "X.npy").is_file():
        return here.parent
    return DEFAULT_ROOT


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
    def load(cls, clean_dir: Path):
        return cls(
            joblib.load(require_file(clean_dir / "terrain_scaler.pkl")),
            joblib.load(require_file(clean_dir / "hydro_scaler.pkl")),
            joblib.load(require_file(clean_dir / "spatial_scaler.pkl")),
            joblib.load(require_file(clean_dir / "surface_scaler.pkl")),
            joblib.load(require_file(clean_dir / "infra_scaler.pkl")),
            joblib.load(require_file(clean_dir / "terrain_clip_bounds.pkl")),
            joblib.load(require_file(clean_dir / "spatial_clip_bounds.pkl")),
        )

    def transform_views(self, raw: np.ndarray) -> Dict[str, np.ndarray]:
        raw = np.asarray(raw)
        if raw.ndim != 2 or raw.shape[1] != EXPECTED_D or not np.isfinite(raw).all():
            raise ValueError(f"Invalid raw feature block {raw.shape}")

        terrain = raw[:, VIEW_INDICES["terrain"]].astype(np.float32, copy=True)
        for fname, pos in {"profile_curv": 2, "plan_curv": 3}.items():
            b = self.terrain_bounds[fname]
            terrain[:, pos] = np.tanh(
                np.clip(terrain[:, pos], float(b["lower"]), float(b["upper"]))
            )
        terrain[:, 4] = np.log1p(np.clip(terrain[:, 4], 0.0, None))
        terrain = self.terrain_scaler.transform(terrain).astype(np.float32, copy=False)

        hydro = self.hydro_scaler.transform(
            raw[:, VIEW_INDICES["hydrological"]]
        ).astype(np.float32, copy=False)

        spatial = raw[:, VIEW_INDICES["spatial"]].astype(np.float32, copy=True)
        for pos, fname in enumerate(
            ["local_std", "elev_percentile", "elev_contrast", "slope_variability"]
        ):
            b = self.spatial_bounds[fname]
            spatial[:, pos] = np.clip(
                spatial[:, pos], float(b["lower"]), float(b["upper"])
            )
        spatial = self.spatial_scaler.transform(spatial).astype(np.float32, copy=False)

        surface = self.surface_scaler.transform(
            raw[:, VIEW_INDICES["surface"]]
        ).astype(np.float32, copy=False)
        infra = self.infra_scaler.transform(
            raw[:, VIEW_INDICES["infrastructure"]]
        ).astype(np.float32, copy=False)
        landuse = raw[:, VIEW_INDICES["landuse"]].astype(np.float32, copy=True)

        return {
            "terrain": terrain,
            "hydrological": hydro,
            "spatial": spatial,
            "surface": surface,
            "infrastructure": infra,
            "landuse": landuse,
        }


def load_input_ranges(npz_path: Path):
    """Load the exact per-view arrays saved by final training."""
    with np.load(require_file(npz_path), allow_pickle=False) as z:
        mins = {}
        maxs = {}
        for view in VIEW_ORDER:
            mink = f"{view}_min"
            maxk = f"{view}_max"
            if mink not in z.files or maxk not in z.files:
                raise KeyError(
                    f"Expected {mink}/{maxk} in {npz_path.name}; keys={z.files}"
                )
            mins[view] = np.asarray(z[mink], dtype=np.float32).reshape(-1)
            maxs[view] = np.asarray(z[maxk], dtype=np.float32).reshape(-1)
            if mins[view].size != VIEW_DIMS[view] or maxs[view].size != VIEW_DIMS[view]:
                raise ValueError(f"Bad frozen input range shape for {view}")
            if np.any(maxs[view] < mins[view]):
                raise ValueError(f"Invalid frozen input range for {view}")
    return mins, maxs


def preprocess_to_x01(raw, project_pre, mins, maxs):
    pviews = project_pre.transform_views(raw)
    out = {}
    for view in VIEW_ORDER:
        lo = mins[view]
        hi = maxs[view]
        den = hi - lo
        safe = np.where(den > 0, den, 1.0).astype(np.float32)
        z = (pviews[view] - lo) / safe
        if np.any(den == 0):
            z[:, den == 0] = 0.0
        out[view] = np.clip(z, 0.0, 1.0).astype(np.float32, copy=False)
    x01 = np.concatenate([out[v] for v in VIEW_ORDER], axis=1).astype(np.float32)
    if x01.shape != (len(raw), EXPECTED_D) or not np.isfinite(x01).all():
        raise ValueError("Invalid INA-CR query representation")
    return out, x01


class GeneratorEncode(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(int(d), LATENT_DIM), nn.Tanh())
    def forward(self, x):
        return self.encoder(x)


class GeneratorDecode(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.decoder = nn.Sequential(nn.Linear(LATENT_DIM, int(d)), nn.Sigmoid())
    def forward(self, x):
        return self.decoder(x)


class Prediction(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(LATENT_DIM, LATENT_DIM), nn.Tanh())
    def forward(self, x):
        return self.encoder(x)


def load_models(path: Path, device):
    payload = torch.load(require_file(path), map_location=device, weights_only=False)
    if "model_state" not in payload:
        raise KeyError("Final model is missing model_state")
    state = payload["model_state"]
    enc = [GeneratorEncode(VIEW_DIMS[v]).to(device) for v in VIEW_ORDER]
    dec = [GeneratorDecode(VIEW_DIMS[v]).to(device) for v in VIEW_ORDER]
    pred = [Prediction().to(device) for _ in VIEW_ORDER]
    for m, sd in zip(enc, state["encoders"]):
        m.load_state_dict(sd); m.eval()
    for m, sd in zip(dec, state["decoders"]):
        m.load_state_dict(sd); m.eval()
    for m, sd in zip(pred, state["predictors"]):
        m.load_state_dict(sd); m.eval()
    return enc, dec, pred


def load_latent_means(path: Path):
    with np.load(require_file(path), allow_pickle=False) as z:
        out = {}
        for v in VIEW_ORDER:
            if v not in z.files:
                raise KeyError(f"Missing latent mean {v}; keys={z.files}")
            out[v] = np.asarray(z[v], dtype=np.float32).reshape(LATENT_DIM)
    return out


def load_component_ranges(path: Path):
    df = pd.read_csv(require_file(path))
    out = {}
    for v in VIEW_ORDER:
        row = df[df["view"].astype(str).str.lower() == v]
        if len(row) != 1:
            raise ValueError(f"Expected one frozen range row for {v}")
        row = row.iloc[0]
        out[v] = {}
        for c in COMPONENTS:
            lo = float(row[f"{c}_clean_min"])
            hi = float(row[f"{c}_clean_max"])
            if not np.isfinite([lo, hi]).all() or hi <= lo:
                raise ValueError(f"Bad frozen {c} range for {v}")
            out[v][c] = (lo, hi)
    return out


def score_injected_rows(
    query_views,
    q_neighbors,
    reference_x01,
    enc,
    dec,
    pred,
    latent_means,
    component_ranges,
    device,
):
    ref_views = {v: reference_x01[:, VIEW_SLICES[v]] for v in VIEW_ORDER}
    mean_t = {
        v: torch.from_numpy(latent_means[v]).to(device).view(1, -1)
        for v in VIEW_ORDER
    }
    n = q_neighbors.shape[0]
    agg = {c: np.zeros(n, dtype=np.float64) for c in COMPONENTS}

    with torch.no_grad():
        for vi, v in enumerate(VIEW_ORDER):
            raw_parts = {c: [] for c in COMPONENTS}
            for s in range(0, n, QUERY_BATCH):
                e = min(s + QUERY_BATCH, n)
                q = [
                    torch.from_numpy(query_views[name][s:e]).to(device)
                    for name in VIEW_ORDER
                ]
                zall = [enc[j](q[j]) for j in range(len(VIEW_ORDER))]
                zv = zall[vi]

                rec = torch.sum((dec[vi](zv) - q[vi]) ** 2, dim=1)
                mean_query = torch.stack(zall, dim=0).mean(dim=0)
                prd = torch.sum((mean_query - pred[vi](zv)) ** 2, dim=1) / LATENT_DIM

                ng_np = ref_views[v][q_neighbors[s:e]]
                ng = torch.from_numpy(np.asarray(ng_np, dtype=np.float32)).to(device)
                zng = enc[vi](ng).mean(dim=1)

                ins = torch.zeros(e - s, device=device)
                nei = torch.zeros(e - s, device=device)

                for j, zj in enumerate(zall):
                    aa_i = torch.exp(F.cosine_similarity(zv, zj, dim=1))
                    bb_i = RELEASED_BATCH_SIZE_COEFFICIENT * torch.exp(
                        F.cosine_similarity(zv, mean_t[VIEW_ORDER[j]], dim=1)
                    )
                    ins += -torch.log(aa_i / bb_i)

                    aa_n = torch.exp(F.cosine_similarity(zng, zj, dim=1))
                    bb_n = (
                        RELEASED_BATCH_SIZE_COEFFICIENT
                        * torch.exp(
                            F.cosine_similarity(zng, mean_t[VIEW_ORDER[j]], dim=1)
                        )
                        + torch.exp(
                            F.cosine_similarity(zng, mean_t[v], dim=1)
                        )
                    )
                    nei += -torch.log(aa_n / bb_n)

                ins /= len(VIEW_ORDER) * LATENT_DIM
                nei /= len(VIEW_ORDER) * LATENT_DIM

                raw_parts["reconstruction"].append(rec.cpu().numpy())
                raw_parts["neighborhood"].append(nei.cpu().numpy())
                raw_parts["prediction"].append(prd.cpu().numpy())
                raw_parts["instance"].append(ins.cpu().numpy())

            for c in COMPONENTS:
                arr = np.concatenate(raw_parts[c]).astype(np.float64, copy=False)
                lo, hi = component_ranges[v][c]
                # Frozen clean range, NO query clipping.
                agg[c] += (arr - lo) / (hi - lo)

    total = (
        WEIGHTS[0] * agg["reconstruction"]
        + WEIGHTS[1] * agg["neighborhood"]
        + WEIGHTS[2] * agg["prediction"]
        + WEIGHTS[3] * agg["instance"]
    )
    if not np.isfinite(total).all():
        raise FloatingPointError("Non-finite injected INA-CR score")
    return total.astype(np.float32)


def exact_ranking(scores):
    """Deterministic global ordering: high score first; global index breaks exact ties."""
    n = len(scores)
    return np.lexsort(
        (np.arange(n, dtype=np.int64), -np.asarray(scores, dtype=np.float64))
    )


def main(cli_root=None):
    root = resolve_root(cli_root)
    train = root / "21_INACR_Final_Training_100k"
    audit = root / "22_INACR_Full_Clean_Audit"
    injected_dir = (
        root / "03_Injected_Proposed_Results_Seed_42" / RATIO_FOLDER
    )
    project_clean = root / "03_Proposed_Results_Seed_42"
    out = root / "23_INACR_StageE_0p1_Extreme_Check"
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("INA-CR STAGE E — SINGLE FROZEN INJECTED CHECK")
    print(f"Ratio={RATIO_LABEL} | Scenario=Extreme")
    print("No retraining, refitting, or injection regeneration.")
    print("=" * 80)

    # Hard prerequisite: completed clean audit.
    audit_summary_path = require_file(audit / "INA_CR_Full_Clean_Audit_Summary.json")
    with audit_summary_path.open("r", encoding="utf-8") as f:
        audit_summary = json.load(f)
    if audit_summary.get("status") != "PASS":
        raise RuntimeError("Full-clean audit did not PASS.")
    if int(audit_summary.get("n_scored", -1)) != EXPECTED_N:
        raise RuntimeError("Full-clean audit sample count mismatch.")
    ref_repro_diff = float(
        audit_summary["max_abs_difference_reference_members_vs_frozen_training_scores"]
    )
    if ref_repro_diff >= NUMERICAL_TOLERANCE:
        raise RuntimeError("Clean-audit reference reproduction check failed.")

    clean_score_path = require_file(
        audit / "INA_CR_clean_full_scores_float32.dat",
        "full-clean score memmap",
    )
    clean_scores = np.memmap(
        clean_score_path, dtype=np.float32, mode="r", shape=(EXPECTED_N,)
    )
    if not np.isfinite(clean_scores).all():
        raise RuntimeError("Full-clean score vector contains non-finite values.")

    X_clean = np.load(
        require_file(root / "X.npy"), mmap_mode="r", allow_pickle=False
    )
    X_inj = np.load(
        require_file(injected_dir / SCENARIO_FILE),
        mmap_mode="r",
        allow_pickle=False,
    )
    if X_clean.shape != (EXPECTED_N, EXPECTED_D) or X_inj.shape != X_clean.shape:
        raise ValueError("Clean/injected matrix shape mismatch.")

    inject_idx = np.load(
        require_file(injected_dir / "inject_indices.npy"),
        allow_pickle=False,
    ).astype(np.int64).reshape(-1)
    if np.unique(inject_idx).size != inject_idx.size:
        raise ValueError("inject_indices contains duplicates")
    if np.any(inject_idx < 0) or np.any(inject_idx >= EXPECTED_N):
        raise IndexError("inject_indices out of range")

    expected_injected = int(math.floor(RATIO_VALUE * EXPECTED_N))
    if inject_idx.size != expected_injected:
        raise ValueError(
            f"Injected count={inject_idx.size}, expected floor({RATIO_VALUE}*N)="
            f"{expected_injected}"
        )

    labels_path = injected_dir / "injection_ground_truth_labels.npy"
    label_match = None
    if labels_path.is_file():
        labels = np.load(labels_path, mmap_mode="r", allow_pickle=False)
        gt_idx = np.flatnonzero(np.asarray(labels) > 0)
        label_match = bool(np.array_equal(np.sort(inject_idx), gt_idx))
        if not label_match:
            raise RuntimeError("inject_indices and ground-truth label file disagree.")

    # Verify sampled non-injected rows are byte/value identical.
    rng = np.random.default_rng(20260815)
    mask = np.ones(EXPECTED_N, dtype=bool)
    mask[inject_idx] = False
    candidates = np.flatnonzero(mask)
    check_idx = rng.choice(candidates, size=min(4096, candidates.size), replace=False)
    noninj_equal = bool(
        np.array_equal(
            np.asarray(X_clean[check_idx]),
            np.asarray(X_inj[check_idx]),
        )
    )
    del mask, candidates
    if not noninj_equal:
        raise RuntimeError("Sampled non-injected raw rows differ from clean data.")

    # Frozen artifacts.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, torch.cuda.get_device_name(0) if device.type == "cuda" else "")

    project_pre = ViewPreprocessor.load(project_clean)
    mins, maxs = load_input_ranges(
        train / "INA_CR_input_minmax01_ranges.npz"
    )
    enc, dec, pred = load_models(
        train / "INA_CR_final_model_100k_200epochs.pt", device
    )
    latent_means = load_latent_means(
        train / "INA_CR_frozen_reference_latent_means.npz"
    )
    component_ranges = load_component_ranges(
        train / "INA_CR_frozen_component_ranges.csv"
    )
    ref_x01 = np.load(
        require_file(train / "INA_CR_reference_X01_100k.npy"),
        mmap_mode="r",
        allow_pickle=False,
    ).astype(np.float32, copy=False)
    ref_idx = np.load(
        require_file(train / "INA_CR_reference_indices_100k.npy"),
        allow_pickle=False,
    ).astype(np.int64)

    if ref_x01.shape != (REFERENCE_N, EXPECTED_D) or ref_idx.shape != (REFERENCE_N,):
        raise ValueError("Frozen reference artifact shape mismatch")

    # Preprocess ONLY injected rows.
    raw_injected_rows = np.asarray(X_inj[inject_idx], dtype=np.float32)
    query_views, query_x01 = preprocess_to_x01(
        raw_injected_rows, project_pre, mins, maxs
    )

    # Frozen 100k kNN reference. Query k+1 so identity-self can be removed.
    print(f"Querying neighbors for {inject_idx.size:,} injected rows only...")
    t_knn = time.perf_counter()
    knn = NearestNeighbors(
        n_neighbors=K_NEIGHBORS + 1,
        algorithm="ball_tree",
        metric="euclidean",
    ).fit(ref_x01)
    candidate_ng = knn.kneighbors(query_x01, return_distance=False).astype(np.int64)

    global_to_ref = np.full(EXPECTED_N, -1, dtype=np.int32)
    global_to_ref[ref_idx] = np.arange(REFERENCE_N, dtype=np.int32)
    own_ref = global_to_ref[inject_idx]
    n_injected_in_reference = int(np.count_nonzero(own_ref >= 0))

    q_neighbors = np.empty((inject_idx.size, K_NEIGHBORS), dtype=np.int64)
    n_own_reference_actually_removed = 0
    for i in range(inject_idx.size):
        cand = candidate_ng[i]
        own = int(own_ref[i])
        if own >= 0 and np.any(cand == own):
            cand = cand[cand != own]
            n_own_reference_actually_removed += 1
        q_neighbors[i] = cand[:K_NEIGHBORS]
    knn_seconds = time.perf_counter() - t_knn

    print("Scoring injected rows only...")
    t_score = time.perf_counter()
    injected_scores = score_injected_rows(
        query_views=query_views,
        q_neighbors=q_neighbors,
        reference_x01=ref_x01,
        enc=enc,
        dec=dec,
        pred=pred,
        latent_means=latent_means,
        component_ranges=component_ranges,
        device=device,
    )
    model_seconds = time.perf_counter() - t_score

    np.save(out / "INA_CR_injected_rows_scores.npy", injected_scores)
    np.save(out / "inject_indices.npy", inject_idx)

    clean_at_injected = np.asarray(clean_scores[inject_idx], dtype=np.float32)
    delta = injected_scores.astype(np.float64) - clean_at_injected.astype(np.float64)
    delta_tol = 1e-7
    increased = int(np.count_nonzero(delta > delta_tol))
    decreased = int(np.count_nonzero(delta < -delta_tol))
    unchanged = int(delta.size - increased - decreased)

    # Patch ONLY injected rows into the already-audited clean vector.
    patched = np.asarray(clean_scores, dtype=np.float32).copy()
    patched[inject_idx] = injected_scores

    # One deterministic ranking; slice it at each CEIL budget.
    ranking = exact_ranking(patched)
    max_k_expected = int(math.ceil(max(BUDGETS) * EXPECTED_N))
    if max_k_expected != 85820:
        raise RuntimeError(f"Unexpected top-2% count {max_k_expected}")

    budget_rows = []
    for b in BUDGETS:
        k = int(math.ceil(b * EXPECTED_N))
        selected = ranking[:k]
        detected = int(np.isin(inject_idx, selected, assume_unique=False).sum())
        threshold = float(patched[selected[-1]])
        n_gt = int(np.count_nonzero(patched > threshold))
        n_eq = int(np.count_nonzero(patched == threshold))
        budget_rows.append({
            "Injection_Ratio": RATIO_LABEL,
            "Scenario": "Extreme",
            "Budget_Fraction": b,
            "Budget_Percent": 100.0 * b,
            "Selected_Count": k,
            "Total_Injected": int(inject_idx.size),
            "Detected_Injected": detected,
            "Detection_Rate_Percent": 100.0 * detected / inject_idx.size,
            "Raw_Final_Score_Threshold": threshold,
            "Scores_Greater_Than_Threshold": n_gt,
            "Scores_Equal_To_Threshold": n_eq,
        })

    with (out / "INA_CR_StageE_Budget_Detection_Rates.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        w = csv.DictWriter(f, fieldnames=list(budget_rows[0].keys()))
        w.writeheader()
        w.writerows(budget_rows)

    np.save(out / "INA_CR_0p1_extreme_full_patched_scores.npy", patched)

    changed_raw_rows = int(
        np.count_nonzero(
            np.any(
                np.asarray(X_clean[inject_idx], dtype=np.float32)
                != raw_injected_rows,
                axis=1,
            )
        )
    )

    summary = {
        "stage": "INA_CR_stageE_single_frozen_injected_check",
        "status": "PASS",
        "ratio": RATIO_LABEL,
        "ratio_value": RATIO_VALUE,
        "scenario": "Extreme",
        "total_samples": EXPECTED_N,
        "total_injected": int(inject_idx.size),
        "expected_injected_floor_ratio_times_n": expected_injected,
        "changed_raw_rows_among_inject_indices": changed_raw_rows,
        "ground_truth_file_present": bool(labels_path.is_file()),
        "ground_truth_matches_inject_indices": label_match,
        "sampled_noninjected_rows_checked": int(check_idx.size),
        "sampled_noninjected_rows_equal_clean": noninj_equal,
        "clean_audit_reference_reproduction_max_abs_diff": ref_repro_diff,
        "clean_audit_tolerance": NUMERICAL_TOLERANCE,
        "injected_rows_in_100k_reference_population": n_injected_in_reference,
        "own_clean_reference_neighbor_actually_removed": n_own_reference_actually_removed,
        "identity_self_exclusion_rule": (
            "For an injected query whose global index is in the frozen 100k reference, "
            "exclude that query's corresponding clean reference row if it appears among "
            "the k+1 nearest candidates, then retain k=6."
        ),
        "knn_seconds": knn_seconds,
        "model_scoring_seconds": model_seconds,
        "clean_injected_positions_score_mean": float(clean_at_injected.mean()),
        "injected_score_mean": float(injected_scores.mean()),
        "paired_delta_mean": float(delta.mean()),
        "paired_delta_median": float(np.median(delta)),
        "paired_score_increased_count": increased,
        "paired_score_unchanged_count": unchanged,
        "paired_score_decreased_count": decreased,
        "budget_rule": "ceil(budget_fraction * 4,290,958)",
        "budget_selected_counts": {
            f"{100*b:g}%": int(math.ceil(b * EXPECTED_N))
            for b in BUDGETS
        },
        "budgets": budget_rows,
        "ranking_score": "frozen native/final INA-CR total score; higher is more anomalous",
        "ranking_normalization": "none",
        "patching": (
            "All non-injected positions reuse the completed full-clean score exactly; "
            "only authoritative injected positions are rescored."
        ),
        "injected_data_used_for_training_or_tuning": False,
        "note": (
            "This is a technical single-case final-evaluation check. The frozen model/"
            "protocol is not changed based on detection performance."
        ),
    }

    with (out / "INA_CR_StageE_Summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print("INA-CR STAGE E RESULT: PASS")
    print(f"Injected rows: {inject_idx.size:,}")
    print(
        f"Injected rows also in 100k reference: {n_injected_in_reference:,} | "
        f"own clean reference removed when present: {n_own_reference_actually_removed:,}"
    )
    print(f"kNN time: {knn_seconds:.2f} s | model scoring: {model_seconds:.2f} s")
    print(
        f"Paired score change: increased={increased:,}, unchanged={unchanged:,}, "
        f"decreased={decreased:,}; mean delta={delta.mean():.6f}"
    )
    for row in budget_rows:
        print(
            f"Top {row['Budget_Percent']:>4.2f}% | "
            f"k={row['Selected_Count']:,} | "
            f"detected={row['Detected_Injected']:,}/{row['Total_Injected']:,} | "
            f"DR={row['Detection_Rate_Percent']:.2f}%"
        )
    print("No retraining/refitting or injection regeneration was performed.")
    print("=" * 80)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, default=None)
    args = p.parse_args()
    main(args.root)
