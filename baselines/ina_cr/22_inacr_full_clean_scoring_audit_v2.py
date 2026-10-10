"""INA-CR full-clean scoring/runtime audit.

This script loads the frozen final INA-CR model/artifacts from:
    21_INACR_Final_Training_100k

It then scores ALL 4,290,958 clean pixels using the Stage-D2-validated
query-independent scoring rule.

No injected data or final-test labels are read.

Important details
-----------------
* kNN reference is the frozen clean 100k population.
* For a query row that itself belongs to the 100k reference set, its own
  clean-reference row is excluded from its kNN list (leave-self-out), matching
  the clean-reference score-range fitting logic.
* Query preprocessing uses the already frozen project preprocessing plus the
  saved INA-CR [0,1] input ranges.
* Component normalization uses the saved clean-reference component ranges.
* Query score extrapolation is NOT clipped.
* Scores are written incrementally to a memmap and the script is resumable.
"""
from __future__ import annotations
import os

import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors

DEFAULT_ROOT = Path(os.environ.get("GIS_ANOMALY_PROJECT_ROOT", Path.cwd())).resolve()

EXPECTED_N = 4_290_958
EXPECTED_D = 23
FINAL_TRAIN_N = 100_000

LATENT_DIM = 32
BATCH_SIZE_COEFF = 100
K_NEIBS = 6
WEIGHTS = (1.0, 0.25, 1.0, 0.25)

VIEW_ORDER = ["terrain", "hydrological", "spatial", "surface", "infrastructure", "landuse"]
VIEW_DIMS = {"terrain": 5, "hydrological": 3, "spatial": 4, "surface": 3, "infrastructure": 1, "landuse": 7}
VIEW_INDICES = {
    "terrain": [0, 1, 3, 4, 5],
    "hydrological": [2, 10, 12],
    "spatial": [6, 7, 8, 9],
    "surface": [13, 14, 15],
    "infrastructure": [11],
    "landuse": list(range(16, 23)),
}
COMPONENTS = ["reconstruction", "neighborhood", "prediction", "instance"]
BUDGETS = [0.001, 0.0025, 0.005, 0.01, 0.02]


def require_file(path: Path, label=""):
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label or 'required file'}:\n{path}")
    return path


def resolve_root(cli_root):
    if cli_root:
        return Path(cli_root).expanduser().resolve()
    here = Path(__file__).resolve().parent
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
    def load(cls, clean_dir):
        return cls(
            joblib.load(require_file(clean_dir / "terrain_scaler.pkl")),
            joblib.load(require_file(clean_dir / "hydro_scaler.pkl")),
            joblib.load(require_file(clean_dir / "spatial_scaler.pkl")),
            joblib.load(require_file(clean_dir / "surface_scaler.pkl")),
            joblib.load(require_file(clean_dir / "infra_scaler.pkl")),
            joblib.load(require_file(clean_dir / "terrain_clip_bounds.pkl")),
            joblib.load(require_file(clean_dir / "spatial_clip_bounds.pkl")),
        )

    def transform(self, raw):
        raw = np.asarray(raw)
        if raw.ndim != 2 or raw.shape[1] != EXPECTED_D or not np.isfinite(raw).all():
            raise ValueError(f"Invalid raw block {raw.shape}")

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

        return {
            "terrain": terrain,
            "hydrological": hydro,
            "spatial": spatial,
            "surface": surface,
            "infrastructure": infra,
            "landuse": landuse,
        }


class GeneratorEncode(nn.Module):
    def __init__(self, data_dim):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(int(data_dim), LATENT_DIM), nn.Tanh())
    def forward(self, x):
        return self.encoder(x)


class GeneratorDecode(nn.Module):
    def __init__(self, data_dim):
        super().__init__()
        self.decoder = nn.Sequential(nn.Linear(LATENT_DIM, int(data_dim)), nn.Sigmoid())
    def forward(self, x):
        return self.decoder(x)


class Prediction(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(LATENT_DIM, LATENT_DIM), nn.Tanh())
    def forward(self, x):
        return self.encoder(x)


def load_models(model_path, device):
    payload = torch.load(model_path, map_location=device, weights_only=False)
    enc = [GeneratorEncode(VIEW_DIMS[n]).to(device) for n in VIEW_ORDER]
    dec = [GeneratorDecode(VIEW_DIMS[n]).to(device) for n in VIEW_ORDER]
    pred = [Prediction().to(device) for _ in VIEW_ORDER]
    for m, sd in zip(enc, payload["model_state"]["encoders"]):
        m.load_state_dict(sd); m.eval()
    for m, sd in zip(dec, payload["model_state"]["decoders"]):
        m.load_state_dict(sd); m.eval()
    for m, sd in zip(pred, payload["model_state"]["predictors"]):
        m.load_state_dict(sd); m.eval()
    return enc, dec, pred


def load_input_ranges(path):
    z = np.load(path, allow_pickle=False)
    mins = {n: np.asarray(z[f"{n}_min"], dtype=np.float32) for n in VIEW_ORDER}
    maxs = {n: np.asarray(z[f"{n}_max"], dtype=np.float32) for n in VIEW_ORDER}
    return mins, maxs


def apply_input_ranges(views, mins, maxs):
    out = {}
    for n in VIEW_ORDER:
        lo, hi = mins[n], maxs[n]
        den = hi - lo
        safe = np.where(den > 0, den, 1.0).astype(np.float32)
        z = (views[n].astype(np.float32, copy=False) - lo) / safe
        if np.any(den == 0):
            z[:, den == 0] = 0.0
        out[n] = np.clip(z, 0.0, 1.0).astype(np.float32, copy=False)
    return out


def load_component_ranges(path):
    import pandas as pd
    df = pd.read_csv(path)
    by_name = {str(r["view"]): r for _, r in df.iterrows()}
    ranges = []
    for n in VIEW_ORDER:
        r = by_name[n]
        ranges.append({
            c: (float(r[f"{c}_clean_min"]), float(r[f"{c}_clean_max"]))
            for c in COMPONENTS
        })
    return ranges


def score_chunk(
    query_views,
    query_neighbor_idx,
    ref_views,
    encoders,
    decoders,
    predictors,
    ref_latent_means,
    ranges,
    device,
    torch_chunk=2048,
):
    n = len(next(iter(query_views.values())))
    means_t = [torch.from_numpy(ref_latent_means[i]).to(device) for i in range(len(VIEW_ORDER))]
    agg = {c: np.zeros(n, dtype=np.float64) for c in COMPONENTS}

    with torch.no_grad():
        for v, name in enumerate(VIEW_ORDER):
            raw_parts = {c: [] for c in COMPONENTS}

            for s in range(0, n, torch_chunk):
                e = min(s + torch_chunk, n)
                q = [torch.from_numpy(query_views[nm][s:e]).to(device) for nm in VIEW_ORDER]
                zall = [encoders[j](q[j]) for j in range(len(VIEW_ORDER))]
                zv = zall[v]

                rec = torch.sum((decoders[v](zv) - q[v]) ** 2, dim=1)
                mean_query = torch.stack(zall, dim=0).mean(dim=0)
                prd = torch.sum((mean_query - predictors[v](zv)) ** 2, dim=1) / LATENT_DIM

                ng_np = ref_views[name][query_neighbor_idx[s:e]]
                ng = torch.from_numpy(np.asarray(ng_np, dtype=np.float32)).to(device)
                zng = encoders[v](ng).mean(dim=1)

                ins = torch.zeros(e - s, device=device)
                nei = torch.zeros(e - s, device=device)

                for j, zj in enumerate(zall):
                    aa_i = torch.exp(F.cosine_similarity(zv, zj, dim=1))
                    bb_i = BATCH_SIZE_COEFF * torch.exp(
                        F.cosine_similarity(zv, means_t[j][None, :], dim=1)
                    )
                    ins += -torch.log(aa_i / bb_i)

                    aa_n = torch.exp(F.cosine_similarity(zng, zj, dim=1))
                    bb_n = (
                        BATCH_SIZE_COEFF
                        * torch.exp(F.cosine_similarity(zng, means_t[j][None, :], dim=1))
                        + torch.exp(F.cosine_similarity(zng, means_t[v][None, :], dim=1))
                    )
                    nei += -torch.log(aa_n / bb_n)

                ins /= len(VIEW_ORDER) * LATENT_DIM
                nei /= len(VIEW_ORDER) * LATENT_DIM

                raw_parts["reconstruction"].append(rec.cpu().numpy())
                raw_parts["neighborhood"].append(nei.cpu().numpy())
                raw_parts["prediction"].append(prd.cpu().numpy())
                raw_parts["instance"].append(ins.cpu().numpy())

            for comp in COMPONENTS:
                arr = np.concatenate(raw_parts[comp]).astype(np.float64, copy=False)
                lo, hi = ranges[v][comp]
                z = arr.copy() if hi == lo else (arr - lo) / (hi - lo)
                agg[comp] += z

    total = (
        WEIGHTS[0] * agg["reconstruction"]
        + WEIGHTS[1] * agg["neighborhood"]
        + WEIGHTS[2] * agg["prediction"]
        + WEIGHTS[3] * agg["instance"]
    )
    return total.astype(np.float32, copy=False)


def exact_topk_indices(scores, frac):
    scores = np.asarray(scores)
    n = len(scores)
    k = max(1, int(round(n * frac)))
    # stable exact-k: score descending, global row index ascending for exact ties
    order = np.lexsort((np.arange(n, dtype=np.int64), -scores.astype(np.float64)))
    return order[:k], k


def main(cli_root=None, query_chunk=20_000, torch_chunk=2048, resume=False):
    root = resolve_root(cli_root)
    final_dir = root / "21_INACR_Final_Training_100k"
    clean_dir = root / "03_Proposed_Results_Seed_42"
    out = root / "22_INACR_Full_Clean_Audit"
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("INA-CR FULL-CLEAN SCORING/RUNTIME AUDIT")
    print("No injected data or final-test labels are used.")
    print("=" * 80)

    # Keep the same Python/scikit-learn environment used for the completed final
    # INA-CR training.  The project RobustScaler objects may emit an
    # InconsistentVersionWarning because they were originally serialized under a
    # different scikit-learn version.  Do not change sklearn between final
    # training and this audit; consistency with the trained model is the priority.
    import sklearn
    print("scikit-learn runtime:", sklearn.__version__)

    X = np.load(require_file(root / "X.npy", "X.npy"), mmap_mode="r", allow_pickle=False)
    if X.shape != (EXPECTED_N, EXPECTED_D):
        raise ValueError(f"Unexpected X shape {X.shape}")

    ref_idx = np.load(require_file(final_dir / "INA_CR_reference_indices_100k.npy"), allow_pickle=False).astype(np.int64)
    ref_X01 = np.load(require_file(final_dir / "INA_CR_reference_X01_100k.npy"), allow_pickle=False).astype(np.float32)
    ref_neighbors_saved = np.load(require_file(final_dir / "INA_CR_reference_neighbors_k6.npy"), allow_pickle=False).astype(np.int64)

    if ref_idx.shape != (FINAL_TRAIN_N,) or ref_X01.shape != (FINAL_TRAIN_N, EXPECTED_D):
        raise ValueError("Final reference artifacts have unexpected shape.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device, torch.cuda.get_device_name(0) if device == "cuda" else "")

    pre = ViewPreprocessor.load(clean_dir)
    mins, maxs = load_input_ranges(require_file(final_dir / "INA_CR_input_minmax01_ranges.npz"))
    ranges = load_component_ranges(require_file(final_dir / "INA_CR_frozen_component_ranges.csv"))

    zmeans_file = np.load(require_file(final_dir / "INA_CR_frozen_reference_latent_means.npz"), allow_pickle=False)
    ref_latent_means = [np.asarray(zmeans_file[n], dtype=np.float32) for n in VIEW_ORDER]

    enc, dec, pred = load_models(require_file(final_dir / "INA_CR_final_model_100k_200epochs.pt"), device)

    # reconstruct reference views exactly from saved frozen X01
    ref_views = {}
    pos = 0
    for n in VIEW_ORDER:
        d = VIEW_DIMS[n]
        ref_views[n] = ref_X01[:, pos:pos+d]
        pos += d

    print("Fitting kNN index on frozen 100k reference...")
    t0 = time.time()
    nbrs = NearestNeighbors(n_neighbors=K_NEIBS + 1, algorithm="ball_tree").fit(ref_X01)
    fit_sec = time.time() - t0
    print(f"kNN fit time: {fit_sec:.3f} s")

    # map global dataset index -> reference-row position, else -1
    ref_pos = np.full(EXPECTED_N, -1, dtype=np.int32)
    ref_pos[ref_idx] = np.arange(FINAL_TRAIN_N, dtype=np.int32)

    score_path = out / "INA_CR_clean_full_scores_float32.dat"
    progress_path = out / "INA_CR_clean_full_scoring_progress.json"

    # IMPORTANT:
    # Check pre-existing state BEFORE opening/creating the memmap.  The original
    # audit script opened the memmap first, which created score_path and then
    # incorrectly interpreted that newly-created file as an old run.
    score_preexists = score_path.is_file()
    progress_preexists = progress_path.is_file()

    start = 0

    if resume:
        if not score_preexists or not progress_preexists:
            raise RuntimeError(
                "--resume requires BOTH the existing score file and progress file.\n"
                f"score exists: {score_preexists}\n"
                f"progress exists: {progress_preexists}"
            )

        with progress_path.open("r", encoding="utf-8") as f:
            p = json.load(f)

        start = int(p["next_start"])

        if not (0 <= start <= EXPECTED_N):
            raise RuntimeError(f"Invalid saved next_start={start}")

        mode = "r+"
        print(f"Resuming at row {start:,}/{EXPECTED_N:,}")

    else:
        if progress_preexists:
            raise RuntimeError(
                "A progress file from an earlier scoring run exists. "
                "Use --resume to continue that run, or remove the output folder "
                "if you intentionally want to restart from row 0."
            )

        if score_preexists:
            # This specifically recovers from the v1 startup bug: v1 created the
            # score memmap before checking whether an earlier run existed, then
            # exited before writing any progress.  A score file without a progress
            # file therefore contains no auditable completed chunks and is safe
            # to overwrite on a non-resume start.
            print(
                "Found a score file but no progress file. "
                "Treating it as a stale pre-run artifact and overwriting it."
            )

        mode = "w+"

    scores = np.memmap(
        score_path,
        dtype=np.float32,
        mode=mode,
        shape=(EXPECTED_N,),
    )

    t_all = time.time()
    knn_query_sec = 0.0
    score_sec = 0.0

    for s in range(start, EXPECTED_N, query_chunk):
        e = min(s + query_chunk, EXPECTED_N)
        raw = np.asarray(X[s:e], dtype=np.float32)
        views = apply_input_ranges(pre.transform(raw), mins, maxs)
        Xq = np.concatenate([views[n] for n in VIEW_ORDER], axis=1).astype(np.float32, copy=False)

        tq = time.time()
        # k+1 lets us remove self for rows that belong to the frozen reference set.
        _, cand = nbrs.kneighbors(Xq, n_neighbors=K_NEIBS + 1)
        knn_query_sec += time.time() - tq

        global_ids = np.arange(s, e, dtype=np.int64)
        own_ref = ref_pos[global_ids]

        qnbr = np.empty((e - s, K_NEIBS), dtype=np.int64)
        for i in range(e - s):
            c = cand[i]
            own = int(own_ref[i])
            if own >= 0:
                c = c[c != own]
            qnbr[i] = c[:K_NEIBS]

        # Cross-check exact saved leave-self-out neighbors for reference rows.
        mask = own_ref >= 0
        if np.any(mask):
            expected = ref_neighbors_saved[own_ref[mask]]
            if not np.array_equal(qnbr[mask], expected):
                raise RuntimeError(
                    "Reference-row kNN mismatch against frozen training neighbor artifact."
                )

        ts = time.time()
        chunk_scores = score_chunk(
            views, qnbr, ref_views, enc, dec, pred,
            ref_latent_means, ranges, device, torch_chunk=torch_chunk
        )
        score_sec += time.time() - ts

        if not np.isfinite(chunk_scores).all():
            raise FloatingPointError(f"Non-finite scores in rows [{s},{e}).")

        scores[s:e] = chunk_scores
        scores.flush()

        with progress_path.open("w", encoding="utf-8") as f:
            json.dump({
                "next_start": e,
                "n_total": EXPECTED_N,
                "query_chunk": query_chunk,
                "torch_chunk": torch_chunk,
            }, f, indent=2)

        elapsed = time.time() - t_all
        done = e - start
        rate = done / max(elapsed, 1e-9)
        eta = (EXPECTED_N - e) / max(rate, 1e-9)
        print(
            f"Rows {s:,}-{e:,}/{EXPECTED_N:,} | "
            f"chunk score min={chunk_scores.min():.6f} max={chunk_scores.max():.6f} | "
            f"elapsed={elapsed/60:.1f} min | ETA={eta/60:.1f} min"
        )

    scores.flush()
    clean = np.asarray(scores, dtype=np.float32)

    # Compare the 100k reference members against the frozen final-training clean score artifact.
    ref_saved = np.load(require_file(final_dir / "INA_CR_clean_reference_total_scores.npy"), allow_pickle=False)
    max_ref_diff = float(np.max(np.abs(clean[ref_idx].astype(np.float64) - ref_saved.astype(np.float64))))

    budget_rows = []
    tie_rows = []
    for b in BUDGETS:
        idx, k = exact_topk_indices(clean, b)
        boundary = float(clean[idx[-1]])
        n_equal = int(np.count_nonzero(clean == boundary))
        n_greater = int(np.count_nonzero(clean > boundary))
        needed_from_tie = k - n_greater

        budget_rows.append({
            "budget_fraction": b,
            "budget_percent": 100*b,
            "k": k,
            "boundary_score": boundary,
            "score_min_selected": float(clean[idx].min()),
            "score_max_selected": float(clean[idx].max()),
        })
        tie_rows.append({
            "budget_fraction": b,
            "k": k,
            "boundary_score": boundary,
            "n_scores_greater_than_boundary": n_greater,
            "n_scores_equal_boundary": n_equal,
            "n_selected_from_boundary_tie": needed_from_tie,
        })

    with (out / "INA_CR_Clean_Budget_Boundaries.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(budget_rows[0].keys()))
        w.writeheader(); w.writerows(budget_rows)

    with (out / "INA_CR_Clean_Tie_Diagnostics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(tie_rows[0].keys()))
        w.writeheader(); w.writerows(tie_rows)

    summary = {
        "stage": "INA_CR_full_clean_scoring_audit",
        "status": "PASS" if max_ref_diff < 1e-5 else "FAIL",
        "injected_data_used": False,
        "final_test_labels_used": False,
        "n_scored": EXPECTED_N,
        "reference_n": FINAL_TRAIN_N,
        "query_chunk": query_chunk,
        "torch_chunk": torch_chunk,
        "knn_fit_seconds": fit_sec,
        "knn_query_seconds_total": knn_query_sec,
        "model_score_seconds_total": score_sec,
        "wall_seconds_this_run": time.time() - t_all,
        "score_min": float(clean.min()),
        "score_max": float(clean.max()),
        "score_mean": float(clean.mean()),
        "score_std": float(clean.std()),
        "finite": bool(np.isfinite(clean).all()),
        "max_abs_difference_reference_members_vs_frozen_training_scores": max_ref_diff,
        "reference_score_match_tolerance": 1e-5,
        "budgets": budget_rows,
        "ties": tie_rows,
        "score_file": score_path.name,
        "note": (
            "Full clean scores are query-independent and use frozen clean-reference "
            "statistics/ranges. No injected data or test labels were used."
        ),
    }

    with (out / "INA_CR_Full_Clean_Audit_Summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print("INA-CR FULL-CLEAN AUDIT RESULT:", summary["status"])
    print(f"Scored: {EXPECTED_N:,}")
    print(
        f"Score: min={clean.min():.6f}, max={clean.max():.6f}, "
        f"mean={clean.mean():.6f}, std={clean.std():.6f}"
    )
    print(f"Reference-member max |diff| vs frozen training scores: {max_ref_diff:.3e}")
    print(f"kNN query total: {knn_query_sec/60:.2f} min")
    print(f"model scoring total: {score_sec/60:.2f} min")
    print("No injected data or final-test labels were used.")
    print("=" * 80)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, default=None)
    p.add_argument("--query-chunk", type=int, default=20_000)
    p.add_argument("--torch-chunk", type=int, default=2048)
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()
    main(a.root, a.query_chunk, a.torch_chunk, a.resume)
