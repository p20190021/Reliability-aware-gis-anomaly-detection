"""INA-CR final frozen evaluation over all 15 injected cases.

Cases
-----
Injection ratios:
    0.05%, 0.1%, 0.5%

Scenarios:
    extreme
    noise_strong
    noise_moderate
    noise_mild
    inconsistent

Frozen evaluation protocol
--------------------------
* Reuse the completed PASS full-clean score vector.
* Re-score ONLY the authoritative injected rows.
* Reuse all non-injected clean scores exactly.
* No retraining.
* No scaler/range/latent-mean refitting.
* No injection regeneration.
* Final score is not normalized before ranking.
* Higher INA-CR score = more anomalous.
* Exact project candidate counts use:
      k = ceil(budget * 4,290,958)
  at 0.1%, 0.25%, 0.5%, 1%, and 2%.

Primary metric
--------------
Injected-anomaly detection rate (%).

Important
---------
The 0.1%-Extreme case is recomputed and hard-checked against the completed
Stage-E PASS result before the script is allowed to finish.
"""
from __future__ import annotations
import os

import argparse
import csv
import gc
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors


# =============================================================================
# Frozen experiment definition
# =============================================================================

DEFAULT_ROOT = Path(os.environ.get("GIS_ANOMALY_PROJECT_ROOT", Path.cwd())).resolve()

EXPECTED_N = 4_290_958
EXPECTED_D = 23
REFERENCE_N = 100_000
LATENT_DIM = 32
K_NEIGHBORS = 6
RELEASED_BATCH_SIZE_COEFFICIENT = 100
QUERY_BATCH = 2048
NUMERICAL_TOLERANCE = 1e-5

BUDGETS = [0.0010, 0.0025, 0.0050, 0.0100, 0.0200]
WEIGHTS = (1.0, 0.25, 1.0, 0.25)

RATIO_CONFIG = [
    {
        "label": "0.05%",
        "ratio": 0.0005,
        "folder": "Injected_Datasets_with_0.05%_Injection_Rate",
        "tag": "ratio_0p0005",
    },
    {
        "label": "0.1%",
        "ratio": 0.0010,
        "folder": "Injected_Datasets_with_0.1%_Injection_Rate",
        "tag": "ratio_0p0010",
    },
    {
        "label": "0.5%",
        "ratio": 0.0050,
        "folder": "Injected_Datasets_with_0.5%_Injection_Rate",
        "tag": "ratio_0p0050",
    },
]

SCENARIO_FILES = [
    ("extreme", "Extreme", "X_injected_extreme.npy"),
    ("noise_strong", "Strong noise", "X_injected_noise_strong.npy"),
    ("noise_moderate", "Moderate noise", "X_injected_noise_moderate.npy"),
    ("noise_mild", "Mild noise", "X_injected_noise_mild.npy"),
    ("inconsistent", "Cross-feature inconsistency", "X_injected_inconsistent.npy"),
]

VIEW_ORDER = [
    "terrain",
    "hydrological",
    "spatial",
    "surface",
    "infrastructure",
    "landuse",
]

VIEW_INDICES = {
    "terrain": [0, 1, 3, 4, 5],
    "hydrological": [2, 10, 12],
    "spatial": [6, 7, 8, 9],
    "surface": [13, 14, 15],
    "infrastructure": [11],
    "landuse": list(range(16, 23)),
}

VIEW_DIMS = {
    name: len(VIEW_INDICES[name])
    for name in VIEW_ORDER
}

VIEW_SLICES: Dict[str, slice] = {}
_cursor = 0
for _view in VIEW_ORDER:
    VIEW_SLICES[_view] = slice(
        _cursor,
        _cursor + VIEW_DIMS[_view],
    )
    _cursor += VIEW_DIMS[_view]
assert _cursor == EXPECTED_D

COMPONENTS = [
    "reconstruction",
    "neighborhood",
    "prediction",
    "instance",
]


# =============================================================================
# Helpers / paths
# =============================================================================

def require_file(path: Path, label: str = "") -> Path:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {label or 'required file'}:\n{path}"
        )
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


def save_csv(path: Path, rows: List[dict]):
    if not rows:
        raise ValueError(f"No rows to save: {path}")

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


# =============================================================================
# Exact frozen six-view preprocessing
# =============================================================================

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
    def load(
        cls,
        clean_dir: Path,
    ) -> "ViewPreprocessor":
        return cls(
            terrain_scaler=joblib.load(
                require_file(
                    clean_dir / "terrain_scaler.pkl"
                )
            ),
            hydro_scaler=joblib.load(
                require_file(
                    clean_dir / "hydro_scaler.pkl"
                )
            ),
            spatial_scaler=joblib.load(
                require_file(
                    clean_dir / "spatial_scaler.pkl"
                )
            ),
            surface_scaler=joblib.load(
                require_file(
                    clean_dir / "surface_scaler.pkl"
                )
            ),
            infra_scaler=joblib.load(
                require_file(
                    clean_dir / "infra_scaler.pkl"
                )
            ),
            terrain_bounds=joblib.load(
                require_file(
                    clean_dir / "terrain_clip_bounds.pkl"
                )
            ),
            spatial_bounds=joblib.load(
                require_file(
                    clean_dir / "spatial_clip_bounds.pkl"
                )
            ),
        )

    def transform_views(
        self,
        raw: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        raw = np.asarray(raw)

        if (
            raw.ndim != 2
            or raw.shape[1] != EXPECTED_D
            or not np.isfinite(raw).all()
        ):
            raise ValueError(
                f"Invalid raw feature block {raw.shape}"
            )

        terrain = raw[
            :,
            VIEW_INDICES["terrain"],
        ].astype(
            np.float32,
            copy=True,
        )

        for fname, pos in {
            "profile_curv": 2,
            "plan_curv": 3,
        }.items():
            bounds = self.terrain_bounds[fname]
            terrain[:, pos] = np.tanh(
                np.clip(
                    terrain[:, pos],
                    float(bounds["lower"]),
                    float(bounds["upper"]),
                )
            )

        terrain[:, 4] = np.log1p(
            np.clip(
                terrain[:, 4],
                0.0,
                None,
            )
        )

        terrain = (
            self.terrain_scaler
            .transform(terrain)
            .astype(np.float32, copy=False)
        )

        hydro = (
            self.hydro_scaler
            .transform(
                raw[
                    :,
                    VIEW_INDICES[
                        "hydrological"
                    ],
                ]
            )
            .astype(np.float32, copy=False)
        )

        spatial = raw[
            :,
            VIEW_INDICES["spatial"],
        ].astype(
            np.float32,
            copy=True,
        )

        for pos, fname in enumerate(
            [
                "local_std",
                "elev_percentile",
                "elev_contrast",
                "slope_variability",
            ]
        ):
            bounds = self.spatial_bounds[fname]
            spatial[:, pos] = np.clip(
                spatial[:, pos],
                float(bounds["lower"]),
                float(bounds["upper"]),
            )

        spatial = (
            self.spatial_scaler
            .transform(spatial)
            .astype(np.float32, copy=False)
        )

        surface = (
            self.surface_scaler
            .transform(
                raw[
                    :,
                    VIEW_INDICES["surface"],
                ]
            )
            .astype(np.float32, copy=False)
        )

        infra = (
            self.infra_scaler
            .transform(
                raw[
                    :,
                    VIEW_INDICES[
                        "infrastructure"
                    ],
                ]
            )
            .astype(np.float32, copy=False)
        )

        landuse = raw[
            :,
            VIEW_INDICES["landuse"],
        ].astype(
            np.float32,
            copy=True,
        )

        views = {
            "terrain": terrain,
            "hydrological": hydro,
            "spatial": spatial,
            "surface": surface,
            "infrastructure": infra,
            "landuse": landuse,
        }

        for view in VIEW_ORDER:
            if (
                views[view].shape
                != (
                    len(raw),
                    VIEW_DIMS[view],
                )
            ):
                raise ValueError(
                    f"Bad processed {view} shape "
                    f"{views[view].shape}"
                )

            if not np.isfinite(
                views[view]
            ).all():
                raise ValueError(
                    f"Non-finite {view} values"
                )

        return views


def load_input_ranges(
    path: Path,
):
    with np.load(
        require_file(path),
        allow_pickle=False,
    ) as data:
        mins = {}
        maxs = {}

        for view in VIEW_ORDER:
            min_key = f"{view}_min"
            max_key = f"{view}_max"

            if (
                min_key not in data.files
                or max_key not in data.files
            ):
                raise KeyError(
                    f"Missing {min_key}/{max_key}; "
                    f"keys={data.files}"
                )

            mins[view] = np.asarray(
                data[min_key],
                dtype=np.float32,
            ).reshape(-1)

            maxs[view] = np.asarray(
                data[max_key],
                dtype=np.float32,
            ).reshape(-1)

            if (
                mins[view].size
                != VIEW_DIMS[view]
                or maxs[view].size
                != VIEW_DIMS[view]
            ):
                raise ValueError(
                    f"Bad input range shape for "
                    f"{view}"
                )

            if np.any(
                maxs[view] < mins[view]
            ):
                raise ValueError(
                    f"Invalid input range for {view}"
                )

    return mins, maxs


def preprocess_to_x01(
    raw: np.ndarray,
    project_preprocessor: ViewPreprocessor,
    mins,
    maxs,
):
    project_views = (
        project_preprocessor
        .transform_views(raw)
    )

    query_views = {}

    for view in VIEW_ORDER:
        lo = mins[view]
        hi = maxs[view]
        den = hi - lo

        safe = np.where(
            den > 0,
            den,
            1.0,
        ).astype(np.float32)

        z = (
            project_views[view]
            - lo
        ) / safe

        if np.any(den == 0):
            z[:, den == 0] = 0.0

        query_views[view] = np.clip(
            z,
            0.0,
            1.0,
        ).astype(
            np.float32,
            copy=False,
        )

    x01 = np.concatenate(
        [
            query_views[v]
            for v in VIEW_ORDER
        ],
        axis=1,
    ).astype(
        np.float32,
        copy=False,
    )

    if (
        x01.shape
        != (
            len(raw),
            EXPECTED_D,
        )
        or not np.isfinite(x01).all()
    ):
        raise ValueError(
            "Invalid frozen INA-CR query representation"
        )

    return query_views, x01


# =============================================================================
# Frozen INA-CR network shapes / loading
# =============================================================================

class GeneratorEncode(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(
                int(d),
                LATENT_DIM,
            ),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.encoder(x)


class GeneratorDecode(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(
                LATENT_DIM,
                int(d),
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(x)


class Prediction(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(
                LATENT_DIM,
                LATENT_DIM,
            ),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.encoder(x)


def load_models(
    path: Path,
    device: torch.device,
):
    payload = torch.load(
        require_file(path),
        map_location=device,
        weights_only=False,
    )

    if "model_state" not in payload:
        raise KeyError(
            "Final INA-CR checkpoint has no model_state"
        )

    state = payload["model_state"]

    encoders = [
        GeneratorEncode(
            VIEW_DIMS[v]
        ).to(device)
        for v in VIEW_ORDER
    ]

    decoders = [
        GeneratorDecode(
            VIEW_DIMS[v]
        ).to(device)
        for v in VIEW_ORDER
    ]

    predictors = [
        Prediction().to(device)
        for _ in VIEW_ORDER
    ]

    for model, model_state in zip(
        encoders,
        state["encoders"],
    ):
        model.load_state_dict(
            model_state
        )
        model.eval()

    for model, model_state in zip(
        decoders,
        state["decoders"],
    ):
        model.load_state_dict(
            model_state
        )
        model.eval()

    for model, model_state in zip(
        predictors,
        state["predictors"],
    ):
        model.load_state_dict(
            model_state
        )
        model.eval()

    return (
        encoders,
        decoders,
        predictors,
    )


def load_latent_means(
    path: Path,
):
    with np.load(
        require_file(path),
        allow_pickle=False,
    ) as data:
        result = {}

        for view in VIEW_ORDER:
            if view not in data.files:
                raise KeyError(
                    f"Missing latent mean {view}; "
                    f"keys={data.files}"
                )

            result[view] = np.asarray(
                data[view],
                dtype=np.float32,
            ).reshape(LATENT_DIM)

    return result


def load_component_ranges(
    path: Path,
):
    df = pd.read_csv(
        require_file(path)
    )

    result = {}

    for view in VIEW_ORDER:
        row = df[
            df["view"]
            .astype(str)
            .str.lower()
            == view
        ]

        if len(row) != 1:
            raise ValueError(
                f"Expected one frozen component "
                f"range row for {view}"
            )

        row = row.iloc[0]
        result[view] = {}

        for component in COMPONENTS:
            lo = float(
                row[
                    f"{component}_clean_min"
                ]
            )
            hi = float(
                row[
                    f"{component}_clean_max"
                ]
            )

            if (
                not np.isfinite(
                    [lo, hi]
                ).all()
                or hi <= lo
            ):
                raise ValueError(
                    f"Bad frozen {component} "
                    f"range for {view}"
                )

            result[view][component] = (
                lo,
                hi,
            )

    return result


# =============================================================================
# Frozen query-independent scorer
# =============================================================================

def score_injected_rows(
    query_views,
    query_neighbors,
    reference_x01,
    encoders,
    decoders,
    predictors,
    latent_means,
    component_ranges,
    device,
):
    reference_views = {
        view: reference_x01[
            :,
            VIEW_SLICES[view],
        ]
        for view in VIEW_ORDER
    }

    mean_t = {
        view: torch.from_numpy(
            latent_means[view]
        ).to(device).view(1, -1)
        for view in VIEW_ORDER
    }

    n = query_neighbors.shape[0]

    aggregate = {
        component: np.zeros(
            n,
            dtype=np.float64,
        )
        for component in COMPONENTS
    }

    with torch.no_grad():
        for view_id, view in enumerate(
            VIEW_ORDER
        ):
            raw_parts = {
                component: []
                for component
                in COMPONENTS
            }

            for start in range(
                0,
                n,
                QUERY_BATCH,
            ):
                end = min(
                    start + QUERY_BATCH,
                    n,
                )

                q = [
                    torch.from_numpy(
                        query_views[name][
                            start:end
                        ]
                    ).to(device)
                    for name in VIEW_ORDER
                ]

                latent_all = [
                    encoders[j](
                        q[j]
                    )
                    for j in range(
                        len(VIEW_ORDER)
                    )
                ]

                z_view = latent_all[
                    view_id
                ]

                reconstruction = torch.sum(
                    (
                        decoders[
                            view_id
                        ](z_view)
                        - q[view_id]
                    )
                    ** 2,
                    dim=1,
                )

                mean_query_latent = (
                    torch.stack(
                        latent_all,
                        dim=0,
                    )
                    .mean(dim=0)
                )

                prediction = (
                    torch.sum(
                        (
                            mean_query_latent
                            - predictors[
                                view_id
                            ](z_view)
                        )
                        ** 2,
                        dim=1,
                    )
                    / LATENT_DIM
                )

                ng_np = (
                    reference_views[
                        view
                    ][
                        query_neighbors[
                            start:end
                        ]
                    ]
                )

                ng = torch.from_numpy(
                    np.asarray(
                        ng_np,
                        dtype=np.float32,
                    )
                ).to(device)

                z_ng = encoders[
                    view_id
                ](ng).mean(dim=1)

                instance = torch.zeros(
                    end - start,
                    device=device,
                )

                neighborhood = torch.zeros(
                    end - start,
                    device=device,
                )

                for j, latent_j in enumerate(
                    latent_all
                ):
                    aa_instance = torch.exp(
                        F.cosine_similarity(
                            z_view,
                            latent_j,
                            dim=1,
                        )
                    )

                    bb_instance = (
                        RELEASED_BATCH_SIZE_COEFFICIENT
                        * torch.exp(
                            F.cosine_similarity(
                                z_view,
                                mean_t[
                                    VIEW_ORDER[j]
                                ],
                                dim=1,
                            )
                        )
                    )

                    instance += -torch.log(
                        aa_instance
                        / bb_instance
                    )

                    aa_neighborhood = torch.exp(
                        F.cosine_similarity(
                            z_ng,
                            latent_j,
                            dim=1,
                        )
                    )

                    bb_neighborhood = (
                        RELEASED_BATCH_SIZE_COEFFICIENT
                        * torch.exp(
                            F.cosine_similarity(
                                z_ng,
                                mean_t[
                                    VIEW_ORDER[j]
                                ],
                                dim=1,
                            )
                        )
                        + torch.exp(
                            F.cosine_similarity(
                                z_ng,
                                mean_t[view],
                                dim=1,
                            )
                        )
                    )

                    neighborhood += -torch.log(
                        aa_neighborhood
                        / bb_neighborhood
                    )

                instance /= (
                    len(VIEW_ORDER)
                    * LATENT_DIM
                )

                neighborhood /= (
                    len(VIEW_ORDER)
                    * LATENT_DIM
                )

                raw_parts[
                    "reconstruction"
                ].append(
                    reconstruction
                    .cpu()
                    .numpy()
                )

                raw_parts[
                    "neighborhood"
                ].append(
                    neighborhood
                    .cpu()
                    .numpy()
                )

                raw_parts[
                    "prediction"
                ].append(
                    prediction
                    .cpu()
                    .numpy()
                )

                raw_parts[
                    "instance"
                ].append(
                    instance
                    .cpu()
                    .numpy()
                )

            for component in COMPONENTS:
                raw = np.concatenate(
                    raw_parts[component]
                ).astype(
                    np.float64,
                    copy=False,
                )

                lo, hi = (
                    component_ranges[
                        view
                    ][component]
                )

                # Frozen clean-reference
                # component normalization.
                # NO query clipping.
                aggregate[
                    component
                ] += (
                    raw - lo
                ) / (
                    hi - lo
                )

    total = (
        WEIGHTS[0]
        * aggregate["reconstruction"]
        + WEIGHTS[1]
        * aggregate["neighborhood"]
        + WEIGHTS[2]
        * aggregate["prediction"]
        + WEIGHTS[3]
        * aggregate["instance"]
    )

    if not np.isfinite(
        total
    ).all():
        raise FloatingPointError(
            "Non-finite injected INA-CR score"
        )

    return total.astype(
        np.float32,
        copy=False,
    )


# =============================================================================
# Exact deterministic ranking
# =============================================================================

def exact_topmax_order(
    scores: np.ndarray,
    max_k: int,
) -> np.ndarray:
    """
    Return the exact deterministic top-max_k ordering without sorting all N rows.

    Ordering rule:
      1. higher score first;
      2. lower global row index breaks exact ties.
    """
    scores = np.asarray(
        scores,
        dtype=np.float32,
    )

    n = scores.size

    if not (
        1 <= max_k <= n
    ):
        raise ValueError(
            f"Invalid max_k={max_k}"
        )

    cut = n - max_k

    threshold = float(
        np.partition(
            scores,
            cut,
        )[cut]
    )

    greater = np.flatnonzero(
        scores > threshold
    )

    equal = np.flatnonzero(
        scores == threshold
    )

    need_equal = (
        max_k
        - greater.size
    )

    if not (
        0 <= need_equal
        <= equal.size
    ):
        raise RuntimeError(
            "Exact-top-k threshold "
            "construction failed."
        )

    # flatnonzero is already ascending,
    # exactly matching our deterministic
    # tie-breaking rule.
    candidate = np.concatenate(
        [
            greater,
            equal[:need_equal],
        ]
    )

    if candidate.size != max_k:
        raise RuntimeError(
            "Exact top-max-k candidate "
            "count mismatch."
        )

    local_order = np.lexsort(
        (
            candidate,
            -scores[candidate]
            .astype(np.float64),
        )
    )

    return candidate[
        local_order
    ].astype(
        np.int64,
        copy=False,
    )


# =============================================================================
# Stage-E hard regression check
# =============================================================================

def stagee_expected_rows(
    stagee_dir: Path,
):
    summary_path = require_file(
        stagee_dir
        / "INA_CR_StageE_Summary.json",
        "Stage-E summary",
    )

    with summary_path.open(
        "r",
        encoding="utf-8",
    ) as f:
        summary = json.load(f)

    if summary.get(
        "status"
    ) != "PASS":
        raise RuntimeError(
            "Stage E did not PASS."
        )

    df = pd.read_csv(
        require_file(
            stagee_dir
            / "INA_CR_StageE_Budget_Detection_Rates.csv",
            "Stage-E budget results",
        )
    )

    expected = {}

    for _, row in df.iterrows():
        b = float(
            row[
                "Budget_Fraction"
            ]
        )

        expected[b] = {
            "Selected_Count": int(
                row[
                    "Selected_Count"
                ]
            ),
            "Detected_Injected": int(
                row[
                    "Detected_Injected"
                ]
            ),
            "Detection_Rate_Percent": float(
                row[
                    "Detection_Rate_Percent"
                ]
            ),
        }

    if set(expected) != set(BUDGETS):
        raise RuntimeError(
            "Stage-E budget set mismatch."
        )

    return expected


def check_against_stagee(
    rows,
    expected,
):
    for row in rows:
        b = float(
            row[
                "Budget_Fraction"
            ]
        )

        exp = expected[b]

        if (
            int(
                row[
                    "Selected_Count"
                ]
            )
            != exp[
                "Selected_Count"
            ]
        ):
            raise RuntimeError(
                f"Stage-E selected-count "
                f"regression mismatch at {b}"
            )

        if (
            int(
                row[
                    "Detected_Injected"
                ]
            )
            != exp[
                "Detected_Injected"
            ]
        ):
            raise RuntimeError(
                f"Stage-E detected-count "
                f"regression mismatch at {b}"
            )

        if not math.isclose(
            float(
                row[
                    "Detection_Rate_Percent"
                ]
            ),
            exp[
                "Detection_Rate_Percent"
            ],
            abs_tol=1e-12,
            rel_tol=0.0,
        ):
            raise RuntimeError(
                f"Stage-E DR regression "
                f"mismatch at {b}"
            )


# =============================================================================
# Main
# =============================================================================

def main(
    cli_root: str | None = None,
):
    root = resolve_root(
        cli_root
    )

    train_dir = (
        root
        / "21_INACR_Final_Training_100k"
    )

    audit_dir = (
        root
        / "22_INACR_Full_Clean_Audit"
    )

    stagee_dir = (
        root
        / "23_INACR_StageE_0p1_Extreme_Check"
    )

    project_clean_dir = (
        root
        / "03_Proposed_Results_Seed_42"
    )

    injected_root = (
        root
        / "03_Injected_Proposed_Results_Seed_42"
    )

    out_dir = (
        root
        / "24_INACR_Final_All15_Evaluation"
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 80)
    print(
        "INA-CR FINAL FROZEN ALL-15 "
        "INJECTED EVALUATION"
    )
    print(
        "3 injection ratios x 5 scenarios"
    )
    print(
        "No retraining, refitting, "
        "or injection regeneration."
    )
    print("=" * 80)

    t_all = time.perf_counter()

    # -------------------------------------------------------------------------
    # Hard prerequisites
    # -------------------------------------------------------------------------

    with require_file(
        audit_dir
        / "INA_CR_Full_Clean_Audit_Summary.json"
    ).open(
        "r",
        encoding="utf-8",
    ) as f:
        audit_summary = json.load(f)

    if audit_summary.get(
        "status"
    ) != "PASS":
        raise RuntimeError(
            "Full-clean audit did not PASS."
        )

    if int(
        audit_summary.get(
            "n_scored",
            -1,
        )
    ) != EXPECTED_N:
        raise RuntimeError(
            "Full-clean audit N mismatch."
        )

    ref_repro_diff = float(
        audit_summary[
            "max_abs_difference_reference_members_vs_frozen_training_scores"
        ]
    )

    if (
        ref_repro_diff
        >= NUMERICAL_TOLERANCE
    ):
        raise RuntimeError(
            "Clean reference reproduction "
            "check is outside tolerance."
        )

    stagee_expected = (
        stagee_expected_rows(
            stagee_dir
        )
    )

    # -------------------------------------------------------------------------
    # Frozen clean data / scores
    # -------------------------------------------------------------------------

    x_clean = np.load(
        require_file(
            root / "X.npy"
        ),
        mmap_mode="r",
        allow_pickle=False,
    )

    if x_clean.shape != (
        EXPECTED_N,
        EXPECTED_D,
    ):
        raise ValueError(
            f"Unexpected clean X shape "
            f"{x_clean.shape}"
        )

    clean_scores = np.memmap(
        require_file(
            audit_dir
            / "INA_CR_clean_full_scores_float32.dat"
        ),
        dtype=np.float32,
        mode="r",
        shape=(EXPECTED_N,),
    )

    if not np.isfinite(
        clean_scores
    ).all():
        raise RuntimeError(
            "Full-clean score vector "
            "contains non-finite values."
        )

    # -------------------------------------------------------------------------
    # Frozen model / preprocessing / reference
    # -------------------------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Device:",
        device,
        (
            torch.cuda.get_device_name(0)
            if device.type == "cuda"
            else ""
        ),
    )

    project_preprocessor = (
        ViewPreprocessor.load(
            project_clean_dir
        )
    )

    mins, maxs = load_input_ranges(
        train_dir
        / "INA_CR_input_minmax01_ranges.npz"
    )

    (
        encoders,
        decoders,
        predictors,
    ) = load_models(
        train_dir
        / "INA_CR_final_model_100k_200epochs.pt",
        device,
    )

    latent_means = load_latent_means(
        train_dir
        / "INA_CR_frozen_reference_latent_means.npz"
    )

    component_ranges = (
        load_component_ranges(
            train_dir
            / "INA_CR_frozen_component_ranges.csv"
        )
    )

    reference_x01 = np.load(
        require_file(
            train_dir
            / "INA_CR_reference_X01_100k.npy"
        ),
        mmap_mode="r",
        allow_pickle=False,
    ).astype(
        np.float32,
        copy=False,
    )

    reference_idx = np.load(
        require_file(
            train_dir
            / "INA_CR_reference_indices_100k.npy"
        ),
        allow_pickle=False,
    ).astype(
        np.int64,
        copy=False,
    )

    if (
        reference_x01.shape
        != (
            REFERENCE_N,
            EXPECTED_D,
        )
        or reference_idx.shape
        != (
            REFERENCE_N,
        )
    ):
        raise ValueError(
            "Frozen reference artifact "
            "shape mismatch."
        )

    # One kNN index for all 15 cases.
    print(
        "Building one frozen 100k "
        "reference kNN index..."
    )

    t_knn_fit = time.perf_counter()

    knn = NearestNeighbors(
        n_neighbors=(
            K_NEIGHBORS + 1
        ),
        algorithm="ball_tree",
        metric="euclidean",
    ).fit(
        reference_x01
    )

    knn_fit_seconds = (
        time.perf_counter()
        - t_knn_fit
    )

    global_to_ref = np.full(
        EXPECTED_N,
        -1,
        dtype=np.int32,
    )

    global_to_ref[
        reference_idx
    ] = np.arange(
        REFERENCE_N,
        dtype=np.int32,
    )

    max_k = int(
        math.ceil(
            max(BUDGETS)
            * EXPECTED_N
        )
    )

    if max_k != 85820:
        raise RuntimeError(
            f"Unexpected top-2% count "
            f"{max_k}"
        )

    all_budget_rows = []
    all_scenario_rows = []

    # -------------------------------------------------------------------------
    # Final 15 cases
    # -------------------------------------------------------------------------

    case_number = 0

    for ratio_cfg in RATIO_CONFIG:
        ratio_label = ratio_cfg[
            "label"
        ]

        ratio_value = float(
            ratio_cfg[
                "ratio"
            ]
        )

        ratio_dir = (
            injected_root
            / ratio_cfg["folder"]
        )

        inject_idx = np.load(
            require_file(
                ratio_dir
                / "inject_indices.npy"
            ),
            allow_pickle=False,
        ).astype(
            np.int64,
            copy=False,
        ).reshape(-1)

        if (
            np.unique(
                inject_idx
            ).size
            != inject_idx.size
        ):
            raise ValueError(
                f"{ratio_label}: duplicate "
                "injection indices."
            )

        if (
            np.any(
                inject_idx < 0
            )
            or np.any(
                inject_idx >= EXPECTED_N
            )
        ):
            raise IndexError(
                f"{ratio_label}: injection "
                "index out of range."
            )

        expected_injected = int(
            math.floor(
                ratio_value
                * EXPECTED_N
            )
        )

        if inject_idx.size != expected_injected:
            raise ValueError(
                f"{ratio_label}: injected "
                f"count={inject_idx.size}, "
                f"expected={expected_injected}."
            )

        labels_path = (
            ratio_dir
            / "injection_ground_truth_labels.npy"
        )

        labels_match = None

        if labels_path.is_file():
            labels = np.load(
                labels_path,
                mmap_mode="r",
                allow_pickle=False,
            )

            if labels.size != EXPECTED_N:
                raise ValueError(
                    f"{ratio_label}: ground-truth "
                    "label length mismatch."
                )

            gt_idx = np.flatnonzero(
                np.asarray(labels) > 0
            )

            labels_match = bool(
                np.array_equal(
                    np.sort(
                        inject_idx
                    ),
                    gt_idx,
                )
            )

            if not labels_match:
                raise RuntimeError(
                    f"{ratio_label}: "
                    "inject_indices and "
                    "ground truth disagree."
                )

        own_reference_rows = (
            global_to_ref[
                inject_idx
            ]
        )

        n_injected_in_reference = int(
            np.count_nonzero(
                own_reference_rows >= 0
            )
        )

        ratio_out = (
            out_dir
            / ratio_cfg["tag"]
        )

        ratio_out.mkdir(
            parents=True,
            exist_ok=True,
        )

        np.save(
            ratio_out
            / "inject_indices.npy",
            inject_idx,
        )

        for (
            scenario,
            scenario_display,
            scenario_file,
        ) in SCENARIO_FILES:
            case_number += 1

            print(
                "\n"
                + "=" * 78
            )
            print(
                f"CASE {case_number}/15 | "
                f"{ratio_label} | "
                f"{scenario_display}"
            )
            print(
                "=" * 78
            )

            case_start = (
                time.perf_counter()
            )

            x_injected = np.load(
                require_file(
                    ratio_dir
                    / scenario_file
                ),
                mmap_mode="r",
                allow_pickle=False,
            )

            if x_injected.shape != (
                EXPECTED_N,
                EXPECTED_D,
            ):
                raise ValueError(
                    f"{ratio_label}/{scenario}: "
                    f"matrix shape "
                    f"{x_injected.shape}"
                )

            # Check 4096 deterministic non-injected rows.
            rng = np.random.default_rng(
                20260815
                + case_number
            )

            injected_mask = np.zeros(
                EXPECTED_N,
                dtype=bool,
            )

            injected_mask[
                inject_idx
            ] = True

            noninjected_pool = (
                np.flatnonzero(
                    ~injected_mask
                )
            )

            check_n = min(
                4096,
                noninjected_pool.size,
            )

            check_idx = rng.choice(
                noninjected_pool,
                size=check_n,
                replace=False,
            )

            noninjected_equal = bool(
                np.array_equal(
                    np.asarray(
                        x_clean[
                            check_idx
                        ]
                    ),
                    np.asarray(
                        x_injected[
                            check_idx
                        ]
                    ),
                )
            )

            del (
                injected_mask,
                noninjected_pool,
            )

            if not noninjected_equal:
                raise RuntimeError(
                    f"{ratio_label}/{scenario}: "
                    "sampled non-injected rows "
                    "differ from clean data."
                )

            raw_injected_rows = np.asarray(
                x_injected[
                    inject_idx
                ],
                dtype=np.float32,
            )

            clean_raw_at_injected = (
                np.asarray(
                    x_clean[
                        inject_idx
                    ],
                    dtype=np.float32,
                )
            )

            changed_raw_rows = int(
                np.count_nonzero(
                    np.any(
                        raw_injected_rows
                        != clean_raw_at_injected,
                        axis=1,
                    )
                )
            )

            (
                query_views,
                query_x01,
            ) = preprocess_to_x01(
                raw_injected_rows,
                project_preprocessor,
                mins,
                maxs,
            )

            # k+1 candidate query lets us remove the
            # corresponding clean-reference identity
            # row when it remains near the injected query.
            t_knn = time.perf_counter()

            candidate_neighbors = (
                knn.kneighbors(
                    query_x01,
                    return_distance=False,
                )
                .astype(
                    np.int64,
                    copy=False,
                )
            )

            query_neighbors = np.empty(
                (
                    inject_idx.size,
                    K_NEIGHBORS,
                ),
                dtype=np.int64,
            )

            removed_own_reference = 0

            for i in range(
                inject_idx.size
            ):
                candidate = (
                    candidate_neighbors[i]
                )

                own_ref = int(
                    own_reference_rows[i]
                )

                if (
                    own_ref >= 0
                    and np.any(
                        candidate
                        == own_ref
                    )
                ):
                    candidate = (
                        candidate[
                            candidate
                            != own_ref
                        ]
                    )

                    removed_own_reference += 1

                query_neighbors[i] = (
                    candidate[
                        :K_NEIGHBORS
                    ]
                )

            knn_seconds = (
                time.perf_counter()
                - t_knn
            )

            t_model = time.perf_counter()

            injected_scores = (
                score_injected_rows(
                    query_views=
                        query_views,
                    query_neighbors=
                        query_neighbors,
                    reference_x01=
                        reference_x01,
                    encoders=encoders,
                    decoders=decoders,
                    predictors=predictors,
                    latent_means=
                        latent_means,
                    component_ranges=
                        component_ranges,
                    device=device,
                )
            )

            model_seconds = (
                time.perf_counter()
                - t_model
            )

            clean_at_injected = np.asarray(
                clean_scores[
                    inject_idx
                ],
                dtype=np.float32,
            )

            delta = (
                injected_scores
                .astype(np.float64)
                - clean_at_injected
                .astype(np.float64)
            )

            delta_tol = 1e-7

            increased = int(
                np.count_nonzero(
                    delta > delta_tol
                )
            )

            decreased = int(
                np.count_nonzero(
                    delta < -delta_tol
                )
            )

            unchanged = int(
                delta.size
                - increased
                - decreased
            )

            # Patch only authoritative injected positions.
            patched = np.asarray(
                clean_scores,
                dtype=np.float32,
            ).copy()

            patched[
                inject_idx
            ] = injected_scores

            ranking_top2 = exact_topmax_order(
                patched,
                max_k,
            )

            scenario_budget_rows = []

            for budget in BUDGETS:
                k = int(
                    math.ceil(
                        budget
                        * EXPECTED_N
                    )
                )

                selected = (
                    ranking_top2[:k]
                )

                detected = int(
                    np.isin(
                        inject_idx,
                        selected,
                        assume_unique=False,
                    ).sum()
                )

                threshold = float(
                    patched[
                        selected[-1]
                    ]
                )

                scores_greater = int(
                    np.count_nonzero(
                        patched
                        > threshold
                    )
                )

                scores_equal = int(
                    np.count_nonzero(
                        patched
                        == threshold
                    )
                )

                row = {
                    "Injection_Ratio":
                        ratio_label,
                    "Injection_Ratio_Fraction":
                        ratio_value,
                    "Scenario":
                        scenario_display,
                    "Scenario_Key":
                        scenario,
                    "Budget_Fraction":
                        budget,
                    "Budget_Percent":
                        100.0 * budget,
                    "Selected_Count":
                        k,
                    "Total_Injected":
                        int(
                            inject_idx.size
                        ),
                    "Detected_Injected":
                        detected,
                    "Missed_Injected":
                        int(
                            inject_idx.size
                            - detected
                        ),
                    "Detection_Rate_Percent":
                        (
                            100.0
                            * detected
                            / inject_idx.size
                        ),
                    "Raw_Final_Score_Threshold":
                        threshold,
                    "Scores_Greater_Than_Threshold":
                        scores_greater,
                    "Scores_Equal_To_Threshold":
                        scores_equal,
                }

                scenario_budget_rows.append(
                    row
                )

                all_budget_rows.append(
                    row
                )

            # Hard regression check against Stage E.
            if (
                ratio_label == "0.1%"
                and scenario == "extreme"
            ):
                check_against_stagee(
                    scenario_budget_rows,
                    stagee_expected,
                )

                print(
                    "Stage-E 0.1%-Extreme "
                    "regression check: PASS"
                )

            case_out = (
                ratio_out
                / scenario
            )

            case_out.mkdir(
                parents=True,
                exist_ok=True,
            )

            np.save(
                case_out
                / "INA_CR_injected_rows_scores.npy",
                injected_scores,
            )

            np.save(
                case_out
                / "INA_CR_top2pct_ordered_indices.npy",
                ranking_top2,
            )

            save_csv(
                case_out
                / "INA_CR_Budget_Detection_Rates.csv",
                scenario_budget_rows,
            )

            case_seconds = (
                time.perf_counter()
                - case_start
            )

            scenario_row = {
                "Injection_Ratio":
                    ratio_label,
                "Injection_Ratio_Fraction":
                    ratio_value,
                "Scenario":
                    scenario_display,
                "Scenario_Key":
                    scenario,
                "Total_Injected":
                    int(
                        inject_idx.size
                    ),
                "Expected_Injected":
                    expected_injected,
                "Changed_Raw_Rows_Among_Inject_Indices":
                    changed_raw_rows,
                "Ground_Truth_File_Present":
                    bool(
                        labels_path.is_file()
                    ),
                "Ground_Truth_Matches_Inject_Indices":
                    labels_match,
                "Sampled_NonInjected_Rows_Checked":
                    check_n,
                "Sampled_NonInjected_Rows_Equal_Clean":
                    noninjected_equal,
                "Injected_Rows_In_100k_Reference":
                    n_injected_in_reference,
                "Own_Clean_Reference_Removed_When_Present":
                    removed_own_reference,
                "Clean_Score_Mean_At_Injected_Positions":
                    float(
                        clean_at_injected.mean()
                    ),
                "Injected_Score_Mean":
                    float(
                        injected_scores.mean()
                    ),
                "Paired_Delta_Mean":
                    float(
                        delta.mean()
                    ),
                "Paired_Delta_Median":
                    float(
                        np.median(delta)
                    ),
                "Paired_Score_Increased_Count":
                    increased,
                "Paired_Score_Unchanged_Count":
                    unchanged,
                "Paired_Score_Decreased_Count":
                    decreased,
                "KNN_Seconds":
                    knn_seconds,
                "Model_Scoring_Seconds":
                    model_seconds,
                "Case_Wall_Seconds":
                    case_seconds,
            }

            all_scenario_rows.append(
                scenario_row
            )

            with (
                case_out
                / "INA_CR_Case_Diagnostics.json"
            ).open(
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(
                    scenario_row,
                    f,
                    indent=2,
                )

            print(
                f"Injected score mean="
                f"{injected_scores.mean():.6f} | "
                f"clean-at-injected mean="
                f"{clean_at_injected.mean():.6f}"
            )

            print(
                f"Paired change: "
                f"increased={increased:,}, "
                f"unchanged={unchanged:,}, "
                f"decreased={decreased:,}, "
                f"mean delta={delta.mean():.6f}"
            )

            print(
                f"kNN={knn_seconds:.2f}s | "
                f"model={model_seconds:.2f}s"
            )

            for row in scenario_budget_rows:
                print(
                    f"  Top "
                    f"{row['Budget_Percent']:>4.2f}% | "
                    f"k={row['Selected_Count']:,} | "
                    f"{row['Detected_Injected']:,}/"
                    f"{row['Total_Injected']:,} | "
                    f"DR="
                    f"{row['Detection_Rate_Percent']:.2f}%"
                )

            del (
                x_injected,
                raw_injected_rows,
                clean_raw_at_injected,
                query_views,
                query_x01,
                candidate_neighbors,
                query_neighbors,
                injected_scores,
                clean_at_injected,
                delta,
                patched,
                ranking_top2,
            )

            gc.collect()

            if device.type == "cuda":
                torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # Aggregate final outputs
    # -------------------------------------------------------------------------

    save_csv(
        out_dir
        / "INA_CR_All15_Budget_Detection_Rates.csv",
        all_budget_rows,
    )

    save_csv(
        out_dir
        / "INA_CR_All15_Scenario_Diagnostics.csv",
        all_scenario_rows,
    )

    budget_df = pd.DataFrame(
        all_budget_rows
    )

    mean_by_ratio = (
        budget_df
        .groupby(
            [
                "Injection_Ratio",
                "Injection_Ratio_Fraction",
                "Budget_Fraction",
                "Budget_Percent",
                "Selected_Count",
            ],
            as_index=False,
        )
        .agg(
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
        )
        .sort_values(
            [
                "Injection_Ratio_Fraction",
                "Budget_Fraction",
            ]
        )
    )

    mean_by_ratio.to_csv(
        out_dir
        / "INA_CR_Mean_Detection_Rates_By_Ratio.csv",
        index=False,
    )

    # Reference 0.1% scenario-wide table.
    ref_df = budget_df[
        budget_df[
            "Injection_Ratio"
        ]
        == "0.1%"
    ].copy()

    reference_wide = (
        ref_df.pivot(
            index="Scenario",
            columns="Budget_Percent",
            values="Detection_Rate_Percent",
        )
        .reset_index()
    )

    reference_wide.columns = [
        (
            "Scenario"
            if col == "Scenario"
            else f"Top_{float(col):g}%"
        )
        for col in reference_wide.columns
    ]

    reference_wide.to_csv(
        out_dir
        / "INA_CR_Reference_0p1_Scenario_Detection_Table.csv",
        index=False,
    )

    scenario_df = pd.DataFrame(
        all_scenario_rows
    )

    total_wall_seconds = (
        time.perf_counter()
        - t_all
    )

    summary = {
        "stage":
            "INA_CR_final_all15_frozen_evaluation",
        "status":
            "COMPLETE",
        "total_samples":
            EXPECTED_N,
        "reference_count":
            REFERENCE_N,
        "ratios":
            [
                cfg["label"]
                for cfg in RATIO_CONFIG
            ],
        "scenarios":
            [
                display
                for _, display, _
                in SCENARIO_FILES
            ],
        "total_cases":
            15,
        "primary_metric":
            "Injected-anomaly detection rate (%)",
        "budget_rule":
            "ceil(budget_fraction * 4,290,958)",
        "budget_selected_counts":
            {
                f"{100*b:g}%":
                    int(
                        math.ceil(
                            b
                            * EXPECTED_N
                        )
                    )
                for b in BUDGETS
            },
        "ranking_score":
            (
                "frozen native/final INA-CR "
                "total score; higher is more anomalous"
            ),
        "ranking_normalization":
            "none",
        "patching":
            (
                "All non-injected positions reuse "
                "the completed full-clean score exactly; "
                "only authoritative injected positions "
                "are rescored."
            ),
        "identity_self_exclusion":
            (
                "For injected queries whose global index "
                "belongs to the frozen 100k clean reference, "
                "exclude the corresponding clean-reference "
                "row if it appears in the k+1 nearest "
                "candidates, then retain k=6."
            ),
        "full_clean_audit_status":
            audit_summary["status"],
        "full_clean_reference_repro_max_abs_diff":
            ref_repro_diff,
        "stageE_regression_check":
            "PASS",
        "all_sampled_noninjected_equality_checks_pass":
            bool(
                scenario_df[
                    "Sampled_NonInjected_Rows_Equal_Clean"
                ].all()
            ),
        "knn_fit_seconds":
            knn_fit_seconds,
        "sum_case_knn_seconds":
            float(
                scenario_df[
                    "KNN_Seconds"
                ].sum()
            ),
        "sum_case_model_scoring_seconds":
            float(
                scenario_df[
                    "Model_Scoring_Seconds"
                ].sum()
            ),
        "wall_seconds":
            total_wall_seconds,
        "injected_data_used_for_training_or_tuning":
            False,
        "model_or_parameters_changed_after_stageE":
            False,
        "output_files": {
            "all75_budget_rows":
                "INA_CR_All15_Budget_Detection_Rates.csv",
            "all15_diagnostics":
                "INA_CR_All15_Scenario_Diagnostics.csv",
            "mean_by_ratio":
                "INA_CR_Mean_Detection_Rates_By_Ratio.csv",
            "reference_0p1_scenario_table":
                "INA_CR_Reference_0p1_Scenario_Detection_Table.csv",
        },
        "note":
            (
                "Final frozen evaluation only. "
                "No retraining, refitting, hyperparameter "
                "selection, or injection regeneration."
            ),
    }

    with (
        out_dir
        / "INA_CR_Final_All15_Summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    print(
        "\n"
        + "=" * 80
    )
    print(
        "INA-CR FINAL ALL-15 "
        "EVALUATION RESULT: COMPLETE"
    )
    print(
        "Stage-E regression check: PASS"
    )
    print(
        "All sampled non-injected "
        "equality checks: PASS"
    )
    print(
        f"Total wall time: "
        f"{total_wall_seconds/60:.2f} min"
    )

    print(
        "\nMean detection rate over "
        "five scenarios by ratio:"
    )

    print(
        mean_by_ratio[
            [
                "Injection_Ratio",
                "Budget_Percent",
                "Mean_Detection_Rate_Percent",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\nReference 0.1% "
        "scenario table:"
    )

    print(
        reference_wide.to_string(
            index=False
        )
    )

    print(
        "\nNo retraining/refitting or "
        "injection regeneration was performed."
    )
    print(
        "=" * 80
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        type=str,
        default=None,
    )

    args = parser.parse_args()

    main(
        cli_root=args.root,
    )
