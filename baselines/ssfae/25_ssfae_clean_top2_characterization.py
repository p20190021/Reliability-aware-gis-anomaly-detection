"""
SSF-AE CLEAN TOP-2% CHARACTERIZATION

Run this ONLY after:
- clean SSF-AE training is frozen;
- clean full-scene scoring is complete;
- the exact top-2% boundary has been audited and is not pathologically tied.

Purpose
-------
Characterize the exact clean SSF-AE top-2% anomaly set in the same way as
the manuscript's existing clean-data characterization and the INA-CR clean
characterization.

This script performs NO training, NO tuning, NO injected-data scoring.

It computes:
1. Flood-context overlap and enrichment.
2. Terrain / hydrological characteristics:
   - slope
   - transformed flow accumulation
3. Eight-neighbour connected-component morphology with NO morphological
   preprocessing.
4. Clean top-2% Jaccard overlap against:
   - IF, LOF, MD, PCA, AE, HBOS, Pure, Full
   - INA-CR
   SCoNE is intentionally excluded from this clean spatial/Jaccard
   characterization because its clean top-2% set cuts through a very large
   native-score tie and is therefore not uniquely defined by its score.

The flood-risk layer is treated only as contextual information, NOT complete
anomaly ground truth. Connected-component statistics describe morphology,
NOT detection accuracy.

Expected project layout
-----------------------
.../Code/
    X.npy
    y.npy
    valid_mask_1.npy

    04_Baseline_Comparison_Results_Seed_42/
        IF_top2pct_selected_indices.npy
        LOF_top2pct_selected_indices.npy
        MD_top2pct_selected_indices.npy
        PCA_top2pct_selected_indices.npy
        AE_top2pct_selected_indices.npy
        HBOS_top2pct_selected_indices.npy
        Pure_top2pct_selected_indices.npy
        Full_top2pct_selected_indices.npy

    25_INACR_Clean_Characterization/
        INA_CR_top2pct_selected_indices.npy
        INA_CR_top2pct_anomaly_mask.npy

    SSFAE/
        results/
            clean_scoring/
                SSFAE_Clean_Scoring_Summary.json
                SSFAE_Clean_Top2pct_Selected_Indices.npy

Outputs
-------
.../Code/SSFAE/results/clean_characterization/
    SSFAE_Clean_Top2pct_Anomaly_Mask_Vector.npy
    SSFAE_Flood_Context.csv
    SSFAE_Terrain_Hydrology.csv
    SSFAE_Connected_Components.csv
    SSFAE_Clean_Top2_Jaccard.csv
    SSFAE_Clean_Characterization_Summary.json

Run
---
    python 25_ssfae_clean_top2_characterization.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import label


# ---------------------------------------------------------------------
# Frozen project constants
# ---------------------------------------------------------------------
EXPECTED_N = 4_290_958
EXPECTED_TOP2 = 85_820
EXPECTED_MASK_SHAPE = (2897, 2696)

# Same feature definitions used by the manuscript clean-data analysis.
SLOPE_INDEX = 1
FLOW_ACC_INDEX = 2

# Existing contextual layer count from the established clean analysis.
# This is only a consistency check; y.npy itself remains the source of truth.
EXPECTED_FLOOD_LABELLED_PIXELS = 478_232

METHODS_EXISTING = [
    "IF",
    "LOF",
    "MD",
    "PCA",
    "AE",
    "HBOS",
    "Pure",
    "Full",
]

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent

SSFAE_SCORING_DIR = SCRIPT_DIR / "results" / "clean_scoring"
OUTPUT_DIR = SCRIPT_DIR / "results" / "clean_characterization"

BASELINE_COMPARISON_DIR = (
    ROOT_DIR / "04_Baseline_Comparison_Results_Seed_42"
)

INACR_DIR = (
    ROOT_DIR / "25_INACR_Clean_Characterization"
)


def require_file(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"Required file was not found:\n{path}"
        )
    return path


def load_indices(path: Path, expected_count=EXPECTED_TOP2):
    require_file(path)

    idx = np.load(
        path,
        mmap_mode="r",
        allow_pickle=False,
    )

    idx = np.asarray(
        idx,
        dtype=np.int64,
    ).reshape(-1)

    if idx.size != expected_count:
        raise ValueError(
            f"{path.name} contains {idx.size:,} indices; "
            f"expected {expected_count:,}."
        )

    if np.any(idx < 0) or np.any(idx >= EXPECTED_N):
        raise ValueError(
            f"{path.name} contains out-of-range index values."
        )

    if np.unique(idx).size != expected_count:
        raise ValueError(
            f"{path.name} contains duplicate indices."
        )

    return idx


def indices_to_mask(indices):
    mask = np.zeros(
        EXPECTED_N,
        dtype=bool,
    )
    mask[indices] = True

    if int(mask.sum()) != EXPECTED_TOP2:
        raise RuntimeError(
            "Top-2% vector mask count is incorrect."
        )

    return mask


def calculate_jaccard(first_mask, second_mask):
    intersection = int(
        np.logical_and(
            first_mask,
            second_mask,
        ).sum()
    )

    union = int(
        np.logical_or(
            first_mask,
            second_mask,
        ).sum()
    )

    return (
        intersection / union
        if union > 0
        else np.nan
    ), intersection, union


def exact_top_k_indices_from_scores(scores, selected_count=EXPECTED_TOP2):
    """
    Reconstruct the exact deterministic top-k set from a frozen score vector.

    Higher score = more anomalous.
    Exact kth-score ties are resolved by lower original pixel index.
    """
    scores = np.asarray(scores).reshape(-1)

    if scores.size != EXPECTED_N:
        raise ValueError(
            f"INA-CR clean score vector has {scores.size:,} values; "
            f"expected {EXPECTED_N:,}."
        )

    if not np.isfinite(scores).all():
        raise ValueError(
            "INA-CR clean score vector contains non-finite values."
        )

    boundary_position = scores.size - selected_count
    threshold = np.partition(
        scores,
        boundary_position,
    )[boundary_position]

    above = np.flatnonzero(
        scores > threshold
    )
    tied = np.flatnonzero(
        scores == threshold
    )

    need = selected_count - above.size

    if need < 0 or need > tied.size:
        raise RuntimeError(
            "INA-CR exact top-k tie accounting failed."
        )

    # np.flatnonzero is ascending by original global index, so taking the
    # first `need` tied observations exactly reproduces the frozen rule.
    selected = np.concatenate(
        [
            above,
            tied[:need],
        ]
    ).astype(
        np.int64,
        copy=False,
    )

    if selected.size != selected_count:
        raise RuntimeError(
            "INA-CR reconstructed clean top-2% set has incorrect size."
        )

    return np.sort(selected), float(threshold)


def unique_existing_paths(paths):
    """Return de-duplicated existing paths, preserving deterministic order."""
    unique = []
    seen = set()

    for path in paths:
        path = Path(path)

        if not path.exists():
            continue

        try:
            key = str(path.resolve()).lower()
        except OSError:
            key = str(path).lower()

        if key not in seen:
            seen.add(key)
            unique.append(path)

    return unique


def discover_inacr_artifact(filename):
    """
    Find an INA-CR artifact without assuming that the final-evaluation folder
    still sits directly under ROOT_DIR.

    Search order:
      1. Established expected location.
      2. Any matching file anywhere below the project Code root.

    If more than one recursive candidate exists, stop rather than silently
    choosing an ambiguous artifact.
    """
    expected_locations = []

    if filename == "INA_CR_top_2p00pct_indices.npy":
        expected_locations.append(
            INACR_DIR
            / "clean"
            / "exact_budget_rankings"
            / filename
        )

    elif filename == "INA_CR_clean_anomaly_labels_top2pct.npy":
        expected_locations.append(
            INACR_DIR
            / "clean"
            / filename
        )

    elif filename == "INA_CR_clean_raw_scores.npy":
        expected_locations.append(
            INACR_DIR
            / "clean"
            / filename
        )

    direct = unique_existing_paths(
        expected_locations
    )

    if len(direct) == 1:
        return direct[0]

    recursive = unique_existing_paths(
        ROOT_DIR.rglob(filename)
    )

    if len(recursive) == 1:
        return recursive[0]

    if len(recursive) > 1:
        formatted = "\n".join(
            f"  {p}"
            for p in recursive
        )

        raise RuntimeError(
            f"Multiple INA-CR artifacts named {filename} were found.\n"
            "Please keep only the authoritative frozen result or set the "
            "path explicitly in the script.\n"
            f"Candidates:\n{formatted}"
        )

    return None


def load_inacr_top2():
    """
    Load the authoritative frozen INA-CR clean exact top-2% set.

    Actual project layout:
      Code/
        25_INACR_Clean_Characterization/
          INA_CR_top2pct_selected_indices.npy
          INA_CR_top2pct_anomaly_mask.npy

    Preferred source is the exact selected-index file because it directly
    represents the frozen 85,820-pixel clean ranking. The anomaly-mask file
    is accepted only as a fallback.

    No INA-CR training, refitting, rescoring, or injected-data processing
    is performed here.
    """

    candidate_indices = (
        INACR_DIR
        / "INA_CR_top2pct_selected_indices.npy"
    )

    if candidate_indices.exists():
        print(
            "INA-CR clean top-2% source:",
            candidate_indices,
        )

        return (
            load_indices(
                candidate_indices
            ),
            candidate_indices,
        )

    candidate_mask = (
        INACR_DIR
        / "INA_CR_top2pct_anomaly_mask.npy"
    )

    if candidate_mask.exists():
        print(
            "INA-CR exact index file not found; using clean top-2% mask:",
            candidate_mask,
        )

        mask = np.load(
            candidate_mask,
            mmap_mode="r",
            allow_pickle=False,
        )

        mask = np.asarray(
            mask
        ).reshape(-1)

        if mask.size != EXPECTED_N:
            raise ValueError(
                "INA-CR clean top-2% anomaly mask has wrong length."
            )

        idx = np.flatnonzero(
            mask > 0
        ).astype(
            np.int64,
            copy=False,
        )

        if idx.size != EXPECTED_TOP2:
            raise ValueError(
                "INA-CR clean top-2% anomaly mask has wrong selected count."
            )

        return (
            idx,
            candidate_mask,
        )

    raise FileNotFoundError(
        "Could not find the frozen INA-CR clean top-2% result.\n"
        "Expected one of:\n"
        f"  {candidate_indices}\n"
        f"  {candidate_mask}\n\n"
        "Based on the current project structure, both should be inside "
        "Code\\25_INACR_Clean_Characterization."
    )


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    x_path = require_file(
        ROOT_DIR / "X.npy"
    )

    y_path = require_file(
        ROOT_DIR / "y.npy"
    )

    valid_mask_path = require_file(
        ROOT_DIR / "valid_mask_1.npy"
    )

    ssfae_indices_path = require_file(
        SSFAE_SCORING_DIR
        / "SSFAE_Clean_Top2pct_Selected_Indices.npy"
    )

    ssfae_summary_path = require_file(
        SSFAE_SCORING_DIR
        / "SSFAE_Clean_Scoring_Summary.json"
    )

    # -----------------------------------------------------------------
    # Validate that clean scoring is frozen and the 2% boundary is unique.
    # -----------------------------------------------------------------
    with open(
        ssfae_summary_path,
        "r",
        encoding="utf-8",
    ) as f:
        score_summary = json.load(f)

    if (
        score_summary.get("status")
        != "clean_fullscene_scoring_and_audit_complete_injected_data_not_used"
    ):
        raise RuntimeError(
            "SSF-AE clean scoring summary is not in the expected frozen state."
        )

    top2_audit = None

    for row in score_summary.get(
        "budget_tie_audit",
        [],
    ):
        if row.get("budget") == "2%":
            top2_audit = row
            break

    if top2_audit is None:
        raise RuntimeError(
            "2% boundary audit is missing from SSF-AE scoring summary."
        )

    if int(
        top2_audit["selected_count"]
    ) != EXPECTED_TOP2:
        raise RuntimeError(
            "SSF-AE 2% selected count is not 85,820."
        )

    # A tie count of 1 is simply the unique kth score.
    if int(
        top2_audit[
            "boundary_tie_excess"
        ]
    ) != 0:
        raise RuntimeError(
            "SSF-AE top-2% boundary is not uniquely determined by score. "
            "Do not perform deterministic clean spatial characterization."
        )

    ssfae_indices = load_indices(
        ssfae_indices_path
    )

    ssfae_mask = indices_to_mask(
        ssfae_indices
    )

    np.save(
        OUTPUT_DIR
        / "SSFAE_Clean_Top2pct_Anomaly_Mask_Vector.npy",
        ssfae_mask,
    )

    # -----------------------------------------------------------------
    # Load clean project data.
    # -----------------------------------------------------------------
    X = np.load(
        x_path,
        mmap_mode="r",
        allow_pickle=False,
    )

    if X.shape != (
        EXPECTED_N,
        23,
    ):
        raise ValueError(
            f"Unexpected X shape: {X.shape}"
        )

    y = np.load(
        y_path,
        mmap_mode="r",
        allow_pickle=False,
    )

    y = np.asarray(
        y
    ).reshape(-1)

    if y.size != EXPECTED_N:
        raise ValueError(
            f"Unexpected y length: {y.size}"
        )

    valid_mask = np.load(
        valid_mask_path,
        mmap_mode="r",
        allow_pickle=False,
    ).astype(
        bool,
        copy=False,
    )

    if valid_mask.shape != EXPECTED_MASK_SHAPE:
        raise ValueError(
            f"Unexpected valid-mask shape: {valid_mask.shape}"
        )

    if int(
        valid_mask.sum()
    ) != EXPECTED_N:
        raise ValueError(
            "Valid-mask count disagrees with expected N."
        )

    flood_label = (
        y > 0
    )

    flood_count = int(
        flood_label.sum()
    )

    if (
        flood_count
        != EXPECTED_FLOOD_LABELLED_PIXELS
    ):
        raise RuntimeError(
            "Flood-context pixel count differs from the established "
            f"clean analysis: {flood_count:,} vs "
            f"{EXPECTED_FLOOD_LABELLED_PIXELS:,}."
        )

    # -----------------------------------------------------------------
    # 1. Flood-context characterization.
    # -----------------------------------------------------------------
    flood_overlap = int(
        np.logical_and(
            ssfae_mask,
            flood_label,
        ).sum()
    )

    flood_overlap_within_anomalies = (
        100.0
        * flood_overlap
        / EXPECTED_TOP2
    )

    flood_area_recovered = (
        100.0
        * flood_overlap
        / flood_count
    )

    background_flood_prevalence = (
        flood_count
        / EXPECTED_N
    )

    selected_flood_prevalence = (
        flood_overlap
        / EXPECTED_TOP2
    )

    flood_enrichment = (
        selected_flood_prevalence
        / background_flood_prevalence
    )

    flood_row = {
        "Method": "SSF-AE",
        "Selected anomaly pixels": EXPECTED_TOP2,
        "Anomaly ratio (%)": (
            100.0
            * EXPECTED_TOP2
            / EXPECTED_N
        ),
        "Flood-overlap pixels": (
            flood_overlap
        ),
        "Flood overlap within anomalies (%)": (
            flood_overlap_within_anomalies
        ),
        "Flood-labelled area recovered (%)": (
            flood_area_recovered
        ),
        "Flood-overlap enrichment (x)": (
            flood_enrichment
        ),
    }

    flood_df = pd.DataFrame(
        [flood_row]
    )

    flood_path = (
        OUTPUT_DIR
        / "SSFAE_Flood_Context.csv"
    )

    flood_df.to_csv(
        flood_path,
        index=False,
    )

    # -----------------------------------------------------------------
    # 2. Terrain / hydrology characterization.
    # -----------------------------------------------------------------
    slope = np.asarray(
        X[
            :,
            SLOPE_INDEX,
        ],
        dtype=np.float64,
    )

    flow_acc = np.asarray(
        X[
            :,
            FLOW_ACC_INDEX,
        ],
        dtype=np.float64,
    )

    selected_slope = slope[
        ssfae_mask
    ]

    selected_flow = flow_acc[
        ssfae_mask
    ]

    selected_slope_sum = float(
        np.sum(
            selected_slope,
            dtype=np.float64,
        )
    )

    selected_flow_sum = float(
        np.sum(
            selected_flow,
            dtype=np.float64,
        )
    )

    total_slope_sum = float(
        np.sum(
            slope,
            dtype=np.float64,
        )
    )

    total_flow_sum = float(
        np.sum(
            flow_acc,
            dtype=np.float64,
        )
    )

    nonselected_count = (
        EXPECTED_N
        - EXPECTED_TOP2
    )

    mean_slope_selected = (
        selected_slope_sum
        / EXPECTED_TOP2
    )

    mean_slope_nonselected = (
        total_slope_sum
        - selected_slope_sum
    ) / nonselected_count

    mean_flow_selected = (
        selected_flow_sum
        / EXPECTED_TOP2
    )

    mean_flow_nonselected = (
        total_flow_sum
        - selected_flow_sum
    ) / nonselected_count

    terrain_row = {
        "Method": "SSF-AE",
        "Mean slope in anomalies": (
            mean_slope_selected
        ),
        "Median slope in anomalies": float(
            np.median(
                selected_slope
            )
        ),
        "Mean slope in non-selected areas": (
            mean_slope_nonselected
        ),
        "Slope mean difference": (
            mean_slope_selected
            - mean_slope_nonselected
        ),
        "Mean flow accumulation in anomalies": (
            mean_flow_selected
        ),
        "Median flow accumulation in anomalies": float(
            np.median(
                selected_flow
            )
        ),
        "Mean flow accumulation in non-selected areas": (
            mean_flow_nonselected
        ),
        "Flow-accumulation mean difference": (
            mean_flow_selected
            - mean_flow_nonselected
        ),
    }

    terrain_df = pd.DataFrame(
        [terrain_row]
    )

    terrain_path = (
        OUTPUT_DIR
        / "SSFAE_Terrain_Hydrology.csv"
    )

    terrain_df.to_csv(
        terrain_path,
        index=False,
    )

    # -----------------------------------------------------------------
    # 3. Eight-neighbour connected components.
    # -----------------------------------------------------------------
    raster_selected = np.zeros(
        EXPECTED_MASK_SHAPE,
        dtype=bool,
    )

    raster_selected[
        valid_mask
    ] = ssfae_mask

    structure_8 = np.ones(
        (3, 3),
        dtype=np.uint8,
    )

    labelled, n_components = label(
        raster_selected,
        structure=structure_8,
    )

    component_sizes = np.bincount(
        labelled.ravel()
    )[1:]

    if int(
        component_sizes.sum()
    ) != EXPECTED_TOP2:
        raise RuntimeError(
            "Connected-component pixels do not sum to 85,820."
        )

    singleton_count = int(
        np.count_nonzero(
            component_sizes == 1
        )
    )

    max_component = int(
        component_sizes.max()
    )

    components_row = {
        "Method": "SSF-AE",
        "Components": int(
            n_components
        ),
        "Mean component size": float(
            component_sizes.mean()
        ),
        "Median component size": float(
            np.median(
                component_sizes
            )
        ),
        "95th-percentile component size": float(
            np.percentile(
                component_sizes,
                95,
            )
        ),
        "Maximum component size": (
            max_component
        ),
        "Singleton components": (
            singleton_count
        ),
        "Singleton component ratio (%)": (
            100.0
            * singleton_count
            / n_components
        ),
        "Largest component share (%)": (
            100.0
            * max_component
            / EXPECTED_TOP2
        ),
        "Components per 1000 anomaly pixels": (
            1000.0
            * n_components
            / EXPECTED_TOP2
        ),
    }

    components_df = pd.DataFrame(
        [components_row]
    )

    components_path = (
        OUTPUT_DIR
        / "SSFAE_Connected_Components.csv"
    )

    components_df.to_csv(
        components_path,
        index=False,
    )

    # -----------------------------------------------------------------
    # 4. Jaccard overlap with existing clean exact top-2% sets.
    # -----------------------------------------------------------------
    jaccard_rows = []

    for method in METHODS_EXISTING:
        idx_path = (
            BASELINE_COMPARISON_DIR
            / f"{method}_top2pct_selected_indices.npy"
        )

        method_indices = load_indices(
            idx_path
        )

        method_mask = indices_to_mask(
            method_indices
        )

        jac, intersection, union = (
            calculate_jaccard(
                ssfae_mask,
                method_mask,
            )
        )

        jaccard_rows.append(
            {
                "Method": method,
                "Jaccard_with_SSFAE": (
                    jac
                ),
                "Intersection": (
                    intersection
                ),
                "Union": union,
                "Comparison_source": str(
                    idx_path
                ),
            }
        )

    inacr_indices, inacr_source = (
        load_inacr_top2()
    )

    inacr_mask = indices_to_mask(
        inacr_indices
    )

    jac, intersection, union = (
        calculate_jaccard(
            ssfae_mask,
            inacr_mask,
        )
    )

    jaccard_rows.append(
        {
            "Method": "INA-CR",
            "Jaccard_with_SSFAE": (
                jac
            ),
            "Intersection": (
                intersection
            ),
            "Union": union,
            "Comparison_source": str(
                inacr_source
            ),
        }
    )

    jaccard_df = pd.DataFrame(
        jaccard_rows
    ).sort_values(
        "Jaccard_with_SSFAE",
        ascending=False,
    ).reset_index(
        drop=True
    )

    jaccard_path = (
        OUTPUT_DIR
        / "SSFAE_Clean_Top2_Jaccard.csv"
    )

    jaccard_df.to_csv(
        jaccard_path,
        index=False,
    )

    closest = (
        jaccard_df.iloc[0]
        .to_dict()
    )

    # -----------------------------------------------------------------
    # Summary.
    # -----------------------------------------------------------------
    summary = {
        "stage": (
            "SSF_AE_clean_top2_characterization"
        ),
        "status": "COMPLETE",
        "injected_data_used": False,
        "training_or_tuning_performed": False,
        "selected_count": EXPECTED_TOP2,
        "selection_threshold": float(
            top2_audit[
                "boundary_score"
            ]
        ),
        "boundary_tie": {
            "method": "SSF-AE",
            "n_samples": EXPECTED_N,
            "selection_fraction": 0.02,
            "selection_rule": (
                "ceil(0.02*N), deterministic higher-score-first, "
                "lower-index tie break"
            ),
            "selected_count": EXPECTED_TOP2,
            "threshold": float(
                top2_audit[
                    "boundary_score"
                ]
            ),
            "scores_greater_than_threshold": int(
                top2_audit[
                    "strictly_above_boundary"
                ]
            ),
            "scores_equal_to_threshold": int(
                top2_audit[
                    "exact_boundary_tie_count"
                ]
            ),
            "selected_from_boundary_tie": int(
                top2_audit[
                    "slots_taken_from_boundary_tie"
                ]
            ),
            "boundary_tie_affects_selection": (
                int(
                    top2_audit[
                        "boundary_tie_excess"
                    ]
                ) > 0
            ),
        },
        "flood_context": flood_row,
        "terrain_hydrology": terrain_row,
        "connected_components": (
            components_row
        ),
        "closest_existing_method_by_clean_top2_jaccard": {
            "Method": closest[
                "Method"
            ],
            "Jaccard_with_SSFAE": float(
                closest[
                    "Jaccard_with_SSFAE"
                ]
            ),
        },
        "jaccard_with_existing_methods": [
            {
                "Method": row[
                    "Method"
                ],
                "Jaccard_with_SSFAE": float(
                    row[
                        "Jaccard_with_SSFAE"
                    ]
                ),
                "Intersection": int(
                    row[
                        "Intersection"
                    ]
                ),
                "Union": int(
                    row[
                        "Union"
                    ]
                ),
            }
            for row
            in jaccard_df.to_dict(
                orient="records"
            )
        ],
        "scone_included_in_clean_spatial_comparison": False,
        "scone_exclusion_reason": (
            "SCoNE is not included in clean spatial/Jaccard/flood "
            "characterization because its clean top-2% selection cuts "
            "through a very large native-score boundary tie and is not "
            "uniquely determined by score."
        ),
        "interpretation_note": (
            "These are descriptive original-data characteristics, not "
            "accuracy metrics. Flood overlap is contextual alignment, "
            "not complete anomaly ground truth; connected-component "
            "statistics describe spatial morphology."
        ),
        "outputs": {
            "top2_mask_vector": str(
                OUTPUT_DIR
                / "SSFAE_Clean_Top2pct_Anomaly_Mask_Vector.npy"
            ),
            "flood_context_csv": str(
                flood_path
            ),
            "terrain_hydrology_csv": str(
                terrain_path
            ),
            "connected_components_csv": str(
                components_path
            ),
            "jaccard_csv": str(
                jaccard_path
            ),
        },
        "next_step": (
            "After reviewing this clean-only characterization, freeze "
            "the SSF-AE clean characterization and proceed to the final "
            "15 injected cases using the frozen preprocessing, models, "
            "fusion weights, overlap-scoring rule, and exact common budgets."
        ),
    }

    summary_path = (
        OUTPUT_DIR
        / "SSFAE_Clean_Characterization_Summary.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    # -----------------------------------------------------------------
    # Console report.
    # -----------------------------------------------------------------
    print("=" * 76)
    print(
        "SSF-AE CLEAN TOP-2% CHARACTERIZATION COMPLETE"
    )
    print("=" * 76)

    print("\nBoundary:")
    print(
        "  score threshold :",
        top2_audit[
            "boundary_score"
        ],
    )
    print(
        "  boundary tie    :",
        top2_audit[
            "exact_boundary_tie_count"
        ],
    )

    print("\nFlood context:")
    print(
        "  overlap pixels  :",
        flood_overlap,
    )
    print(
        "  within top-2%   :",
        f"{flood_overlap_within_anomalies:.4f}%",
    )
    print(
        "  area recovered  :",
        f"{flood_area_recovered:.4f}%",
    )
    print(
        "  enrichment      :",
        f"{flood_enrichment:.4f}x",
    )

    print("\nTerrain / hydrology:")
    print(
        "  selected slope mean/median:",
        f"{mean_slope_selected:.6f}",
        "/",
        f"{np.median(selected_slope):.6f}",
    )
    print(
        "  non-selected slope mean   :",
        f"{mean_slope_nonselected:.6f}",
    )
    print(
        "  selected flow mean/median :",
        f"{mean_flow_selected:.6f}",
        "/",
        f"{np.median(selected_flow):.6f}",
    )
    print(
        "  non-selected flow mean    :",
        f"{mean_flow_nonselected:.6f}",
    )

    print("\nConnected components (8-neighbour):")
    print(
        "  components      :",
        int(n_components),
    )
    print(
        "  mean size       :",
        f"{component_sizes.mean():.6f}",
    )
    print(
        "  median size     :",
        f"{np.median(component_sizes):.6f}",
    )
    print(
        "  max size        :",
        max_component,
    )
    print(
        "  singleton ratio :",
        f"{components_row['Singleton component ratio (%)']:.4f}%",
    )

    print("\nJaccard with clean exact top-2% sets:")
    print(
        jaccard_df[
            [
                "Method",
                "Jaccard_with_SSFAE",
                "Intersection",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\nClosest clean top-2% method:",
        closest["Method"],
        "| Jaccard =",
        float(
            closest[
                "Jaccard_with_SSFAE"
            ]
        ),
    )

    print("\nSummary:")
    print(
        " ",
        summary_path,
    )

    print(
        "\nSTOP HERE. No injected dataset has been evaluated."
    )


if __name__ == "__main__":
    main()
