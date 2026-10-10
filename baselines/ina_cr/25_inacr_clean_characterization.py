"""INA-CR clean-data characterization for manuscript integration.

Purpose
-------
INA-CR's controlled-injection evaluation is complete. This final clean-data
step makes INA-CR comparable with the manuscript's existing original-data
analysis at the exact top-2% candidate budget.

It computes for INA-CR:
  1. exact top-2% clean selection (ceil rule, deterministic ties);
  2. flood-context overlap / recovery / enrichment;
  3. slope and flow-accumulation characteristics;
  4. eight-neighbour connected-component morphology;
  5. pairwise Jaccard overlap with IF, LOF, MD, PCA, AE, HBOS, Pure, Full;
  6. an augmented Jaccard heatmap including INA-CR.

No model training, injected data, or parameter tuning is performed.

SCoNE note
----------
SCoNE is deliberately NOT added to the clean Jaccard/flood/morphology table
because its completed clean top-2% boundary cuts through a very large score tie.
Its saved deterministic clean top-2% indices are therefore an arbitrary
tie-resolution artifact and should not be interpreted spatially. INA-CR has no
such large boundary-tie problem.
"""
from __future__ import annotations
import os

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import label

import matplotlib.pyplot as plt


DEFAULT_ROOT = Path(os.environ.get("GIS_ANOMALY_PROJECT_ROOT", Path.cwd())).resolve()

EXPECTED_N = 4_290_958
EXPECTED_D = 23
TOP2_FRACTION = 0.02
EXPECTED_TOP2 = 85_820

METHODS_EXISTING = [
    "IF", "LOF", "MD", "PCA", "AE", "HBOS", "Pure", "Full"
]
METHODS_ALL = METHODS_EXISTING + ["INA-CR"]

SLOPE_INDEX = 1
FLOW_ACC_INDEX = 2


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


def exact_top_k_order(scores: np.ndarray, k: int) -> np.ndarray:
    """
    Exact deterministic top-k:
      higher score first;
      lower global pixel index first for exact ties.
    """
    scores = np.asarray(scores).reshape(-1)

    if scores.size != EXPECTED_N:
        raise ValueError(
            f"Expected {EXPECTED_N:,} scores, found {scores.size:,}"
        )

    if not np.isfinite(scores).all():
        raise ValueError("INA-CR clean scores contain non-finite values.")

    cut = scores.size - k
    threshold = np.partition(scores, cut)[cut]

    above = np.flatnonzero(scores > threshold)
    tied = np.flatnonzero(scores == threshold)
    need = k - above.size

    if need < 0 or need > tied.size:
        raise RuntimeError("Deterministic top-k threshold construction failed.")

    selected = np.concatenate([above, tied[:need]])

    order = np.lexsort(
        (
            selected,
            -scores[selected].astype(np.float64),
        )
    )

    selected = selected[order]

    if selected.size != k:
        raise RuntimeError("Exact top-k returned wrong count.")

    return selected.astype(np.int64, copy=False)


def jaccard_from_indices(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    inter = np.intersect1d(a, b, assume_unique=True).size
    union = a.size + b.size - inter
    return float(inter / union) if union else float("nan")


def connected_component_statistics(
    one_dimensional_mask: np.ndarray,
    valid_mask: np.ndarray,
) -> dict:
    one_dimensional_mask = np.asarray(
        one_dimensional_mask,
        dtype=bool,
    ).reshape(-1)

    valid_mask = np.asarray(
        valid_mask,
        dtype=bool,
    )

    expected_valid = int(valid_mask.sum())

    if one_dimensional_mask.size != expected_valid:
        raise ValueError(
            "1-D anomaly mask length does not match valid raster pixels."
        )

    anomaly_grid = np.zeros(
        valid_mask.shape,
        dtype=bool,
    )
    anomaly_grid[valid_mask] = one_dimensional_mask

    connectivity = np.ones(
        (3, 3),
        dtype=np.uint8,
    )

    labelled, n_components = label(
        anomaly_grid,
        structure=connectivity,
    )

    component_sizes = np.bincount(
        labelled.ravel()
    )[1:]

    selected_count = int(
        one_dimensional_mask.sum()
    )

    if int(component_sizes.sum()) != selected_count:
        raise RuntimeError(
            "Connected-component sizes do not sum to selected count."
        )

    singleton_count = int(
        np.sum(component_sizes == 1)
    )

    return {
        "Components": int(n_components),
        "Mean component size": float(np.mean(component_sizes)),
        "Median component size": float(np.median(component_sizes)),
        "95th-percentile component size": float(
            np.percentile(component_sizes, 95)
        ),
        "Maximum component size": int(component_sizes.max()),
        "Singleton components": singleton_count,
        "Singleton component ratio (%)": float(
            singleton_count / n_components * 100.0
        ),
        "Largest component share (%)": float(
            component_sizes.max() / selected_count * 100.0
        ),
        "Components per 1000 anomaly pixels": float(
            n_components / selected_count * 1000.0
        ),
    }


def save_csv(path: Path, rows):
    if isinstance(rows, pd.DataFrame):
        rows.to_csv(path, index=False)
        return

    rows = list(rows)
    if not rows:
        raise ValueError(f"No rows for {path}")

    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main(cli_root: str | None = None):
    root = resolve_root(cli_root)

    audit_dir = root / "22_INACR_Full_Clean_Audit"
    comparison_dir = root / "04_Baseline_Comparison_Results_Seed_42"
    all15_dir = root / "24_INACR_Final_All15_Evaluation"
    out = root / "25_INACR_Clean_Characterization"
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("INA-CR CLEAN-DATA CHARACTERIZATION")
    print("Exact top-2% original-data analysis; no training or injected data.")
    print("=" * 80)

    # ---------------------------------------------------------------------
    # Hard prerequisites
    # ---------------------------------------------------------------------

    with require_file(
        audit_dir / "INA_CR_Full_Clean_Audit_Summary.json"
    ).open("r", encoding="utf-8") as f:
        audit_summary = json.load(f)

    if audit_summary.get("status") != "PASS":
        raise RuntimeError("INA-CR full-clean audit did not PASS.")

    if int(audit_summary.get("n_scored", -1)) != EXPECTED_N:
        raise RuntimeError("INA-CR full-clean audit N mismatch.")

    final_summary_path = all15_dir / "INA_CR_Final_All15_Summary.json"
    if final_summary_path.is_file():
        with final_summary_path.open("r", encoding="utf-8") as f:
            all15_summary = json.load(f)

        if all15_summary.get("status") != "COMPLETE":
            raise RuntimeError("INA-CR final all-15 evaluation is not COMPLETE.")
    else:
        all15_summary = None

    X = np.load(
        require_file(root / "X.npy"),
        mmap_mode="r",
        allow_pickle=False,
    )

    y = np.load(
        require_file(root / "y.npy"),
        mmap_mode="r",
        allow_pickle=False,
    ).reshape(-1)

    if X.shape != (EXPECTED_N, EXPECTED_D):
        raise ValueError(f"Unexpected X shape: {X.shape}")

    if y.size != EXPECTED_N:
        raise ValueError(f"Unexpected y length: {y.size}")

    valid_mask_path = (
        root / "valid_mask_1.npy"
        if (root / "valid_mask_1.npy").is_file()
        else root / "valid_mask.npy"
    )

    valid_mask = np.load(
        require_file(valid_mask_path),
        allow_pickle=False,
    ).astype(bool, copy=False)

    if int(valid_mask.sum()) != EXPECTED_N:
        raise ValueError(
            "Valid-mask population does not match X.npy."
        )

    flood_label = np.asarray(y > 0, dtype=bool)
    total_flood_pixels = int(flood_label.sum())
    background_flood_fraction = (
        total_flood_pixels / EXPECTED_N
    )

    # ---------------------------------------------------------------------
    # INA-CR exact clean top-2%
    # ---------------------------------------------------------------------

    scores = np.memmap(
        require_file(
            audit_dir / "INA_CR_clean_full_scores_float32.dat"
        ),
        dtype=np.float32,
        mode="r",
        shape=(EXPECTED_N,),
    )

    selected_count = int(
        math.ceil(TOP2_FRACTION * EXPECTED_N)
    )

    if selected_count != EXPECTED_TOP2:
        raise RuntimeError(
            f"Unexpected exact top-2% count: {selected_count}"
        )

    inacr_idx = exact_top_k_order(
        scores,
        selected_count,
    )

    inacr_mask = np.zeros(
        EXPECTED_N,
        dtype=bool,
    )
    inacr_mask[inacr_idx] = True

    threshold = float(
        scores[inacr_idx[-1]]
    )
    greater_count = int(
        np.count_nonzero(scores > threshold)
    )
    equal_count = int(
        np.count_nonzero(scores == threshold)
    )
    selected_from_tie = (
        selected_count - greater_count
    )

    np.save(
        out / "INA_CR_top2pct_selected_indices.npy",
        inacr_idx,
    )
    np.save(
        out / "INA_CR_top2pct_anomaly_mask.npy",
        inacr_mask,
    )

    # ---------------------------------------------------------------------
    # Flood-context characterization
    # ---------------------------------------------------------------------

    flood_overlap_pixels = int(
        np.count_nonzero(
            flood_label & inacr_mask
        )
    )

    overlap_within_anomalies = (
        flood_overlap_pixels
        / selected_count
        * 100.0
    )

    flood_area_recovered = (
        flood_overlap_pixels
        / total_flood_pixels
        * 100.0
    )

    enrichment = (
        (flood_overlap_pixels / selected_count)
        / background_flood_fraction
    )

    inacr_overall_row = {
        "Method": "INA-CR",
        "Selected anomaly pixels": selected_count,
        "Anomaly ratio (%)": selected_count / EXPECTED_N * 100.0,
        "Flood-overlap pixels": flood_overlap_pixels,
        "Flood overlap within anomalies (%)": overlap_within_anomalies,
        "Flood-labelled area recovered (%)": flood_area_recovered,
        "Flood-overlap enrichment (x)": enrichment,
    }

    # ---------------------------------------------------------------------
    # Terrain / hydrological characteristics
    # ---------------------------------------------------------------------

    slope_values = X[:, SLOPE_INDEX]
    flow_values = X[:, FLOW_ACC_INDEX]

    selected_slope = np.asarray(
        slope_values[inacr_mask]
    )
    selected_flow = np.asarray(
        flow_values[inacr_mask]
    )

    total_slope_sum = float(
        np.sum(slope_values, dtype=np.float64)
    )
    total_flow_sum = float(
        np.sum(flow_values, dtype=np.float64)
    )

    selected_slope_sum = float(
        np.sum(selected_slope, dtype=np.float64)
    )
    selected_flow_sum = float(
        np.sum(selected_flow, dtype=np.float64)
    )

    normal_count = (
        EXPECTED_N - selected_count
    )

    mean_slope_selected = (
        selected_slope_sum / selected_count
    )
    mean_slope_nonselected = (
        (total_slope_sum - selected_slope_sum)
        / normal_count
    )

    mean_flow_selected = (
        selected_flow_sum / selected_count
    )
    mean_flow_nonselected = (
        (total_flow_sum - selected_flow_sum)
        / normal_count
    )

    terrain_row = {
        "Method": "INA-CR",
        "Mean slope in anomalies": mean_slope_selected,
        "Median slope in anomalies": float(
            np.median(selected_slope)
        ),
        "Mean slope in non-selected areas": mean_slope_nonselected,
        "Slope mean difference": (
            mean_slope_selected
            - mean_slope_nonselected
        ),
        "Mean flow accumulation in anomalies": mean_flow_selected,
        "Median flow accumulation in anomalies": float(
            np.median(selected_flow)
        ),
        "Mean flow accumulation in non-selected areas": mean_flow_nonselected,
        "Flow-accumulation mean difference": (
            mean_flow_selected
            - mean_flow_nonselected
        ),
    }

    # ---------------------------------------------------------------------
    # Connected components
    # ---------------------------------------------------------------------

    component_row = connected_component_statistics(
        inacr_mask,
        valid_mask,
    )
    component_row["Method"] = "INA-CR"

    # ---------------------------------------------------------------------
    # Existing method indices + Jaccard
    # ---------------------------------------------------------------------

    method_indices = {
        "INA-CR": inacr_idx,
    }

    for method in METHODS_EXISTING:
        path = (
            comparison_dir
            / f"{method}_top2pct_selected_indices.npy"
        )

        idx = np.load(
            require_file(
                path,
                f"{method} selected-index file",
            ),
            allow_pickle=False,
        ).astype(
            np.int64,
            copy=False,
        ).reshape(-1)

        if idx.size != EXPECTED_TOP2:
            raise ValueError(
                f"{method}: expected {EXPECTED_TOP2} selected rows; "
                f"found {idx.size}"
            )

        if np.unique(idx).size != idx.size:
            raise ValueError(
                f"{method}: duplicate selected indices"
            )

        method_indices[method] = idx

    matrix = np.zeros(
        (len(METHODS_ALL), len(METHODS_ALL)),
        dtype=np.float64,
    )

    for i, m1 in enumerate(METHODS_ALL):
        for j, m2 in enumerate(METHODS_ALL):
            matrix[i, j] = jaccard_from_indices(
                method_indices[m1],
                method_indices[m2],
            )

    jaccard_df = pd.DataFrame(
        matrix,
        index=METHODS_ALL,
        columns=METHODS_ALL,
    )

    jaccard_df.to_csv(
        out / "INA_CR_Augmented_Pairwise_Jaccard.csv"
    )

    inacr_jaccard_rows = [
        {
            "Method": method,
            "Jaccard_with_INA_CR": float(
                jaccard_df.loc[
                    "INA-CR",
                    method,
                ]
            ),
        }
        for method in METHODS_EXISTING
    ]

    save_csv(
        out / "INA_CR_Jaccard_Against_Existing_Methods.csv",
        inacr_jaccard_rows,
    )

    # ---------------------------------------------------------------------
    # Optional SCoNE tie warning/audit
    # ---------------------------------------------------------------------

    scone_warning = (
        "SCoNE not included in clean spatial/Jaccard/flood characterization "
        "because its clean top-2% selection cuts through a very large score tie."
    )

    scone_diag_candidates = [
        root
        / "14_SCoNE_Baseline_Results_Seed_42"
        / "final"
        / "clean"
        / "SCoNE_clean_top2_tie_diagnostics.json",
        root
        / "17_SCoNE_Final_FullClean"
        / "final"
        / "clean"
        / "SCoNE_clean_top2_tie_diagnostics.json",
    ]

    scone_diag = None
    for candidate in scone_diag_candidates:
        if candidate.is_file():
            with candidate.open("r", encoding="utf-8") as f:
                scone_diag = json.load(f)
            break

    # ---------------------------------------------------------------------
    # Save manuscript-oriented tables
    # ---------------------------------------------------------------------

    save_csv(
        out / "INA_CR_Clean_Overall_Row.csv",
        [inacr_overall_row],
    )
    save_csv(
        out / "INA_CR_Terrain_Hydrology_Row.csv",
        [terrain_row],
    )
    save_csv(
        out / "INA_CR_Connected_Component_Row.csv",
        [component_row],
    )

    selection_audit = {
        "method": "INA-CR",
        "n_samples": EXPECTED_N,
        "selection_fraction": TOP2_FRACTION,
        "selection_rule": "ceil(0.02*N), deterministic higher-score-first, lower-index tie break",
        "selected_count": selected_count,
        "threshold": threshold,
        "scores_greater_than_threshold": greater_count,
        "scores_equal_to_threshold": equal_count,
        "selected_from_boundary_tie": int(selected_from_tie),
        "boundary_tie_affects_selection": bool(
            equal_count > selected_from_tie
        ),
    }

    with (
        out / "INA_CR_Clean_Selection_Audit.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            selection_audit,
            f,
            indent=2,
        )

    # ---------------------------------------------------------------------
    # Augmented Jaccard heatmap
    # ---------------------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(9.2, 7.6)
    )

    image = ax.imshow(
        matrix,
        vmin=0.0,
        vmax=1.0,
        aspect="equal",
    )

    ax.set_xticks(
        np.arange(
            len(METHODS_ALL)
        )
    )
    ax.set_yticks(
        np.arange(
            len(METHODS_ALL)
        )
    )

    ax.set_xticklabels(
        METHODS_ALL,
        rotation=45,
        ha="right",
    )
    ax.set_yticklabels(
        METHODS_ALL
    )

    ax.set_title(
        "Pairwise overlap of exact top-2% anomaly sets"
    )

    for i in range(
        len(METHODS_ALL)
    ):
        for j in range(
            len(METHODS_ALL)
        ):
            ax.text(
                j,
                i,
                f"{matrix[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
            )

    fig.colorbar(
        image,
        ax=ax,
        label="Jaccard similarity",
    )

    fig.tight_layout()

    fig.savefig(
        out / "Pairwise_Jaccard_Matrix_with_INA_CR.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    # ---------------------------------------------------------------------
    # Final summary
    # ---------------------------------------------------------------------

    closest = max(
        inacr_jaccard_rows,
        key=lambda r: r[
            "Jaccard_with_INA_CR"
        ],
    )

    summary = {
        "stage": "INA_CR_clean_characterization",
        "status": "COMPLETE",
        "injected_data_used": False,
        "training_or_tuning_performed": False,
        "selected_count": selected_count,
        "selection_threshold": threshold,
        "boundary_tie": selection_audit,
        "flood_context": inacr_overall_row,
        "terrain_hydrology": terrain_row,
        "connected_components": component_row,
        "closest_existing_method_by_clean_top2_jaccard": closest,
        "jaccard_with_existing_methods": inacr_jaccard_rows,
        "scone_included_in_clean_spatial_comparison": False,
        "scone_exclusion_reason": scone_warning,
        "scone_tie_diagnostics_if_found": scone_diag,
        "note": (
            "These are descriptive original-data characteristics, not accuracy "
            "metrics. Flood overlap is contextual alignment, not complete anomaly "
            "ground truth; connected-component statistics describe morphology."
        ),
    }

    with (
        out / "INA_CR_Clean_Characterization_Summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    # Simple manuscript-ready text file.
    text = f"""INA-CR CLEAN CHARACTERIZATION — MANUSCRIPT NOTES

Exact top-2% clean selection:
  selected = {selected_count:,}
  threshold = {threshold:.8f}
  boundary equal-score count = {equal_count:,}
  selected from boundary = {selected_from_tie:,}
  tie affects exact selection = {selection_audit['boundary_tie_affects_selection']}

Flood-context:
  overlap pixels = {flood_overlap_pixels:,}
  overlap within INA-CR anomalies = {overlap_within_anomalies:.2f}%
  flood-labelled area recovered = {flood_area_recovered:.2f}%
  enrichment = {enrichment:.2f}x

Terrain/hydrology:
  mean slope selected = {mean_slope_selected:.3f}
  median slope selected = {np.median(selected_slope):.3f}
  mean slope non-selected = {mean_slope_nonselected:.3f}
  mean flow accumulation selected = {mean_flow_selected:.3f}
  median flow accumulation selected = {np.median(selected_flow):.3f}
  mean flow accumulation non-selected = {mean_flow_nonselected:.3f}

Spatial morphology:
  components = {component_row['Components']:,}
  singleton ratio = {component_row['Singleton component ratio (%)']:.2f}%
  largest component share = {component_row['Largest component share (%)']:.2f}%

Highest clean-set overlap:
  {closest['Method']} — Jaccard {closest['Jaccard_with_INA_CR']:.3f}

Interpretation rule:
  These original-data characteristics describe what INA-CR prioritizes; they are
  not ground-truth accuracy measures. SCoNE is excluded from clean-set spatial/
  Jaccard/flood characterization because its top-2% boundary has a massive tie.
"""

    (
        out
        / "INA_CR_Manuscript_Notes.txt"
    ).write_text(
        text,
        encoding="utf-8",
    )

    print("\n" + "=" * 80)
    print("INA-CR CLEAN CHARACTERIZATION RESULT: COMPLETE")
    print(
        f"Exact top-2%: {selected_count:,} | "
        f"threshold={threshold:.6f} | "
        f"boundary ties={equal_count:,}"
    )
    print(
        f"Flood overlap: {flood_overlap_pixels:,} | "
        f"within anomalies={overlap_within_anomalies:.2f}% | "
        f"enrichment={enrichment:.2f}x"
    )
    print(
        f"Connected components: {component_row['Components']:,} | "
        f"singleton ratio={component_row['Singleton component ratio (%)']:.2f}%"
    )
    print(
        f"Closest existing clean top-2% set: "
        f"{closest['Method']} "
        f"(Jaccard={closest['Jaccard_with_INA_CR']:.3f})"
    )
    print("SCoNE excluded from clean spatial/Jaccard characterization due tie degeneracy.")
    print("No training, tuning, or injected-data evaluation was performed.")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=str,
        default=None,
    )
    args = parser.parse_args()
    main(args.root)
