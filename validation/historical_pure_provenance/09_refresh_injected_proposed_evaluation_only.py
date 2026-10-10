import os
"""
09_refresh_injected_proposed_evaluation_only.py

Evaluation/reporting only. No model training and no anomaly-score recomputation.

Reads the already corrected Full/Pure pre-final-clipping/raw ranking scores from
all three injection-ratio folders and regenerates the established consolidated
CSV/XLSX result files using deterministic exact top-k ranking.

Required source files in each scenario folder:
    full_final_anomaly_score_raw.npy
    pure_final_anomaly_score_raw.npy

Required ratio-specific file:
    inject_indices.npy

Outputs inside EACH injection-ratio folder (same established names):
    Injected_Proposed_Detection_Rates_Raw_Final_Scores.csv
    Injected_Proposed_Detection_Rates_Raw_Final_Scores.xlsx
"""

from pathlib import Path
import math
import numpy as np
import pandas as pd
from openpyxl.styles import Font, Alignment
from openpyxl.utils import get_column_letter

ROOT_DIR = Path(os.environ.get("GIS_ANOMALY_PROJECT_ROOT", Path.cwd())).resolve()

INJECTED_ROOT = ROOT_DIR / "03_Injected_Proposed_Results_Seed_42"

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

METHOD_FILES = {
    "Full": "full_final_anomaly_score_raw.npy",
    "Pure": "pure_final_anomaly_score_raw.npy",
}

BUDGETS = [0.0010, 0.0025, 0.0050, 0.0100, 0.0200]


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required file was not found:\n{path}")
    return path


def load_vector(path: Path, expected_length=None) -> np.ndarray:
    require_file(path)
    arr = np.load(path, mmap_mode="r", allow_pickle=False)
    arr = np.asarray(arr).reshape(-1)
    if expected_length is not None and arr.size != expected_length:
        raise ValueError(
            f"{path.name} contains {arr.size:,} values; "
            f"expected {expected_length:,}."
        )
    if not np.isfinite(arr).all():
        raise ValueError(f"{path} contains NaN or infinite values.")
    return arr


def exact_top_k_order(scores: np.ndarray, k: int) -> np.ndarray:
    scores = np.asarray(scores).reshape(-1)
    if not 1 <= k <= scores.size:
        raise ValueError("Invalid top-k size.")

    partition_position = scores.size - k
    threshold = np.partition(scores, partition_position)[partition_position]

    above = np.flatnonzero(scores > threshold)
    tied = np.flatnonzero(scores == threshold)
    need = k - above.size
    if need < 0:
        raise RuntimeError("Unexpected top-k partition result.")

    selected = np.concatenate([above, tied[:need]])
    if selected.size != k:
        raise RuntimeError("Exact top-k selection returned an incorrect count.")

    order = np.lexsort((selected, -scores[selected]))
    return selected[order]


def evaluate_method(scores, injected_indices, budget):
    n = scores.size
    k = int(math.ceil(budget * n))
    selected = exact_top_k_order(scores, k)
    threshold = float(scores[selected[-1]])

    injected_mask = np.zeros(n, dtype=bool)
    injected_mask[injected_indices] = True
    detected = int(injected_mask[selected].sum())
    total = int(injected_indices.size)
    missed = total - detected

    return {
        "Budget_Fraction": float(budget),
        "Budget_Percent": float(budget * 100.0),
        "Selected_Count": int(k),
        "Total_Injected": total,
        "Detected_Injected": detected,
        "Missed_Injected": missed,
        "Detection_Rate_Percent": float(100.0 * detected / total),
        "Miss_Rate_Percent": float(100.0 * missed / total),
        "Raw_Final_Score_Threshold": threshold,
    }


def find_scenario_dir(ratio_dir: Path, scenario: str) -> Path:
    direct = ratio_dir / scenario
    candidates = []
    if direct.is_dir():
        candidates.append(direct)
    candidates.extend(
        p for p in ratio_dir.rglob(scenario)
        if p.is_dir() and p != direct
    )

    usable = []
    for p in candidates:
        if all((p / filename).exists() for filename in METHOD_FILES.values()):
            usable.append(p)

    usable = sorted(set(usable), key=lambda p: str(p).lower())
    if len(usable) != 1:
        raise FileNotFoundError(
            f"Expected exactly one usable folder for {scenario} under "
            f"{ratio_dir}, found {len(usable)}: {usable}"
        )
    return usable[0]


def format_workbook(path: Path) -> None:
    from openpyxl import load_workbook
    wb = load_workbook(path)
    for ws in wb.worksheets:
        ws.freeze_panes = "A2"
        ws.auto_filter.ref = ws.dimensions
        for cell in ws[1]:
            cell.font = Font(bold=True)
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        for col_cells in ws.columns:
            max_len = max(
                len(str(cell.value)) if cell.value is not None else 0
                for cell in col_cells
            )
            ws.column_dimensions[get_column_letter(col_cells[0].column)].width = min(max_len + 2, 55)
    wb.save(path)


def process_ratio(ratio_label: str, config: dict) -> None:
    ratio_value = float(config["ratio"])
    ratio_dir = INJECTED_ROOT / str(config["folder"])
    require_file(ratio_dir / "inject_indices.npy")

    injected_indices = np.load(
        ratio_dir / "inject_indices.npy",
        allow_pickle=False,
    ).astype(np.int64, copy=False).reshape(-1)

    if injected_indices.size == 0:
        raise ValueError(f"{ratio_label}: empty inject_indices.npy")
    if np.unique(injected_indices).size != injected_indices.size:
        raise ValueError(f"{ratio_label}: duplicate injected indices")

    detailed_rows = []
    validation_rows = []
    score_rows = []
    n_samples = None

    for scenario in SCENARIOS:
        scenario_dir = find_scenario_dir(ratio_dir, scenario)

        for method, filename in METHOD_FILES.items():
            score_path = scenario_dir / filename
            scores = load_vector(score_path, expected_length=n_samples)
            if n_samples is None:
                n_samples = scores.size
                if injected_indices.min() < 0 or injected_indices.max() >= n_samples:
                    raise ValueError(f"{ratio_label}: injected index out of range")

            for budget in BUDGETS:
                row = evaluate_method(scores, injected_indices, budget)
                row["Scenario"] = scenario
                row["Method"] = method
                detailed_rows.append(row)

            validation_rows.append(
                {
                    "Scenario": scenario,
                    "Method": method,
                    "Validation_Passed": True,
                    "Maximum_Absolute_Difference": 0.0,
                    "Reconstructed_Raw_File": str(score_path),
                }
            )

            score_rows.append(
                {
                    "Scenario": scenario,
                    "Method": method,
                    "Minimum_Raw_Final_Score": float(scores.min()),
                    "Maximum_Raw_Final_Score": float(scores.max()),
                    "Mean_Raw_Final_Score": float(np.mean(scores, dtype=np.float64)),
                    "Raw_Final_Score_File": str(score_path),
                }
            )

    expected_injected = int(math.floor(ratio_value * n_samples))
    if injected_indices.size != expected_injected:
        raise ValueError(
            f"{ratio_label}: {injected_indices.size:,} injected indices; "
            f"expected {expected_injected:,}."
        )

    results_df = pd.DataFrame(detailed_rows)[
        [
            "Scenario", "Method", "Budget_Fraction", "Budget_Percent",
            "Selected_Count", "Total_Injected", "Detected_Injected",
            "Missed_Injected", "Detection_Rate_Percent", "Miss_Rate_Percent",
            "Raw_Final_Score_Threshold",
        ]
    ].sort_values(["Scenario", "Budget_Fraction", "Method"], ignore_index=True)

    detection_summary_df = results_df.pivot_table(
        index=["Scenario", "Method"],
        columns="Budget_Percent",
        values="Detection_Rate_Percent",
        aggfunc="first",
    ).reset_index()
    detection_summary_df.columns.name = None

    detected_count_summary_df = results_df.pivot_table(
        index=["Scenario", "Method"],
        columns="Budget_Percent",
        values="Detected_Injected",
        aggfunc="first",
    ).reset_index()
    detected_count_summary_df.columns.name = None

    average_detection_df = (
        results_df.groupby(
            ["Method", "Budget_Fraction", "Budget_Percent"],
            as_index=False,
        )
        .agg(
            Mean_Detection_Rate_Percent=("Detection_Rate_Percent", "mean"),
            Standard_Deviation=("Detection_Rate_Percent", "std"),
            Minimum_Detection_Rate_Percent=("Detection_Rate_Percent", "min"),
            Maximum_Detection_Rate_Percent=("Detection_Rate_Percent", "max"),
        )
    )

    validation_df = pd.DataFrame(validation_rows)
    score_summary_df = pd.DataFrame(score_rows)

    settings_df = pd.DataFrame(
        {
            "Setting": [
                "Injection results folder",
                "Expected injection ratio",
                "Actual injected samples",
                "Number of samples",
                "Selection budgets",
                "Full ranking score",
                "Pure ranking score",
                "Pure spatial definition",
                "Pure fusion weights",
                "Selection procedure",
                "Evaluation metric",
            ],
            "Value": [
                str(ratio_dir),
                ratio_value,
                int(injected_indices.size),
                int(n_samples),
                "0.10%, 0.25%, 0.50%, 1.00%, 2.00%",
                "Saved Full pre-final-clipping/raw final score",
                "Corrected saved Pure pre-final-clipping/raw final score",
                "Pure-specific spatial inconsistency computed from MVAE reconstruction score only",
                "0.75 AE + 0.25 Pure spatial",
                "Deterministic exact top-k from pre-final-clipping/raw ranking score",
                "Injected-anomaly detection rate",
            ],
        }
    )

    out_csv = ratio_dir / "Injected_Proposed_Detection_Rates_Raw_Final_Scores.csv"
    out_xlsx = ratio_dir / "Injected_Proposed_Detection_Rates_Raw_Final_Scores.xlsx"

    results_df.to_csv(out_csv, index=False)
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        results_df.to_excel(writer, sheet_name="Detailed Results", index=False)
        detection_summary_df.to_excel(writer, sheet_name="Detection Rate Summary", index=False)
        detected_count_summary_df.to_excel(writer, sheet_name="Detected Count Summary", index=False)
        average_detection_df.to_excel(writer, sheet_name="Average Across Scenarios", index=False)
        validation_df.to_excel(writer, sheet_name="Reconstruction Validation", index=False)
        score_summary_df.to_excel(writer, sheet_name="Raw Final Score Summary", index=False)
        settings_df.to_excel(writer, sheet_name="Experimental Settings", index=False)

    format_workbook(out_xlsx)

    print("=" * 72)
    print(f"{ratio_label} injection evaluation refreshed")
    print(f"Injected pixels: {injected_indices.size:,}")
    print(out_xlsx)


for label, cfg in RATIO_CONFIG.items():
    process_ratio(label, cfg)

print("=" * 72)
print("ALL THREE INJECTION-RATIO EVALUATION WORKBOOKS REFRESHED")
print("No model training or inference was performed.")
