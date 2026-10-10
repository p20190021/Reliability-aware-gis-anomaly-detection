# Reliability-Aware Multi-View Anomaly Detection for Heterogeneous GIS Data

This repository contains the research code and reproducibility material for
the reliability-aware multi-view anomaly-detection framework developed for
quality assurance of heterogeneous GIS data pipelines.

## Repository contents

- `src/` — canonical reusable implementation of the final Full/Pure scoring logic.
- `scripts/` — reproducibility and evaluation entry points.
- `experiments/` — ablation, stability, sensitivity, and variant experiments.
- `baselines/` — final SCoNE, INA-CR, and SSF-AE implementations.
- `notebooks/research_workflow/` — research notebooks for feature engineering,
  conventional baselines, comparison, and analysis.
- `notebooks/historical_provenance/` — historical proposed-method notebooks
  retained to document the development/training lineage.
- `validation/` — frozen reference implementation and historical Pure-correction
  provenance used to verify the final implementation.
- `reference_results/` — small numerical reference outputs used for reproducibility checks.
- `data/` — source-data manifest and data-access documentation.

## Canonical Full/Pure scoring implementation

Use:

- `src/proposed_method.py`
- `scripts/reproduce_full_pure.py`

The refactored implementation has been checked against the frozen reference
implementation on the saved experimental component-score arrays. Across the
reported Full/Pure injected-anomaly experiments, all 30
injection-ratio × inspection-budget summary values matched, with maximum
absolute numerical difference approximately `1.42e-14`.

## Historical proposed-method notebooks

The notebooks under `notebooks/historical_provenance/` are retained for
research provenance and training logic. Their original Pure post-processing
branch predates the final Pure-specific spatial implementation.

They are therefore **not** the authoritative final Pure scoring path.
The final reported Full/Pure scoring logic is implemented in
`src/proposed_method.py`.

## Reproducibility scope

The current codebase contains the research implementations used in the study
and a verified post-training Full/Pure reproduction path.

The following stages have been verified against saved experimental component
scores:

- spatial refinement;
- robust clean-data calibration;
- reliability weighting;
- final ranking;
- deterministic exact top-k selection;
- injected-anomaly evaluation.

A complete clean re-execution from raw third-party GIS data through model
retraining remains a separate end-to-end reproducibility task. See
`REPRODUCIBILITY.md`.

## Data

Raw third-party geospatial datasets are not redistributed in this repository.
The complete derived 23-feature pixel-level matrix is also not deposited
because it contains pixel-level land-cover indicators derived from UKCEH
Land Cover Map 2022.

The repository instead provides data provenance, acquisition information,
feature definitions, and the research workflow required to regenerate
analysis inputs from appropriately licensed source datasets.

See `data/README.md` and `data/data_manifest.csv`.

## Path configuration

Public scripts do not depend on the original author's machine path.

Where required, set:

`GIS_ANOMALY_PROJECT_ROOT`

to the local research-data/project directory.

Historical notebooks may additionally use:

`GIS_ANOMALY_WORKSPACE_ROOT`

See `PATH_CONFIGURATION.md`.

## Citation

If you use this code, please cite the associated research article and the
specific repository release used in your work.
