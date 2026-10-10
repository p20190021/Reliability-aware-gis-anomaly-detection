# SCoNE baseline implementation protocol

This package independently reimplements **SCoNE (AAAI 2026)** from Eqs. (2)–(5) and integrates it with the current geospatial anomaly-detection experiment without modifying any existing result folders.

## Files

- `scone_core.py` — literal/optimized SCoNE implementation and deterministic exact-top-k utilities.
- `14_scone_baseline_pipeline.py` — project-specific preprocessing, disjoint validation, parameter selection, final clean/test scoring, and exports.

Place **both files** directly in:

```text
<PROJECT_ROOT>
```

## Input policy

SCoNE receives only the six processed physical views:

- terrain: 5 features
- hydrological: 3
- spatial: 4
- surface: 3
- infrastructure: 1
- land use: 7

The code reuses the current clean clipping bounds and view scalers from `03_Proposed_Results_Seed_42`. Injected data never refit preprocessing. SCoNE does not receive MVAE embeddings, Full/Pure scores, proposed spatial/context scores, reliability, agreement, or flood-risk labels.

SCoNE reference centers are sampled only from the saved clean baseline fitting pool:

```text
02_Baseline_Results_Seed_42/Baseline_sample_indices.npy
```

A hard audit checks that this saved pool matches the manuscript-reported 300,000-sample subset generated with seed 7. If it does not, the pipeline stops rather than silently creating a non-comparable baseline.

## Hyperparameter selection

The SCoNE paper fixes final ensemble size `t=200` and searches:

```text
psi = {2,4,8,16,32,64,128,256}
k   = {1,3,5,7,11,21,51,101}, with k <= psi
```

The pipeline creates a **new 0.1% validation injection** using seed `12345`, explicitly disjoint from the union of all existing final seed-42 test-injection indices. It reproduces the same five corruption mechanisms used by the paper's current experiment. All 4,290 validation anomalies plus 20,000 fixed clean validation negatives are used for ROC-AUC parameter selection.

One global `(psi,k)` is frozen by:

1. highest mean ROC-AUC across all five validation corruption scenarios;
2. highest worst-case scenario AUC;
3. smaller `psi`;
4. smaller `k`.

The existing seed-42 test labels are **never** used to choose SCoNE parameters.

## Run order

### 1. Preflight

```bash
python 14_scone_baseline_pipeline.py --stage preflight
```

This checks folders, raw data, preprocessing artifacts, the six-view partition, the exact baseline fitting pool, and all 15 current injected matrices.

### 2. Core parity test

```bash
python 14_scone_baseline_pipeline.py --stage self-test
```

The optimized scorer must exactly equal a literal Eq. (3)–(5) implementation on synthetic six-view data, including exact ties in the one-hot land-use view.

### 3. Runtime benchmark

```bash
python 14_scone_baseline_pipeline.py --stage benchmark
```

This writes:

```text
14_SCoNE_Baseline_Results_Seed_42/SCoNE_Runtime_Benchmark.csv
```

**Send this CSV/output back before starting the full parameter grid if the estimated runtime is large.** We can adjust only engineering batch sizes without changing the method.

### 4. Select parameters

```bash
python 14_scone_baseline_pipeline.py --stage select-params
```

The search is checkpointed by `psi`, so completed `psi` values are reused after interruption. Outputs include:

```text
validation/SCoNE_Parameter_Search_All.csv
validation/SCoNE_Parameter_Search_Ranked.csv
validation/SCoNE_Selected_Parameters.json
```

### 5. Final clean + test evaluation

```bash
python 14_scone_baseline_pipeline.py --stage final
```

The final run uses the frozen `(psi,k,t=200)` and scores:

- the full clean 4,290,958-pixel dataset;
- all 3 existing injection ratios;
- all 5 existing corruption scenarios;
- all 5 existing final selection budgets.

## Final outputs relevant to manuscript comparison

```text
14_SCoNE_Baseline_Results_Seed_42/final/
    SCoNE_Injected_Detection_Results.csv
    SCoNE_Injected_Mean_Results.csv
    SCoNE_Reference_0.1pct_Results.csv
    SCoNE_Injected_Score_Diagnostics.csv
    SCoNE_Final_Evaluation_Audit.json
    clean/
        SCoNE_scores_raw.npy
        SCoNE_scores_norm.npy
        SCoNE_selected_indices_top2.npy
        SCoNE_anomaly_labels_top2.npy
        SCoNE_clean_score_diagnostics.json
```

Each injected scenario also stores its native raw SCoNE score and exact top-2% selected indices for later clean/baseline/Jaccard/stability analyses.

## Ranking convention

SCoNE's published consistency score is high for normal observations. The implementation therefore uses:

```text
S_SCoNE = 1 - Cbar
```

so larger values mean more anomalous, consistent with the current manuscript. **Native raw SCoNE scores are used for all final rankings.** A clean-p1/p99 normalized copy is saved only for visualization and diagnostics.

## Degeneracy diagnostics

Because SCoNE requires the same sampled center to be valid across all six views, the pipeline records:

- fraction of scores equal to 1 (`Cbar=0`);
- number of unique scores;
- top-2% boundary score and global tie count;
- zero-radius fraction for each view, including one-hot land use.

These checks distinguish a genuinely weak SCoNE result from a neighborhood/tie degeneracy.
