# Reproducibility

## Verified post-training path

The final Full/Pure post-training implementation has been executed against the
saved experimental component-score arrays.

Validation result:

- comparable result checks: 30
- matches: 30
- maximum absolute difference: approximately `1.42e-14`

This verifies the implementation of:

1. Full/Pure spatial refinement;
2. clean-data calibration;
3. spatial/base fusion;
4. reliability weighting;
5. pre-final-clipping ranking;
6. deterministic exact top-k selection;
7. injected-anomaly detection-rate evaluation.

## Upstream research workflow

The repository also contains the research code/notebooks used for:

- feature engineering;
- MVAE training and reconstruction scoring;
- feature-space IF/LOF/MD;
- embedding-space IF/LOF/MD;
- context autoencoder;
- conventional baselines;
- SCoNE;
- INA-CR;
- SSF-AE;
- ablation, sensitivity, and stability analyses.

The historical proposed-method notebook is retained for the training lineage,
but its original Pure post-processing must not be used as the final scoring
implementation. The canonical final Pure logic is in `src/proposed_method.py`.

## End-to-end status

A future clean end-to-end execution will re-run the parameterized workflow
from appropriately licensed raw GIS inputs through training and evaluation.
Until that run is completed, the repository does not claim that the cleaned
package has independently reproduced every upstream intermediate array from
raw source data.
