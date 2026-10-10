# SSF-AE baseline: isolated-folder protocol

## Recommended project layout

Keep every SSF-AE-specific implementation file and every SSF-AE-generated
artifact inside a dedicated `SSFAE` folder:

```text
<PROJECT_ROOT>\
│
├── X.npy
├── valid_mask_1.npy
├── y.npy
├── ...existing project files and result folders...
│
└── SSFAE\
    ├── 18_ssfae_preflight.py
    ├── SSF_AE_PREIMPLEMENTATION_PROTOCOL.md
    └── results\
        └── preflight\
            ├── SSFAE_Preflight_Summary.json
            ├── SSFAE_PCA_Explained_Variance.csv
            └── SSFAE_Window_Diagnostics.csv
```

The master inputs `X.npy` and `valid_mask_1.npy` remain in the existing
`Code` folder. SSF-AE only reads them. They are not copied, modified, or
overwritten.

Future SSF-AE files such as `ssfae_core.py`, the main training pipeline,
saved models, clean scores, injected scores, and evaluation tables should also
remain under `Code\SSFAE`.

## Why this layout is preferable

It isolates the new baseline from:

- the proposed-method implementation;
- conventional baseline outputs;
- SCoNE outputs;
- INA-CR outputs;
- existing injected datasets.

It also avoids duplicating the very large shared feature arrays.

## Existing project mapping confirmed from the supplied notebooks

The feature-engineering pipeline saves:

```text
X.npy
valid_mask_1.npy
```

with:

```text
X shape = (4,290,958, 23)
valid_mask shape = (2897, 2696)
valid pixels = 4,290,958
```

The established mapping is:

```python
raster = np.full(valid_mask.shape, np.nan, dtype=np.float32)
raster[valid_mask] = vector
```

and the reverse mapping is:

```python
vector = raster[valid_mask]
```

The proposed-method spatial module already uses the same ordering.

## Running the preflight

Put `18_ssfae_preflight.py` inside:

```text
...\Outlier Paper\Code\SSFAE\
```

Then open a terminal in that folder and run:

```bash
python 18_ssfae_preflight.py
```

No path edit is normally necessary. The script automatically interprets:

```text
script folder           = ...\Code\SSFAE
existing project inputs = ...\Code
SSF-AE outputs          = ...\Code\SSFAE\results\preflight
```

If the project root is different, override it explicitly:

```bash
python 18_ssfae_preflight.py --project-root "C:\path\to\Outlier Paper\Code"
```

## Current stage

The preflight:

- validates `X.npy` and `valid_mask_1.npy`;
- verifies row-to-raster ordering;
- computes the full-clean PCA spectrum in streaming mode;
- reports the number of PCs required for 99.90% cumulative variance;
- diagnoses feature-depth compatibility with SSF-AE;
- reports 40 x 40 window statistics for both stride 10 and stride 20;
- records RAM and GPU availability.

It does not train a model.

## Important paper ambiguity

The SSF-AE paper is internally inconsistent about the sliding step:

```text
methodology text: 40-pixel window, step 10
Table 1:          40-pixel window, step 20
```

The preflight diagnoses both options. We should freeze one before training
without using final injected-test labels for selection.

## What to send back

After the preflight finishes, send:

```text
SSFAE\results\preflight\SSFAE_Preflight_Summary.json
SSFAE\results\preflight\SSFAE_PCA_Explained_Variance.csv
SSFAE\results\preflight\SSFAE_Window_Diagnostics.csv
```

or simply send the console output.
