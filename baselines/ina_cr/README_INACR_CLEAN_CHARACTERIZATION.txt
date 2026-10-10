INA-CR Clean Characterization

Why this is the next step
-------------------------
The frozen INA-CR injected evaluation is complete. The manuscript's existing
baseline-comparison section also characterizes exact top-2% anomaly sets on the
original uninjected dataset using flood context, terrain/hydrology, connected
components, and pairwise Jaccard. INA-CR should be characterized in the same way
before the manuscript tables/figure are updated.

Run
---
Place 25_inacr_clean_characterization.py in the project Code/INA-CR folder:

    python 25_inacr_clean_characterization.py

Expected runtime
----------------
Short: normally a few minutes or less. No kNN queries or neural-network scoring
are repeated; it reuses the completed INA-CR full-clean score vector.

Outputs
-------
25_INACR_Clean_Characterization/
    INA_CR_top2pct_selected_indices.npy
    INA_CR_top2pct_anomaly_mask.npy
    INA_CR_Clean_Overall_Row.csv
    INA_CR_Terrain_Hydrology_Row.csv
    INA_CR_Connected_Component_Row.csv
    INA_CR_Jaccard_Against_Existing_Methods.csv
    INA_CR_Augmented_Pairwise_Jaccard.csv
    Pairwise_Jaccard_Matrix_with_INA_CR.png
    INA_CR_Clean_Selection_Audit.json
    INA_CR_Clean_Characterization_Summary.json
    INA_CR_Manuscript_Notes.txt

Important
---------
SCoNE is intentionally NOT included in the clean Jaccard/flood/spatial table:
its clean top-2% score boundary has a massive tie, so one deterministic selected
set is an arbitrary tie resolution. SCoNE should remain in the injected/global
discrimination comparison using its tie-aware treatment, but not be given an
overinterpreted clean spatial footprint.

After completion, send:
    INA-CR CLEAN CHARACTERIZATION RESULT console block
    INA_CR_Clean_Characterization_Summary.json
    INA_CR_Jaccard_Against_Existing_Methods.csv

Then INA-CR is fully closed experimentally and we can update the manuscript
comparison tables/text with both SCoNE and INA-CR.
