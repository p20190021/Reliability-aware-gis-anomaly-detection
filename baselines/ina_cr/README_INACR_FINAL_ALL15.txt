INA-CR Final Frozen All-15 Evaluation

Purpose
-------
Evaluate the already-frozen INA-CR baseline on:
  * 0.05%, 0.1%, 0.5% injection ratios
  * Extreme, Strong noise, Moderate noise, Mild noise,
    Cross-feature inconsistency

The script re-scores ONLY injected rows and reuses the completed full-clean
score for every non-injected row.

Nothing is retrained or tuned.

Prerequisites
-------------
21_INACR_Final_Training_100k
22_INACR_Full_Clean_Audit
23_INACR_StageE_0p1_Extreme_Check
03_Injected_Proposed_Results_Seed_42

Run
---
Place 24_inacr_final_all15_evaluation.py in the INA-CR directory:

    python 24_inacr_final_all15_evaluation.py

Expected runtime
----------------
Stage E required about 18.5 seconds for 4,290 injected rows. Across all 15 cases
the total number of injected-row scorings is about 139k, so a practical
expectation is roughly 10–20 minutes, depending mainly on BallTree query time.

Hard safeguards
---------------
* Full-clean audit must have PASS status.
* Full-clean reference-score reproduction must remain within 1e-5.
* Each ratio's inject_indices.npy count must equal floor(ratio*N).
* Ground-truth label files, when present, must exactly match inject_indices.npy.
* 4096 sampled non-injected rows are checked for equality in every case.
* The 0.1%-Extreme result is recomputed and must match Stage E EXACTLY.
* Final candidate counts use ceil(), matching the existing project protocol:
      0.10% -> 4,291
      0.25% -> 10,728
      0.50% -> 21,455
      1.00% -> 42,910
      2.00% -> 85,820

Primary outputs
---------------
INA_CR_Final_All15_Summary.json
INA_CR_All15_Budget_Detection_Rates.csv
INA_CR_All15_Scenario_Diagnostics.csv
INA_CR_Mean_Detection_Rates_By_Ratio.csv
INA_CR_Reference_0p1_Scenario_Detection_Table.csv

After completion, send the final console block plus:
  INA_CR_Final_All15_Summary.json
  INA_CR_Mean_Detection_Rates_By_Ratio.csv
  INA_CR_Reference_0p1_Scenario_Detection_Table.csv
