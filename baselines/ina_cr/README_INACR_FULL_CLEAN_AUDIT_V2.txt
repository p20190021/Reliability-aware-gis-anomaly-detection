INA-CR full-clean scoring/runtime audit — v2

What was fixed
--------------
The v1 script created the memmap score file before checking whether a prior
score file existed. On a completely fresh run, that newly-created file therefore
made the script think an older run already existed and it stopped immediately.

v2 checks the state BEFORE creating/opening the memmap.

Your current situation
----------------------
Because v1 stopped before any scoring chunk was completed, it may have left:

    22_INACR_Full_Clean_Audit/INA_CR_clean_full_scores_float32.dat

with NO progress JSON. v2 detects exactly this case and safely overwrites that
stale pre-run score file. You do not need to delete it manually.

Run
---
Place 22_inacr_full_clean_scoring_audit_v2.py in the INA-CR folder:

    python 22_inacr_full_clean_scoring_audit_v2.py

If a real scoring run is later interrupted after progress has been written:

    python 22_inacr_full_clean_scoring_audit_v2.py --resume

scikit-learn warning
--------------------
The project RobustScaler objects were serialized with scikit-learn 1.7.2 while
your current runtime reports 1.5.1. The completed INA-CR training used the same
current runtime/scalers. Do NOT upgrade/downgrade sklearn between final training
and this audit; keep the environment consistent with the trained model. The
warning should remain visible for documentation.

At completion send:
  * INA_CR_Full_Clean_Audit_Summary.json
  * INA_CR_Clean_Tie_Diagnostics.csv
  * final console block beginning "INA-CR FULL-CLEAN AUDIT RESULT"
