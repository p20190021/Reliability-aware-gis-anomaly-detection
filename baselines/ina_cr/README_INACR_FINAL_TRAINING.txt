INA-CR final clean training — 100,000 pixels

This is the frozen final TRAINING run. It does not evaluate injected data.

Place 21_inacr_final_training_100k.py in your INA-CR directory and run:

    python 21_inacr_final_training_100k.py

Expected duration from the completed 100k scaling test:
    approximately 22 hours for training, plus kNN/reference-scoring overhead.

IMPORTANT
---------
The script saves a checkpoint after every epoch.

If the process is interrupted, DO NOT start again normally. Resume with:

    python 21_inacr_final_training_100k.py --resume

The output folder is:

    21_INACR_Final_Training_100k

At successful completion, send back:
    INA_CR_Final_Training_Summary.json
    INA_CR_frozen_component_ranges.csv
    and the final console block beginning:
    "INA-CR FINAL TRAINING RESULT: COMPLETE"

Do not run injected-test evaluation yet. The next script will load these frozen
artifacts and first perform a full-clean scoring/runtime audit before the final
injection scenarios are evaluated.
