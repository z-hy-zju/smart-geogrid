This repository contains three Python scripts for processing 36‑channel sensor data, extracting features, classifying impact areas, and evaluating different 4‑sensor combinations.

─────────────────────────────────────────────────────────────────────────────
File Overview

1. Feature extraction.py
   - Reads raw Excel files (each with 36 columns of time‑series data) from subfolders.
   - For each sensor, computes 14 statistical/spectral features (mean, std, FFT peak, etc.).
   - Saves the features as *_features.xlsx (14 rows × 36 columns) preserving the original subfolder structure.
   - Input folder: "excel" (configurable) → Output folder: "sensor_features" (configurable).

2. Random Forest for Area Classification and Recognition.py
   - Directly uses the raw sensor time‑series (flattened to a 1D vector) as features.
   - Trains a Random Forest classifier with Optuna hyperparameter optimisation.
   - Generates:
        • t‑SNE visualisation of the feature space.
        • Top‑50 feature importance bar chart.
        • Confusion matrix (percentage) with publication‑ready font sizes.
   - Expects the raw data to be organised in class subfolders under "sensor_features" (configurable).

3. Different sensor combinations identification.py
   - First trains a baseline Random Forest on all 36 sensors to compute per‑sensor importance (averaged over its 14 features).
   - Generates 100 combinations of 4 sensors: the top‑4 most important sensors plus 99 variants where one sensor is randomly replaced.
   - For each combination, trains and evaluates a Random Forest (with Optuna) on the selected sensor data.
   - Saves:
        • Confusion matrices for each combination (PNG).
        • Summary CSV (top4_sensor_combination_results.csv).
        • Accuracy distribution bar chart.
   - Expects the raw data organised in class subfolders under "sensor_features" (configurable).

─────────────────────────────────────────────────────────────────────────────
Requirements

Python 3.8+
numpy, pandas, scipy, scikit‑learn, matplotlib, seaborn, optuna, openpyxl

─────────────────────────────────────────────────────────────────────────────
Data Preparation

1. Raw sensor data (Excel files) must be placed in subfolders.  Each Excel file must contain 36 columns (sensors) and any number of rows (time points). No header row is assumed – adjust slicing in the code if your files include headers.

─────────────────────────────────────────────────────────────────────────────
Execution

Run the scripts in any order, but ensure the data paths are correctly set inside each file:

   python "Feature extraction.py"
   python "Random Forest for Area Classification and Recognition.py"
   python "Different sensor combinations identification.py"

Paths (modify inside the scripts if needed):
   - Feature extraction.py: data_folder = "excel", output_folder = "sensor_features"
   - Area classification: base_path = "sensor_features"
   - Sensor combination: base_path = "sensor_features"

─────────────────────────────────────────────────────────────────────────────
Outputs

- Feature extraction: *_features.xlsx files in the output folder.
- Area classification: t‑SNE plot, feature importance bar chart, confusion matrix (displayed interactively).
- Sensor combination: confusion matrices (PNG), accuracy distribution plot, CSV summary with all results.


─────────────────────────────────────────────────────────────────────────────
Notes

- The scripts assume that each raw Excel file contains exactly 36 sensor columns.
- Missing values (NaN) are imputed using the column mean.
- Hyperparameter optimisation uses Optuna (50 trials). You can adjust the number of trials inside the scripts.