# Swim Behavior Analysis Pipeline

A Python pipeline for quantifying rodent swim behavior from pose-estimation tracking data.
Developed for the analysis reported in:

> **Automated swim-based motor phenotyping for early detection of motor impairments in ALS mouse models**
> Gloria et al.,  2026.

---

## Overview

This pipeline takes pose-estimation CSV files from swim videos (LightningPose format) and produces:

1. **Cleaned tracking data** — low-confidence keypoints are interpolated, trajectories smoothed with Savitzky-Golay filter, and the active swim portion is segmented
2. **Swim features** — 24 bilateral kinematic features covering hindlimb stroke mechanics, tail oscillation, forelimb use, body posture, and swim speed
3. **RF-based disease scoring** — P(SOD1) probabilities using pre-trained Random Forest models across 7 timepoints (postnatal day 45–127)
4. **Logistic regression** — binary classification with leave-one-out cross-validation (LOO-CV) and bootstrap confidence intervals

---

## Requirements

- Python ≥ 3.9
- See `requirements.txt` for package versions

```bash
pip install -r requirements.txt
```

---

## Repository structure

```
swim-analysis/
├── README.md
├── requirements.txt
├── LICENSE
│
├── core/
│   ├── features.py        # All feature extraction logic (main analysis code)
│   └── plotting.py        # Matplotlib figure functions
│
├── scripts/
│   ├── 01_clean_data.py           # Step 1 — clean raw tracking CSVs
│   ├── 02_extract_features.py     # Step 2 — extract swim features
│   ├── 03_rf_evaluation.py        # Step 3 — RF-based P(SOD1) scoring
│   └── 04_logistic_regression.py  # Step 4 — logistic regression with LOO-CV
│
├── models/
│   ├── step10b_rf_model_p45.joblib
│   ├── step10b_rf_model_p58.joblib
│   ├── step10b_rf_model_p73.joblib
│   ├── step10b_rf_model_p85.joblib
│   ├── step10b_rf_model_p99.joblib
│   ├── step10b_rf_model_p113.joblib
│   └── step10b_rf_model_p127.joblib
│
└── data/
    ├── raw/           ← place your LightningPose CSVs here
    ├── cleaned/       ← output of Step 1
    ├── metadata/      ← place metadata.csv here
    └── results/       ← output of Steps 2–4
```

---

## Input data format

### Tracking files (`data/raw/`)

Standard LightningPose (or DeepLabCut) output: CSV with a 3-row header (scorer / bodypart / coord).

Required keypoints:

| Keypoint | Role |
|---|---|
| `sternum` | Animal's sternum — used for swim lane segmentation |
| `start` | Start line static reference marker |
| `finish` | Finish line static reference marker |
| `nose` | Nose tip — for body axis and curvature |
| `tail_base` | Tail base — for body axis, curvature, and tail features |
| `L_toe_HL` / `R_toe_HL` | Left/right hindlimb toes |
| `L_palm_FL` / `R_palm_FL` | Left/right forelimb palms |

### Metadata (`data/metadata/metadata.csv`)

One row per video. **Required** columns:

| Column | Description |
|---|---|
| `Video_name` | Filename stem of the tracking CSV (e.g. `mouse01_p45_trial1`) |
| `Animal_ID` | Unique animal identifier |
| `Trial_number` | Trial index |
| `Group` | Experimental group (e.g. `WT`, `SOD1`) |
| `Timepoint` | Age label (e.g. `p45`, `p58`) |
| `Acquisition_date` | Recording date |
| `Camera` | Camera identifier |

**Optional** (preserved if present): `Sex`, `Genotype`, `Age_weeks`, `Experimenter`

---

## Usage

### Step 1 — Clean tracking data

```bash
python scripts/01_clean_data.py \
    --raw_dir   data/raw \
    --clean_dir data/cleaned \
    --cutoff    0.7 \
    --wl        5
```

| Argument | Default | Description |
|---|---|---|
| `--raw_dir` | `data/raw` | Input folder |
| `--clean_dir` | `data/cleaned` | Output folder |
| `--cutoff` | `0.7` | Likelihood cutoff below which keypoints are interpolated |
| `--wl` | `5` | Savitzky-Golay window length (automatically adjusted to odd) |
| `--sternum` | `sternum` | Sternum keypoint name |
| `--start` | `start` | Start line keypoint name |
| `--finish` | `finish` | Finish line keypoint name |

---

### Step 2 — Extract swim features

```bash
python scripts/02_extract_features.py \
    --clean_dir   data/cleaned \
    --metadata    data/metadata/metadata.csv \
    --output_dir  data/results \
    --fps         120 \
    --cm_per_px   0.040207 \
    --lane_length 50
```

| Argument | Default | Description |
|---|---|---|
| `--fps` | `120.0` | Camera frame rate (frames per second) |
| `--cm_per_px` | `0.040207` | Spatial calibration factor |
| `--lane_length` | `50.0` | Swim lane length (cm) — used to compute swim speed |
| `--min_frames` | `120` | Trials shorter than this are excluded (saved separately) |
| `--no_swap_lr` | *(flag)* | Add this flag if the camera is **not** mirrored |

**Output files:**

| File | Content |
|---|---|
| `all_trials_features.csv` | Every trial |
| `fastest_trial_features.csv` | Fastest trial per animal × timepoint |
| `averaged_trial_features.csv` | Mean across all trials per animal × timepoint |
| `removed_short_trials.csv` | Trials below `--min_frames` |

---

### Step 3 — RF-based disease scoring

```bash
python scripts/03_rf_evaluation.py \
    --feature_file data/results/fastest_trial_features.csv \
    --model_dir    models \
    --output_dir   data/results
```

**Output:** `rf_probabilities_p_sod1.csv` — P(SOD1) per animal per timepoint, with closest-timepoint model matching.

---

### Step 4 — Logistic regression

```bash
python scripts/04_logistic_regression.py \
    --feature_file     data/results/fastest_trial_features.csv \
    --reference_group  WT \
    --treatment_groups SOD1 \
    --n_top_features   5 \
    --n_bootstrap      1000 \
    --random_seed      42
```

| Argument | Default | Description |
|---|---|---|
| `--reference_group` | *(required)* | Control group label |
| `--treatment_groups` | *(required)* | One or more treatment labels (collapsed into one class) |
| `--n_top_features` | `5` | Top-N features from RF importances used per timepoint |
| `--n_bootstrap` | `1000` | Bootstrap iterations for 95% CIs and p-values |
| `--C` | `1.0` | LR regularization strength (inverse) |
| `--random_seed` | `42` | Random seed for reproducibility |

**Output files:**

| File | Content |
|---|---|
| `logreg_performance.csv` | LOO-CV accuracy per timepoint |
| `logreg_coefficients.csv` | Coefficients, 95% CI, odds ratios, bootstrap p-values |
| `logreg_probabilities.csv` | Per-animal classification probabilities |

---

## Swim features

| Feature | Description | Unit |
|---|---|---|
| `hl_frequency` | Hindlimb stroke frequency | Hz |
| `hl_amp_avg` | Hindlimb stroke amplitude | cm |
| `hl_efficiency_avg` | Hindlimb stroke efficiency (net forward displacement per cycle) | cm |
| `hl_amp_regularity_cv` | Stroke amplitude regularity (coefficient of variation) | % |
| `hl_ext_speed_avg` | Extension speed (peak-to-trough slope) | cm/s |
| `hl_ext_speed_regularity_cv` | Extension speed regularity | % |
| `hl_flex_speed_avg` | Flexion speed | cm/s |
| `hl_flex_speed_regularity_cv` | Flexion speed regularity | % |
| `hl_loop_area_avg` | Stroke loop area (XY trajectory per cycle) | cm² |
| `hl_loop_area_regularity_cv` | Loop area regularity | % |
| `hl_loop_shape_regularity_cv` | Loop shape regularity (PCA circularity CV) | % |
| `hl_lag` | Inter-limb coordination lag (cross-correlation peak lag) | s |
| `hl_corr_strength` | Inter-limb coordination strength (cross-correlation peak) | r |
| `hl_symmetry` | Left/right symmetry index | — |
| `tail_frequency` | Tail oscillation frequency (spectrogram peak) | Hz |
| `tail_oscillation_speed` | Tail oscillation RMS speed | cm/s |
| `tail_lag_avg` | Hindlimb–tail coordination lag | s |
| `tail_coord_strength_avg` | Hindlimb–tail coordination strength | r |
| `forelimb_use_both` | Percentage of frames with active forelimb displacement | % |
| `body_axis_angle_mean` | Mean body heading angle (nose–tail vector) | ° |
| `body_axis_angle_std` | Heading stability | ° |
| `body_curvature_mean` | Mean body curvature (nose–sternum–tail angle) | ° |
| `body_curvature_std` | Curvature stability | ° |
| `swim_speed` | Overall swim speed (lane length / trial duration) | cm/s |

---

## Pre-trained Random Forest models

Seven models were trained on WT and SOD1 mice at postnatal timepoints p45–p127.
Each `.joblib` bundle contains:

- `model` — trained `sklearn.ensemble.RandomForestClassifier`
- `feature_cols` — ordered list of feature names used during training
- `n_train`, `n_wt`, `n_sod1` — training set sizes

| File | Timepoint |
|---|---|
| `step10b_rf_model_p45.joblib` | Postnatal day 45 |
| `step10b_rf_model_p58.joblib` | Postnatal day 58 |
| `step10b_rf_model_p73.joblib` | Postnatal day 73 |
| `step10b_rf_model_p85.joblib` | Postnatal day 85 |
| `step10b_rf_model_p99.joblib` | Postnatal day 99 |
| `step10b_rf_model_p113.joblib` | Postnatal day 113 |
| `step10b_rf_model_p127.joblib` | Postnatal day 127 |

When an animal's timepoint does not exactly match a trained model, the pipeline
automatically selects the nearest model by age (in days).

---

## Citation

If you use this pipeline, please cite:

```
[Your citation in the journal's preferred format]
```

---

## License

MIT License. See `LICENSE` for details.
