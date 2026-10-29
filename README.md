# GlucoScope

**An explainable, reproducible diabetes-status ML research lab.**

Live application: https://glucoscope-mushtahir-ml.mushtahirkabir.chatgpt.site

Author: Mushtahir Kabir & team. Educational research only; not a medical diagnostic system.

## What it does

GlucoScope trains four classifiers on the UCI CDC Diabetes Health Indicators export and makes the investigation interactive. Visitors can explore fictional profiles, compare errors, inspect feature sensitivity, simulate capacity-limited ranking, investigate the dataset, and learn the math in ten chunks. Predictions execute locally in the browser. There is no health-data storage or prediction API call.

## Measured results

- 253,680 observations; 21 features; 50,736 held-out test records.
- Validation-selected Random forest: test AP **0.4308**, ROC-AUC **0.8274**.
- **5.55% relative AP improvement** versus full logistic regression on the same held-out test set.
- **61.90% fewer inputs**, retaining **98.45%** of full logistic AP in a prespecified short-form experiment.
- At 20% selection capacity, captures **2.41% more positive survey labels** than logistic regression (4,001 versus 3,907).

These are retrospective sample-level results, not evidence of clinical outcomes or future disease prediction. We retain the validation winner even though boosting scored slightly higher on final test AP.

| Model | Accuracy | Precision | Recall | F1 | Average precision | ROC-AUC | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic regression | 0.7885 | 0.3564 | 0.6432 | 0.4587 | 0.4081 | 0.8230 | 0.0989 |
| Decision tree | 0.7792 | 0.3475 | 0.6663 | 0.4567 | 0.4126 | 0.8180 | 0.1001 |
| Random forest | 0.8024 | 0.3742 | 0.6220 | 0.4673 | 0.4308 | 0.8274 | 0.0985 |
| Gradient boosting | 0.7996 | 0.3718 | 0.6356 | 0.4692 | 0.4321 | 0.8287 | 0.0971 |
| Short-form logistic | 0.7836 | 0.3500 | 0.6455 | 0.4539 | 0.4018 | 0.8186 | 0.0995 |
| Always-negative baseline | 0.8607 | 0.0000 | 0.0000 | 0.0000 | 0.1393 | 0.5000 | 0.1199 |


## Quick start: use the trained app

```bash
python -m http.server 8000 --directory dist
```

Open http://localhost:8000. No Python ML dependencies are needed to serve the already-trained app.

## Reproduce training and verification

```bash
python -m venv .venv
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python src/train.py
python src/build_docs.py
npm ci
npm test
python tests/check_artifacts.py
```

Python 3.11+ and Node.js 22 are appropriate. Pinning package versions improves reproducibility; a few minutes of CPU training and several GB of available memory are advisable. The CSV downloads automatically and is checked against a pinned SHA-256. Raw rows and Python model binaries are not committed. All deployment-ready parameters are in dist/assets/models.json. Run training to recreate joblib artifacts.

## Architecture

`UCI CSV → schema validation → grouped split → training CV → sigmoid calibration → validation selection → locked test → JSON export → browser inference`

- `src/train.py`: reproducible experiment and export.
- `src/schema.py`: survey coding and short-form features.
- `dist/assets/inference.mjs`: logistic, tree, forest, and boosting inference.
- `dist/assets/app.mjs`: six interactive research pages.
- `dist/assets/results.json`: measured metrics, curves, importance, subgroup results.
- `artifacts/frozen_selection.json`: winner and thresholds recorded before test evaluation.
- `artifacts/split_assignments.csv`: reproducible row-role membership; no survey answers.
- `tests/`: model parity, DOM interactions, and artifact checks.
- `notebooks/investigation.ipynb`: exploration and report figures.
- `docs/research-report.md`: complete course report.
- `docs/teaching-guide.md`: worked math and presentation preparation.
- `docs/rubric-checklist.md`: assignment mapping.

## Evaluation design

60% training / 10% calibration / 10% validation / 20% final test. Identical full feature profiles stay within one role. Three-fold stratified group CV tunes two candidates per algorithm using AP. Calibration fits only on calibration data; validation selects the full-model winner and each F1 threshold. No choices are revised using the final test. Class balance is preserved rather than artificially equalized.

## Deploy elsewhere

Host the contents of `dist/` on a static host, including GitHub Pages. This site uses hash routing, so it needs no special route rewrites. For Docker:

```bash
docker build -t glucoscope .
docker run --rm -p 8080:80 glucoscope
```

## GitHub handoff

This project is source-controlled for Sites hosting, but it is not yet in your personal GitHub account. Create an empty repository and publish the portable source under your account. A portable ZIP in `dist/downloads/` excludes host-specific identity and raw data. Do not publish credentials or commit downloaded data without checking its license.

## Limits and intended use

Research and education, not diagnosis, medical advice, future-risk forecasting, or validated clinical outreach. Inputs are self-reported and historical. UCI metadata has provenance inconsistencies, documented in the report. Survey weighting is unavailable in this processed export. No independent clinical cohort has been evaluated. Correlations and feature sensitivity do not identify causes. No claim of reduced healthcare costs or disease incidence is made.

## Attribution

[UCI dataset 891](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators), DOI 10.24432/C53919. The data provider's terms govern the dataset. Project code is MIT-licensed; that license does not override third-party data or dependency licenses.
