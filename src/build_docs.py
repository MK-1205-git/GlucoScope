"""Generate documentation and a portable source download from frozen measured results."""
from pathlib import Path
import json, shutil, zipfile

ROOT=Path(__file__).resolve().parents[1]
def build():
    r=json.loads((ROOT/'dist/assets/results.json').read_text())
    a=json.loads((ROOT/'dist/assets/audit.json').read_text())
    models={m['name']:m for m in r['models']}; w=models[r['winner']]; lr=models['Logistic regression']; sh=models['Short-form logistic']
    pct=lambda v:f'{100*v:.2f}%'
    wc=w['capacity'][3]; lc=lr['capacity'][3]
    table='| Model | Accuracy | Precision | Recall | F1 | Average precision | ROC-AUC | Brier |\n|---|---:|---:|---:|---:|---:|---:|---:|\n'
    for m in r['models']:
        table+='| '+m['name']+' | '+' | '.join(f"{m['test'][k]:.4f}" for k in ['accuracy','precision','recall','f1','ap','auc','brier'])+' |\n'
    table+='| Always-negative baseline | '+' | '.join(f"{r['dummy'][k]:.4f}" for k in ['accuracy','precision','recall','f1','ap','auc','brier'])+' |\n'
    splits='\n'.join(f"- {k}: {v['n']:,} rows; {v['positive']:,} positive labels ({pct(v['prevalence'])})." for k,v in a['splits'].items())
    params='\n'.join(f"- {m['name']}: {json.dumps(m['params'])}; training CV AP {m['cv_ap']:.4f}; validation AP {m['validation']['ap']:.4f}; threshold {m['test']['threshold']:.6f}." for m in r['models'])
    cms='\n'.join(f"- {m['name']}: {m['test']['confusion']}" for m in r['models'])
    importances=sorted(w['importance'],key=lambda v:v['mean'],reverse=True)
    importance_text='\n'.join(f"- {i['feature']}: mean AP decrease {i['mean']:.4f}, shuffle SD {i['std']:.4f}." for i in importances[:8])
    report=f'''# GlucoScope — Diabetes-status classification investigation

Author: Mushtahir Kabir & team. Research edition 1.0. Evaluation seed: 42.

## A. Define the problem

Can health, demographic, and lifestyle survey characteristics classify whether a respondent reported prediabetes or diabetes? We compare four algorithms and quantify a shorter input set's performance. This addresses reproducible analysis and model comparison, not medical diagnosis or future disease prediction. The live tool helps visitors inspect the evidence and reproduce an investigation.

The target follows the downloaded UCI export's documentation: Diabetes_binary = 0 (no diabetes) versus 1 (prediabetes or diabetes). The score is not a validated personal medical probability. No claim is made about diagnosing previously undiagnosed disease, treatment effectiveness, or improved patient outcomes.

## B. Understand the dataset

Source: [UCI dataset 891](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators). Direct CSV: {a['source']}.

The verified file has {a['rows']:,} rows, 21 predictor columns, one target, and one ID. All columns load as integers. Missing entries: {a['missing']}. Target counts: {a['negative']:,} negative and {a['positive']:,} positive ({pct(a['prevalence'])}). The website provides the full dictionary and per-feature frequency/positive-rate tables. An exploratory notebook reproduces distributions and visualizations.

There are {a['duplicate_feature_rows']:,} input rows repeating an earlier 21-feature profile, {a['duplicate_feature_target_rows']:,} repeated feature-plus-target rows, and {a['conflicting_profiles']:,} distinct feature profiles associated with both labels. IDs are row indices, not usable longitudinal person identifiers. Identical survey responses need not be duplicate people.

**Provenance caveat:** UCI's summary prose, feature metadata, target descriptions, and linked survey-year information are inconsistent. We pin the actual binary CSV and use its explicit variable table. We do not assert a verified survey year or relabel the target as laboratory-confirmed disease. Resolving upstream metadata with the original provider is a future provenance improvement.

SHA-256: `{a['source_sha256']}`.

## C. Data preparation

1. Remove ID and target from predictors. Validate expected schema, integer categories, ranges, binary outcome, and absence of missing values. Stop if the source checksum changes.
2. Retain rare BMI values (observed range 12–98) because no evidence establishes that every extreme value is invalid. Do not silently winsorize them.
3. Keep repeated observations, but hash all 21 features and assign identical profiles to the same role. Outcome labels are excluded from the grouping key. Assert that no group overlaps roles.
4. Use approximately stratified group splitting; observed counts are listed below. No survey weights are available in this processed export, so results are sample-level.
5. Fit StandardScaler inside each logistic training pipeline. Logistic models treat ordered codes as ordered numeric inputs; this also assumes linear effects per category step. Trees require no scaling. Binary features retain 0/1 codes. No imputation is necessary.
6. Preserve observed class imbalance in all roles. We use AP for model selection and a validation F1 operating threshold; no SMOTE, class balancing, or downsampling is applied. Alternative weighting is future work, not a claimed completed experiment.
7. Prespecify eight short-form features: {', '.join(r['short_features'])}. This subset is a limited experiment, not an exhaustive optimal feature search.

## D. Split and experimental protocol

{splits}

Ten stratified group folds allocate six to training, one to calibration, one to validation, and two to test. Within training, three-fold StratifiedGroupKFold cross-validation and GridSearchCV select hyperparameters using average precision. Every scaler is fit within the training fold. CV has two candidates per algorithm, intentionally limiting search cost.

Fit a sigmoid calibration on the calibration role using FrozenEstimator. Score the validation role and choose a classification threshold maximizing F1. Select the full model with the highest validation AP. Save the chosen model name and thresholds before evaluating test data. Never reselect based on test performance. The short model is an additional ablation and cannot replace the full-model winner through test inspection.

{params}

## E. Algorithms

- Logistic regression: standardized inputs, L2 regularization; C in {{0.1, 1.0}}, max_iter=1000.
- Decision tree: Gini criterion, max_depth in {{4,8}}, min_samples_leaf=100.
- Random forest: 80 trees, max_depth in {{8,12}}, min_samples_leaf=30; bootstrap and default random feature subsampling.
- Gradient boosting: 70 trees, learning_rate=0.07, max_depth in {{2,3}}, min_samples_leaf=60, subsample=0.8; logistic loss.
- Short-form logistic regression: same search and procedure on eight prespecified inputs.
- Dummy baseline: constant training prevalence probability, threshold 0.5, hence always negative.

## F. Final evaluation

{table}

Accuracy measures overall correctness. Precision is the positive predictive value of a flag; recall is the fraction of positive labels detected. F1 balances precision and recall. AP summarizes precision over changes in recall; ROC-AUC measures ranking discrimination. Brier score assesses probability error. The high dummy accuracy and zero dummy recall demonstrate why accuracy alone is insufficient.

Confusion matrices below use rows [actual negative, actual positive] and columns [predicted negative, predicted positive]:

{cms}

The application visualizes precision–recall curves, probability calibration, and confusion matrices. ROC curve coordinates are in results.json and visualized in the notebook. The saved plots are descriptive final-test evidence, not a reason to retune.

## G. Comparison and measurable contributions

**Validation-selected winner:** {r['winner']}. It achieved test AP {w['test']['ap']:.4f}, ROC-AUC {w['test']['auc']:.4f}, F1 {w['test']['f1']:.4f}, recall {pct(w['test']['recall'])}, and precision {pct(w['test']['precision'])}. Gradient boosting has slightly higher final test AP, but did not win the validation selection; we report that instead of switching models after testing.

**Full-model baseline comparison:** selected-model test AP was {pct((w['test']['ap']-lr['test']['ap'])/lr['test']['ap'])} higher relative to logistic regression. Absolute AP difference: {w['test']['ap']-lr['test']['ap']:.4f}. A 95% feature-profile cluster-bootstrap interval for the difference is {r['uncertainty']['ap_difference_vs_logistic_ci'][0]:.4f} to {r['uncertainty']['ap_difference_vs_logistic_ci'][1]:.4f}. This is sample-level retrospective performance, not clinical benefit.

**Shorter inputs:** reducing 21 inputs to 8 removes {pct(r['question_reduction'])} of inputs while retaining {pct(sh['test']['ap']/lr['test']['ap'])} of full logistic AP ({sh['test']['ap']:.4f} versus {lr['test']['ap']:.4f}). A small loss remains. Completion time and user satisfaction were not measured.

**Capacity simulation:** at 20% capacity, all models select {wc['selected']:,} out of {r['test_n']:,} records. Random forest captures {wc['captured']:,} positive labels versus {lc['captured']:,} for logistic regression: {pct((wc['captured']-lc['captured'])/lc['captured'])} more at the same capacity. Random selection's expected yield is {wc['selected']*r['test_positive']/r['test_n']:.1f}. Positive labels are already known for evaluation; this does not prove useful real-world clinical outreach. Ties are broken by stable row order, not outcome.

## H. Interpretation, uncertainty, and trade-offs

Top selected-model permutation importance estimates (AP drop; 1,200 deterministic test records and three shuffles):

{importance_text}

The forest can capture nonlinear relationships and feature interactions that linear log odds may miss. The performance gap is modest, not a dramatic clinical breakthrough. Correlated predictors complicate importance interpretation; important features do not establish causal drivers. General health and medical-history indicators may reflect already-established illness. This limits extrapolation to early detection.

Logistic regression is compact and easier to interpret. A single tree is transparent but less flexible. Forests and boosting offer richer functions but use more storage and explanation effort. The selected forest has substantial false positives and false negatives; a public demo must expose these errors.

Calibration is fit without test data. Brier score before and after calibration is reported in the dashboard; it does not improve for every model. A limited sigmoid mapping may not correct every distortion. Preprocessing impact beyond the explicit full-versus-short comparison was not isolated in a separate controlled ablation; we do not claim a causal performance gain from scaling alone.

Selected-model recall 95% interval: {pct(r['uncertainty']['winner_recall_ci'][0])}–{pct(r['uncertainty']['winner_recall_ci'][1])}. Method: 200 feature-profile cluster-bootstrap replicates, fixed fitted models, percentile interval. This excludes uncertainty from retraining, source bias, and deployment drift. Subgroup tables report sample counts, recall, false-positive rates, AP, and Brier using the same threshold. They are descriptive, not a formal fairness certification.

## I. Software and deployment

Python trains and evaluates models. Exported JSON contains exact coefficients, standardization, tree nodes, and calibration parameters. JavaScript executes those fitted models in the visitor's browser; no backend receives questionnaire answers. The same selected model drives the profile explorer and reported official results. All five models can be explored. Sensitivity analysis substitutes one input with a descriptive dataset median, not a causal intervention.

Verification includes 500 Python/JavaScript comparisons on artificial profiles, input-range rejection, route and interaction checks in a DOM environment, artifact consistency, and local download availability. Visual browser QA is unavailable in this execution environment; no screenshot-based layout certification is claimed. Python inference timing is a local benchmark only, not a public API latency claim.

The source includes GitHub Actions verification, a Dockerfile for static hosting, a notebook, source download, and reproduction instructions. The complete portable source archive can be imported into GitHub. Sites hosting and GitHub are separate destinations.

## J. Conclusion

We investigated binary survey-status classification with 253,680 public records and 21 inputs, compared four algorithms, and selected random forest using validation AP. Its final AP is {w['test']['ap']:.4f}. The eight-input logistic alternative retains {pct(sh['test']['ap']/lr['test']['ap'])} of full logistic AP. Findings demonstrate a reproducible trade-off between ranking quality, input burden, and classification errors. They do not establish diagnosis, future disease prediction, or improved patient outcomes.

Future work: independently verify provenance and target construction, evaluate another cohort, compare ordinal versus one-hot encoding, evaluate class weighting under the same locked protocol, study short-form usability, report repeated-training uncertainty, and collaborate with clinicians before proposing a medical intended use. Additional test-driven iterations require a fresh untouched evaluation set.

## Sources and attribution

- [UCI Machine Learning Repository: CDC Diabetes Health Indicators](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators), DOI 10.24432/C53919.
- [Original linked dataset page](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset): consult provider licensing/acknowledgment terms before redistributing raw data. Raw rows are excluded from this repository and site.
- [scikit-learn: common pitfalls and data leakage](https://scikit-learn.org/stable/common_pitfalls.html).
- [scikit-learn: probability calibration](https://scikit-learn.org/stable/modules/calibration.html).
- [CDC: BRFSS data quality](https://www.cdc.gov/brfss/publications/data_qvr.htm).

Environment recorded at training: {json.dumps(a['versions'])}. Training/evaluation elapsed time in this environment: {r['runtime_seconds']:.1f} seconds. Timing is not a performance guarantee.
'''
    (ROOT/'docs/research-report.md').write_text(report)
    readme=f'''# GlucoScope

**An explainable, reproducible diabetes-status ML research lab.**

Live application: https://glucoscope-mushtahir-ml.mushtahirkabir.chatgpt.site

Author: Mushtahir Kabir & team. Educational research only; not a medical diagnostic system.

## What it does

GlucoScope trains four classifiers on the UCI CDC Diabetes Health Indicators export and makes the investigation interactive. Visitors can explore fictional profiles, compare errors, inspect feature sensitivity, simulate capacity-limited ranking, investigate the dataset, and learn the math in ten chunks. Predictions execute locally in the browser. There is no health-data storage or prediction API call.

## Measured results

- {a['rows']:,} observations; 21 features; {r['test_n']:,} held-out test records.
- Validation-selected {r['winner']}: test AP **{w['test']['ap']:.4f}**, ROC-AUC **{w['test']['auc']:.4f}**.
- **{pct((w['test']['ap']-lr['test']['ap'])/lr['test']['ap'])} relative AP improvement** versus full logistic regression on the same held-out test set.
- **{pct(r['question_reduction'])} fewer inputs**, retaining **{pct(sh['test']['ap']/lr['test']['ap'])}** of full logistic AP in a prespecified short-form experiment.
- At 20% selection capacity, captures **{pct((wc['captured']-lc['captured'])/lc['captured'])} more positive survey labels** than logistic regression ({wc['captured']:,} versus {lc['captured']:,}).

These are retrospective sample-level results, not evidence of clinical outcomes or future disease prediction. We retain the validation winner even though boosting scored slightly higher on final test AP.

{table}

## Quick start: use the trained app

```bash
python -m http.server 8000 --directory dist
```

Open http://localhost:8000. No Python ML dependencies are needed to serve the already-trained app.

## Reproduce training and verification

```bash
python -m venv .venv
source .venv/bin/activate
# Windows PowerShell: .venv\\Scripts\\Activate.ps1
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
'''
    (ROOT/'README.md').write_text(readme)
    downloads=ROOT/'dist/downloads'; downloads.mkdir(exist_ok=True)
    for name in ['research-report.md','teaching-guide.md','rubric-checklist.md']:
        shutil.copyfile(ROOT/'docs'/name,downloads/name)
    # Portable archive intentionally excludes hosting identity, git, original data, and itself.
    with zipfile.ZipFile(downloads/'glucoscope-source.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(ROOT.rglob('*')):
            rel=p.relative_to(ROOT)
            if not p.is_file() or any(part in {'.git','.openai','node_modules','__pycache__','data'} for part in rel.parts): continue
            if p.suffix in {'.joblib','.npz','.zip','.pyc'}: continue
            z.write(p,Path('glucoscope')/rel)
    print('Built report, README, teaching downloads, and portable source ZIP.')

if __name__=='__main__': build()
