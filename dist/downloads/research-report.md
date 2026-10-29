# GlucoScope — Diabetes-status classification investigation

Author: Mushtahir Kabir & team. Research edition 1.0. Evaluation seed: 42.

## A. Define the problem

Can health, demographic, and lifestyle survey characteristics classify whether a respondent reported prediabetes or diabetes? We compare four algorithms and quantify a shorter input set's performance. This addresses reproducible analysis and model comparison, not medical diagnosis or future disease prediction. The live tool helps visitors inspect the evidence and reproduce an investigation.

The target follows the downloaded UCI export's documentation: Diabetes_binary = 0 (no diabetes) versus 1 (prediabetes or diabetes). The score is not a validated personal medical probability. No claim is made about diagnosing previously undiagnosed disease, treatment effectiveness, or improved patient outcomes.

## B. Understand the dataset

Source: [UCI dataset 891](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators). Direct CSV: https://archive.ics.uci.edu/static/public/891/data.csv.

The verified file has 253,680 rows, 21 predictor columns, one target, and one ID. All columns load as integers. Missing entries: 0. Target counts: 218,334 negative and 35,346 positive (13.93%). The website provides the full dictionary and per-feature frequency/positive-rate tables. An exploratory notebook reproduces distributions and visualizations.

There are 25,772 input rows repeating an earlier 21-feature profile, 24,206 repeated feature-plus-target rows, and 1,566 distinct feature profiles associated with both labels. IDs are row indices, not usable longitudinal person identifiers. Identical survey responses need not be duplicate people.

**Provenance caveat:** UCI's summary prose, feature metadata, target descriptions, and linked survey-year information are inconsistent. We pin the actual binary CSV and use its explicit variable table. We do not assert a verified survey year or relabel the target as laboratory-confirmed disease. Resolving upstream metadata with the original provider is a future provenance improvement.

SHA-256: `9f71fda9d4ae5f4878c99b9233b6a16accfa9a17c194116a6b78100540934964`.

## C. Data preparation

1. Remove ID and target from predictors. Validate expected schema, integer categories, ranges, binary outcome, and absence of missing values. Stop if the source checksum changes.
2. Retain rare BMI values (observed range 12–98) because no evidence establishes that every extreme value is invalid. Do not silently winsorize them.
3. Keep repeated observations, but hash all 21 features and assign identical profiles to the same role. Outcome labels are excluded from the grouping key. Assert that no group overlaps roles.
4. Use approximately stratified group splitting; observed counts are listed below. No survey weights are available in this processed export, so results are sample-level.
5. Fit StandardScaler inside each logistic training pipeline. Logistic models treat ordered codes as ordered numeric inputs; this also assumes linear effects per category step. Trees require no scaling. Binary features retain 0/1 codes. No imputation is necessary.
6. Preserve observed class imbalance in all roles. We use AP for model selection and a validation F1 operating threshold; no SMOTE, class balancing, or downsampling is applied. Alternative weighting is future work, not a claimed completed experiment.
7. Prespecify eight short-form features: HighBP, HighChol, BMI, GenHlth, Age, DiffWalk, PhysHlth, Sex. This subset is a limited experiment, not an exhaustive optimal feature search.

## D. Split and experimental protocol

- train: 152,208 rows; 21,209 positive labels (13.93%).
- calibration: 25,368 rows; 3,534 positive labels (13.93%).
- validation: 25,368 rows; 3,534 positive labels (13.93%).
- test: 50,736 rows; 7,069 positive labels (13.93%).

Ten stratified group folds allocate six to training, one to calibration, one to validation, and two to test. Within training, three-fold StratifiedGroupKFold cross-validation and GridSearchCV select hyperparameters using average precision. Every scaler is fit within the training fold. CV has two candidates per algorithm, intentionally limiting search cost.

Fit a sigmoid calibration on the calibration role using FrozenEstimator. Score the validation role and choose a classification threshold maximizing F1. Select the full model with the highest validation AP. Save the chosen model name and thresholds before evaluating test data. Never reselect based on test performance. The short model is an additional ablation and cannot replace the full-model winner through test inspection.

- Logistic regression: {"logisticregression__C": 0.1}; training CV AP 0.4031; validation AP 0.4046; threshold 0.193610.
- Decision tree: {"max_depth": 8}; training CV AP 0.4014; validation AP 0.4054; threshold 0.144304.
- Random forest: {"max_depth": 12}; training CV AP 0.4257; validation AP 0.4232; threshold 0.172295.
- Gradient boosting: {"max_depth": 3}; training CV AP 0.4267; validation AP 0.4220; threshold 0.204706.
- Short-form logistic: {"logisticregression__C": 0.1}; training CV AP 0.3945; validation AP 0.3961; threshold 0.191051.

## E. Algorithms

- Logistic regression: standardized inputs, L2 regularization; C in {0.1, 1.0}, max_iter=1000.
- Decision tree: Gini criterion, max_depth in {4,8}, min_samples_leaf=100.
- Random forest: 80 trees, max_depth in {8,12}, min_samples_leaf=30; bootstrap and default random feature subsampling.
- Gradient boosting: 70 trees, learning_rate=0.07, max_depth in {2,3}, min_samples_leaf=60, subsample=0.8; logistic loss.
- Short-form logistic regression: same search and procedure on eight prespecified inputs.
- Dummy baseline: constant training prevalence probability, threshold 0.5, hence always negative.

## F. Final evaluation

| Model | Accuracy | Precision | Recall | F1 | Average precision | ROC-AUC | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic regression | 0.7885 | 0.3564 | 0.6432 | 0.4587 | 0.4081 | 0.8230 | 0.0989 |
| Decision tree | 0.7792 | 0.3475 | 0.6663 | 0.4567 | 0.4126 | 0.8180 | 0.1001 |
| Random forest | 0.8024 | 0.3742 | 0.6220 | 0.4673 | 0.4308 | 0.8274 | 0.0985 |
| Gradient boosting | 0.7996 | 0.3718 | 0.6356 | 0.4692 | 0.4321 | 0.8287 | 0.0971 |
| Short-form logistic | 0.7836 | 0.3500 | 0.6455 | 0.4539 | 0.4018 | 0.8186 | 0.0995 |
| Always-negative baseline | 0.8607 | 0.0000 | 0.0000 | 0.0000 | 0.1393 | 0.5000 | 0.1199 |


Accuracy measures overall correctness. Precision is the positive predictive value of a flag; recall is the fraction of positive labels detected. F1 balances precision and recall. AP summarizes precision over changes in recall; ROC-AUC measures ranking discrimination. Brier score assesses probability error. The high dummy accuracy and zero dummy recall demonstrate why accuracy alone is insufficient.

Confusion matrices below use rows [actual negative, actual positive] and columns [predicted negative, predicted positive]:

- Logistic regression: [[35456, 8211], [2522, 4547]]
- Decision tree: [[34822, 8845], [2359, 4710]]
- Random forest: [[36314, 7353], [2672, 4397]]
- Gradient boosting: [[36076, 7591], [2576, 4493]]
- Short-form logistic: [[35192, 8475], [2506, 4563]]

The application visualizes precision–recall curves, probability calibration, and confusion matrices. ROC curve coordinates are in results.json and visualized in the notebook. The saved plots are descriptive final-test evidence, not a reason to retune.

## G. Comparison and measurable contributions

**Validation-selected winner:** Random forest. It achieved test AP 0.4308, ROC-AUC 0.8274, F1 0.4673, recall 62.20%, and precision 37.42%. Gradient boosting has slightly higher final test AP, but did not win the validation selection; we report that instead of switching models after testing.

**Full-model baseline comparison:** selected-model test AP was 5.55% higher relative to logistic regression. Absolute AP difference: 0.0227. A 95% feature-profile cluster-bootstrap interval for the difference is 0.0161 to 0.0285. This is sample-level retrospective performance, not clinical benefit.

**Shorter inputs:** reducing 21 inputs to 8 removes 61.90% of inputs while retaining 98.45% of full logistic AP (0.4018 versus 0.4081). A small loss remains. Completion time and user satisfaction were not measured.

**Capacity simulation:** at 20% capacity, all models select 10,148 out of 50,736 records. Random forest captures 4,001 positive labels versus 3,907 for logistic regression: 2.41% more at the same capacity. Random selection's expected yield is 1413.9. Positive labels are already known for evaluation; this does not prove useful real-world clinical outreach. Ties are broken by stable row order, not outcome.

## H. Interpretation, uncertainty, and trade-offs

Top selected-model permutation importance estimates (AP drop; 1,200 deterministic test records and three shuffles):

- BMI: mean AP decrease 0.0970, shuffle SD 0.0251.
- GenHlth: mean AP decrease 0.0618, shuffle SD 0.0107.
- HighChol: mean AP decrease 0.0310, shuffle SD 0.0040.
- HighBP: mean AP decrease 0.0266, shuffle SD 0.0052.
- Age: mean AP decrease 0.0193, shuffle SD 0.0082.
- HeartDiseaseorAttack: mean AP decrease 0.0091, shuffle SD 0.0035.
- PhysHlth: mean AP decrease 0.0077, shuffle SD 0.0033.
- DiffWalk: mean AP decrease 0.0064, shuffle SD 0.0052.

The forest can capture nonlinear relationships and feature interactions that linear log odds may miss. The performance gap is modest, not a dramatic clinical breakthrough. Correlated predictors complicate importance interpretation; important features do not establish causal drivers. General health and medical-history indicators may reflect already-established illness. This limits extrapolation to early detection.

Logistic regression is compact and easier to interpret. A single tree is transparent but less flexible. Forests and boosting offer richer functions but use more storage and explanation effort. The selected forest has substantial false positives and false negatives; a public demo must expose these errors.

Calibration is fit without test data. Brier score before and after calibration is reported in the dashboard; it does not improve for every model. A limited sigmoid mapping may not correct every distortion. Preprocessing impact beyond the explicit full-versus-short comparison was not isolated in a separate controlled ablation; we do not claim a causal performance gain from scaling alone.

Selected-model recall 95% interval: 61.13%–63.20%. Method: 200 feature-profile cluster-bootstrap replicates, fixed fitted models, percentile interval. This excludes uncertainty from retraining, source bias, and deployment drift. Subgroup tables report sample counts, recall, false-positive rates, AP, and Brier using the same threshold. They are descriptive, not a formal fairness certification.

## I. Software and deployment

Python trains and evaluates models. Exported JSON contains exact coefficients, standardization, tree nodes, and calibration parameters. JavaScript executes those fitted models in the visitor's browser; no backend receives questionnaire answers. The same selected model drives the profile explorer and reported official results. All five models can be explored. Sensitivity analysis substitutes one input with a descriptive dataset median, not a causal intervention.

Verification includes 500 Python/JavaScript comparisons on artificial profiles, input-range rejection, route and interaction checks in a DOM environment, artifact consistency, and local download availability. Visual browser QA is unavailable in this execution environment; no screenshot-based layout certification is claimed. Python inference timing is a local benchmark only, not a public API latency claim.

The source includes GitHub Actions verification, a Dockerfile for static hosting, a notebook, source download, and reproduction instructions. The complete portable source archive can be imported into GitHub. Sites hosting and GitHub are separate destinations.

## J. Conclusion

We investigated binary survey-status classification with 253,680 public records and 21 inputs, compared four algorithms, and selected random forest using validation AP. Its final AP is 0.4308. The eight-input logistic alternative retains 98.45% of full logistic AP. Findings demonstrate a reproducible trade-off between ranking quality, input burden, and classification errors. They do not establish diagnosis, future disease prediction, or improved patient outcomes.

Future work: independently verify provenance and target construction, evaluate another cohort, compare ordinal versus one-hot encoding, evaluate class weighting under the same locked protocol, study short-form usability, report repeated-training uncertainty, and collaborate with clinicians before proposing a medical intended use. Additional test-driven iterations require a fresh untouched evaluation set.

## Sources and attribution

- [UCI Machine Learning Repository: CDC Diabetes Health Indicators](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators), DOI 10.24432/C53919.
- [Original linked dataset page](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset): consult provider licensing/acknowledgment terms before redistributing raw data. Raw rows are excluded from this repository and site.
- [scikit-learn: common pitfalls and data leakage](https://scikit-learn.org/stable/common_pitfalls.html).
- [scikit-learn: probability calibration](https://scikit-learn.org/stable/modules/calibration.html).
- [CDC: BRFSS data quality](https://www.cdc.gov/brfss/publications/data_qvr.htm).

Environment recorded at training: {"python": "3.12.14", "sklearn": "1.8.0", "numpy": "2.3.5", "pandas": "2.2.3"}. Training/evaluation elapsed time in this environment: 170.5 seconds. Timing is not a performance guarantee.
