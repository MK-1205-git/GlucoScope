# GlucoScope: learn it, teach it, present it

Work through one chunk at a time. You do not need to learn all of this in one sitting. Open the website beside this guide. Each chunk follows: idea → math → code → demonstration → explain it back.

## Chunk 1 — What are we trying to learn?

One row represents one survey respondent. A **feature** is one input, such as BMI. The **target** is the answer we want to predict. The dataset has 21 features, and the target is binary: 0 means no diabetes reported, 1 means prediabetes or diabetes according to the UCI binary-target documentation.

We ask: can health, demographic, and lifestyle answers predict this recorded label? We are not asking whether the respondent develops diabetes later. There are no future follow-up dates in this export.

Notation: X is the table of inputs; y is the column of true labels. For respondent i, x_i is their input vector and y_i is 0 or 1. A model maps x_i to a probability p_i. A threshold turns p_i into a predicted label.

Code: `src/schema.py` defines the input meanings. `src/train.py` creates X and y and excludes ID.

Demo: open Dataset & methods. Point to the class distribution and explain why the label is not a medical test.

Explain back: what would need to change if we wanted to predict disease five years in the future?

## Chunk 2 — Keep the exam answers hidden

Training is practice; validation chooses a model; test is the final exam. Calibration is a separate task: adjust score-to-probability mapping without refitting the underlying trees or coefficients.

Our four roles are approximately 60% training, 10% calibration, 10% validation, and 20% test. Training uses three-fold cross-validation: fit on two folds and score the third, repeating so each fold serves once as a check. Each candidate's average precision is averaged over the three folds. The best candidate is refit on the full training portion.

Identical feature profiles stay together across all roles and CV folds. This does not prove they are the same person; it prevents identical input patterns from appearing in both fit and evaluation data. Legitimate repeated responses remain in the analysis.

Why standardization happens inside a pipeline: a training mean must be learned from training data. If test data determine the mean or selected features, test information has leaked into training.

Code: `StratifiedGroupKFold`, `GridSearchCV`, and `make_pipeline` in `src/train.py`.

Demo: show the four split counts. Explain why we chose random forest using validation AP even though boosting later had a slightly higher test AP.

Explain back: why would choosing the model with the best final test score weaken the evaluation?

## Chunk 3 — Logistic regression, step by step

1. Standardize each input: z_j = (x_j − μ_j) / σ_j. μ and σ come from the training fold.
2. Form a weighted sum: s = b + Σ_j w_j z_j.
3. Convert it to a probability: sigmoid(s) = 1 / (1 + exp(−s)).

Worked example with two features: z_1 = 1, z_2 = 0.5, weights 0.8 and 0.4, intercept −1. Then s = −1 + 0.8×1 + 0.4×0.5 = 0. The uncalibrated probability is 0.5. If s = 1, the probability is approximately 0.731.

Training penalizes incorrect confident predictions using binary cross-entropy:

L = −(1/N) Σ_i [y_i ln(p_i) + (1−y_i) ln(1−p_i)].

For a positive example, predicting 0.8 gives loss −ln(0.8) ≈ 0.223; predicting 0.1 gives loss ≈ 2.303. The second is much worse.

For the unregularized average loss, the gradient for weight j is (1/N) Σ_i (p_i−y_i)z_ij. The optimizer uses this information to reduce loss. L2 regularization discourages excessively large coefficients; smaller sklearn C means stronger regularization. We compare C=0.1 and C=1 using training cross-validation.

The deployed output also applies the separately learned sigmoid calibration, so it may differ from the raw logistic probability.

Code: training pipeline in `src/train.py`; the logistic branch of `predict` in `dist/assets/inference.mjs`.

Practice: if s = −1, calculate the uncalibrated probability. Answer: approximately 0.269.

## Chunk 4 — Decision trees and random forests

A decision tree divides observations using questions such as BMI ≤ 29.5. A leaf contains observations with similar paths; their class composition produces a probability estimate.

Our classification trees use Gini impurity: G = 1 − Σ_k p_k². A leaf with 50% positives has G = 0.5; a leaf with 10% positives has G = 0.18. A pure leaf has G = 0.

A proposed split is scored by the decrease in impurity, weighting child nodes by their sample counts. Example: parent G=0.5, equally sized children with G=0.18 each. Weighted child impurity is 0.18, so improvement is 0.32.

Deep trees can memorize peculiarities. We tune depth (4 or 8 for a single tree) and require at least 100 training observations per leaf.

A random forest averages predictions from 80 trees. Bootstrap resampling and randomized feature choices make the trees different. Averaging reduces variance; it does not guarantee better performance. We compare depth 8 and 12 and require at least 30 observations per leaf.

Code: `tree_export` writes split arrays and leaf probabilities. `walk` traverses one tree; `predict` averages the forest.

Practice: if three trees predict 0.2, 0.4, and 0.6, their uncalibrated forest prediction is 0.4. Our actual forest has 80 trees and then applies calibration.

## Chunk 5 — Gradient boosting

Boosting adds trees sequentially. For binary logistic loss, a useful intuition is that each new tree fits the direction needed to correct the current predictions. This differs from a forest, whose trees are trained using independent randomization rather than sequential corrections.

The additive score is F(x) = F_0 + η Σ_m tree_m(x). The initial score F_0 is the log odds of the training positive prevalence. η is the learning rate. The raw probability is sigmoid(F(x)).

Our setup uses 70 trees, learning rate 0.07, and subsampling fraction 0.8. We compare tree depth 2 versus 3. For logistic loss, the negative-gradient signal for an observation is y−p; exact leaf updates involve the implementation's optimization of the loss.

Worked additive-score example: F_0 = −1, learning rate 0.1, and tree outputs 2 and 1. F = −1 + 0.1×3 = −0.7, giving probability approximately 0.332 before calibration.

Code: the boosting branch of `export_model` and `predict`.

Explain back: why might increasing the number or depth of trees eventually hurt performance on unseen examples?

## Chunk 6 — Evaluate what went right and wrong

TP: positive label correctly predicted positive. FP: negative label incorrectly flagged positive. FN: positive label missed. TN: negative label correctly predicted negative.

- Accuracy = (TP + TN) / N.
- Precision = TP / (TP + FP).
- Recall = TP / (TP + FN).
- F1 = 2PR / (P + R) = 2TP / (2TP + FP + FN).
- False-positive rate = FP / (FP + TN).

Worked example: TP=40, FP=60, FN=10, TN=890. Accuracy is 930/1000 = 93%. Precision is 40/100 = 40%. Recall is 40/50 = 80%. F1 is 80/150 ≈ 53.3%. Notice how 93% accuracy hides 60 false alarms.

In our dataset, an always-negative classifier already reaches about 86.1% accuracy because negatives are common. It detects no positives. Our selected threshold sacrifices accuracy to achieve nonzero recall and a stronger F1 trade-off. This is deliberate, not an arithmetic error.

Average precision (AP) summarizes the precision–recall curve: AP = Σ_n (R_n−R_(n−1))P_n for successive recall levels. It is not the ordinary accuracy and not necessarily the trapezoidal area under a drawn PR curve.

ROC-AUC measures ranking discrimination using true-positive and false-positive rates. AP is especially informative here because positive labels are relatively uncommon.

Demo: read the selected model's actual confusion matrix and recompute its precision and recall.

## Chunk 7 — Calibration and the threshold

Ranking quality and probability quality are different. A model can rank examples correctly while its numerical probabilities are too large or small.

Sigmoid calibration fits p_cal = 1 / (1 + exp(a×s+b)) on the calibration portion. For sklearn models with a decision function (our logistic and boosting models), s is that score. For the single tree and forest, s is their raw positive-class probability. The coefficients a and b are learned, not manually assigned.

Brier score = (1/N) Σ_i (p_i−y_i)². For p=0.8 and y=1, the squared error is 0.04; for p=0.8 and y=0, it is 0.64. Lower is better. Calibration need not improve this metric on every new dataset; report both values honestly.

The decision threshold t converts a probability into a label: predict positive if p≥t. We select t by maximizing F1 on validation data. We then freeze it before testing. A value below 0.5 is not automatically wrong; it reflects the selected trade-off.

Practice: what happens to recall and false alarms when you lower a threshold? Typically both increase, although discrete scores can create ties.

## Chunk 8 — Quantify an improvement honestly

The eight-feature logistic experiment uses a prespecified subset: HighBP, HighChol, BMI, GenHlth, Age, DiffWalk, PhysHlth, Sex. It is not claimed to be the optimal subset.

Input reduction = (21−8)/21×100 = 61.9%. Performance retention = AP_short / AP_full_logistic ×100. The code computes the actual retention from the held-out results. This does not establish a 61.9% reduction in completion time; that would require timed users.

Capacity simulation: choose the top ceil(k×N) profiles by score. Recall@k = selected positive labels / all positive labels. Precision@k = selected positive labels / selected profiles. At the same capacity, relative gain = (model captured − baseline captured) / baseline captured ×100.

We compare against logistic regression and the expected yield of random selection. Random expectation is selected count × test prevalence; fractional expected counts are normal. Ties use stable input order, never the true labels.

We use 200 cluster-bootstrap repetitions for selected-model recall and AP difference versus logistic regression. Resample feature-profile groups with replacement, preserve records in a group, and use the 2.5th and 97.5th percentiles. This captures test-sample uncertainty conditional on these fitted models, not all uncertainty due to training or a new clinic.

Demo: move capacity from 20% to 100%. At 100%, everyone captures all positive labels, so the ranking advantage disappears.

## Chunk 9 — Explain the model without claiming causation

Permutation importance: compute AP, shuffle one feature, recompute AP, and report the drop. We use a deterministic 1,200-record test subset and three shuffles. This is exploratory interpretation after final evaluation, not feature selection for retraining. Correlated variables can substitute for each other and understate individual importance.

The profile explorer replaces one feature with its dataset median while holding the rest fixed. It shows a change in model score in percentage points. It is a model-sensitivity exercise; it does not predict the effect of weight loss, alcohol, exercise, or treatment on a person's health.

Subgroup reporting uses the same global threshold for every group. Different recall or false-positive rates identify questions for further study. They do not by themselves prove discrimination or fairness. Survey sex coding is binary and incomplete; it is not a comprehensive gender representation.

Explain back: why can an apparent association be caused by confounding or a behavior change after diagnosis?

## Chunk 10 — Architecture, reproducibility, and presentation

Python trains the estimators and exports their parameters. A JSON file contains means, scales, coefficients, tree structures, calibration parameters, and frozen thresholds. The JavaScript engine implements exactly the corresponding inference steps, including float32 conversion at tree splits to match sklearn. Tests compare 500 artificial profiles against Python outputs. No personal profile is sent to a server.

The site loads aggregate public results, not individual survey records. Downloaded profile reports are generated locally. Inputs remain only in page memory and disappear on reload. CSS, JavaScript, JSON, and documentation are served as static files; no paid prediction API is required.

Reproducibility means documenting the dataset hash, package versions, random seed, split membership, hyperparameters, calibration, and selection rule. It does not guarantee bitwise identity across every operating system or library version.

Presentation sequence (8–10 minutes):

1. State the question and non-diagnostic boundary (45 seconds).
2. Show data and class imbalance (1 minute).
3. Explain the four algorithms and split design (2 minutes).
4. Compare actual results and a confusion matrix (90 seconds).
5. Run a fictional profile and explain sensitivity (1 minute).
6. Show questionnaire reduction and capacity simulation (1 minute).
7. Show subgroup differences and limitations (1 minute).
8. Open the source and explain reproduction (45 seconds).

Every teammate should be able to explain the target, test split, confusion matrix, chosen model, one limitation, and one supported numerical claim. Divide implementation work, not understanding.

## Questions a professor or recruiter may ask

- Why did the selected model not have the highest final test AP? Selection used validation data; final test results did not trigger reselection.
- Why is accuracy below the always-negative baseline? We prioritize finding positive labels and use a validation F1 threshold; the baseline has zero recall.
- Are identical rows duplicates of the same person? We cannot establish identity; we retain them but keep profiles together across splits.
- Why not diagnose someone from the score? The label is self-reported existing status, with no independent clinical validation.
- Does the shorter model save users time? We measured input count, not time. A timed study would be a separate experiment.
- Did calibration always help? No. Inspect before/after Brier values; a sigmoid can be imperfect.
- Does a feature's importance show what causes diabetes? No; this is predictive association, not causal identification.
- What would improve the study? External cohort validation, richer provenance, careful survey-design treatment, additional prespecified encoding/imbalance experiments, and clinical collaboration for any medical intended use.
