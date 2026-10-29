# Course requirements and where to demonstrate them

| Requirement | Evidence | Live demonstration |
|---|---|---|
| Real-world question | Report A; README | Overview explanation |
| Source, observations, features | audit.json; report B | Dataset & methods |
| Feature descriptions and data types | schema.py; notebook | Data dictionary |
| Missing values, duplicates, class distribution | Audit and grouped split assertions | Dataset summary and frequencies |
| Preprocessing explained | Report C; train.py | What we actually changed |
| At least three algorithms | Four full models plus short model and dummy | Model comparison table |
| Train/test separation | split_assignments.csv; frozen_selection.json | Four-role split |
| Accuracy, precision, recall, F1 | metrics.csv; results.json | Model evidence |
| Confusion matrix | results.json; notebook | Model-specific matrix |
| Imbalance handled thoughtfully | AP objective; validation thresholds; dummy comparison | Always-negative baseline |
| Model comparison | Validation and test results | Table, PR curves |
| Best model and why | Validation-selected random forest | Selection explanation |
| Influential features | Permutation importance | Input-importance panel |
| Unexpected findings | Boosting slightly wins test AP; dummy high accuracy; calibration mixed | Report H |
| Preprocessing effects | Full-versus-short experiment; remaining effects not isolated | Short-form comparison |
| Trade-offs and limitations | Report H–J; model card | Errors, subgroups, limitations |
| Final conclusion and future work | Report J | Presentation closing |
| Extra: deployed product | Full browser inference and evidence UI | Public live link |
| Extra: reproducibility | Training scripts, notebook, checksums, tests | Source download |

## Demo acceptance checklist

- Open every navigation page.
- Run all three fictional profiles and switch algorithms.
- Select short-form logistic and show eight inputs.
- Explain the score, classification threshold, and a sensitivity result.
- Compare AP, recall, precision, and confusion matrices.
- Explain why the dummy baseline's accuracy is misleading.
- Move capacity from 20% to 100% and show why ranking gains disappear.
- Inspect age and BMI distributions.
- Show subgroup performance without calling it a fairness certificate.
- Download CSV, report, teaching guide, and source.
- Explain that the models run locally in the browser and personal inputs are not uploaded.

## Current verification boundary

Python/JavaScript numeric parity, DOM interactions, and artifact consistency are checked automatically. Visual browser QA was unavailable in the execution environment. Review the live layout on your own laptop and phone before presenting.
