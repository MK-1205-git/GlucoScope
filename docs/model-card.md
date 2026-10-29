# Model card — GlucoScope research edition 1.0

**Intended use:** educational investigation of survey-label classification; reproducible model comparison and retrospective capacity simulation.

**Not intended:** diagnosing disease, ruling out diabetes, forecasting an individual's future disease, treatment or testing decisions, determining eligibility for care, or deploying clinical outreach without independent validation.

**Target:** UCI Diabetes_binary, documented as no diabetes versus prediabetes or diabetes. Data-provider metadata discrepancies are documented in the report. Labels are not independently verified medical outcomes.

**Population:** respondents in the processed public CDC/BRFSS-derived dataset. No inference of general population prevalence is made because survey design weights are absent from the export.

**Inputs:** 21 integer-coded questionnaire characteristics; short-form experiment uses eight. Age, education, and income are category codes. Sex is limited to the source's binary encoding. Identifier and target are excluded.

**Training:** 60% training with three-fold grouped CV; 10% calibration; 10% validation; 20% test. Identical full profiles are grouped. Hyperparameter selection uses CV AP. Validation AP chooses the full model. Validation F1 chooses thresholds. Models are frozen before final testing.

**Selected model:** random forest, 80 trees, maximum depth 12, minimum leaf 30, sigmoid calibration. Exact parameters and thresholds are versioned in artifacts and models.json. All comparison models are also available in the UI.

**Performance and uncertainty:** see results.json and research-report.md. Reported uncertainty is conditional on fitted models and the sampled test profiles. It does not certify individual confidence or external validity.

**Risks:** self-report bias, prevalence shift, historical categories, unknown provenance details, post-diagnosis behavior changes, correlated features, false positives, false negatives, incomplete demographic representation, and lack of independent clinical evaluation.

**Privacy:** inference occurs in browser memory. No questionnaire submission endpoint, analytics, database, or local storage is used. External font requests, if available, load typography only; they do not contain questionnaire fields. Static host access logging is separate from application inputs.

**Updates:** version the dataset checksum and full experiment. Do not tune against the current final test results; use fresh untouched evaluation data for confirmatory future iterations.
