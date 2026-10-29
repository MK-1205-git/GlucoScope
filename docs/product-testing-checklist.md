# GlucoScope: step-by-step product test

Live URL: https://glucoscope-mushtahir-ml.mushtahirkabir.chatgpt.site

Allow 25–35 minutes for a thorough first pass. Start with fictional examples so you can compare exact expected results. Mark each test PASS / FAIL / NOT TESTED and record the device and browser. The expected numbers below refer to the current frozen model artifacts, not a medical assessment.

## Before you start

1. Open the live URL in Chrome or Edge. If you used the old site, press Ctrl+Shift+R (Windows) or Cmd+Shift+R (Mac).
2. You should see a navy sidebar, teal buttons, white panels, and the heading **Diabetes indicator research**. The sidebar lists six pages. If you still see the green marketing-style banner, the old version may be cached.
3. Open the same URL in an incognito/private window. It should load without a ChatGPT login or registration request. Report any sign-in wall: this deployment is intended to be public.
4. Allow the first model download to finish. A loading message may briefly appear; it should be replaced by the dashboard, not persist indefinitely.

## Login and registration: expected behavior

There is no account system in this edition. There is no login, sign-up, password reset, or saved profile history to test. This is intentional: questionnaire predictions run in the browser, and answers are not stored by the application. Users enter answers in **Profile explorer**, not in registration. A downloaded JSON result is saved on the user's own device.

## 1. Navigation and overview

| Test | Action | Expected result |
|---|---|---|
| N1 | Open Overview. | 253,680 responses, 50,736 test records, selected-model ROC-AUC 0.827, and 62% fewer short-form inputs appear. |
| N2 | Click **Enter health information**. | Profile explorer opens with the questionnaire and model output area. |
| N3 | Click each of the six sidebar items. | The page changes, current navigation is highlighted, and the breadcrumb matches the page. No blank screens or error messages. |
| N4 | Use browser Back and Forward after visiting multiple pages. | The appropriate page returns. |
| N5 | Click the GlucoScope name/logo. | Overview opens. |
| N6 | Click **Download report** on Overview. | A Markdown report downloads or opens as text, depending on browser behavior. It is not an empty file. |
| N7 | On Overview inspect the short-form comparison. | 8 inputs instead of 21; 98.4% AP retained; full/short AP approximately 0.408 / 0.402. |

## 2. Where users enter information

1. Open **Profile explorer** or go directly to https://glucoscope-mushtahir-ml.mushtahirkabir.chatgpt.site/#explorer.
2. Set the Model dropdown to **Random forest**.
3. Click **Start a new profile**.
4. All 21 answer fields should clear. Dropdowns should show **Select an answer**. The output should say **Complete the questionnaire**, with no percentage shown.
5. Fields are grouped under **About the profile**, **Health history**, **Daily habits**, and **Health and access to care**. No name or email is requested.
6. Fill every field and click **Run the model**. A percentage, positive/negative model label, model name, threshold, and sensitivity panel should appear.
7. For a quicker manual form test, choose **Short-form logistic**, click **Start a new profile**, fill its eight fields, then run it. Switch back to Random forest; missing full-form answers must still be filled before a new full-model result appears.

## 3. Exact prediction checks

For these checks, choose **Random forest**, then click the example buttons. Loading an example calculates its result immediately; pressing Run again should leave the same result.

| Test | Action | Expected score (rounded) | Expected model label |
|---|---|---:|---|
| P1 | Click **Example A · active adult**. | 3.5% | Negative label |
| P2 | Click **Example B · mixed indicators**. | 11.7% | Negative label |
| P3 | Click **Example C · several indicators**. | 66.9% | Positive label |
| P4 | On Example C, change BMI from 35 to 20. Do not run yet. | Old score disappears; **Answers changed** is shown. | No stale classification should remain. |
| P5 | Now click **Run the model**. | 26.7% | Positive label |
| P6 | Run again without editing. | Still 26.7%. | Same result. |
| P7 | Reload Example C. | Back to 66.9%, BMI back to 35. | Positive label. |

The random-forest threshold is approximately 17.2%, selected to maximize validation F1. That is why 26.7% is classified positive. The label is not a diagnosis, and the score is not clinically validated personal risk.

## 4. Model switching and input validation

| Test | Action | Expected result |
|---|---|---|
| V1 | Load Example B. Select Logistic regression, Decision tree, Random forest, and Gradient boosting in turn. | Each shows its own model name and calculated output; all four use 21 fields. Identical rounded outputs can occasionally occur and are not inherently a bug. |
| V2 | Select **Short-form logistic**. | Exactly eight fields: age group, recorded sex, BMI, high blood pressure, high cholesterol, difficulty walking, general health, poor physical health days. |
| V3 | Return to Random forest. Enter BMI 999 and click Run. | Browser validation or a clear field error blocks a new prediction. No valid new percentage for BMI 999. |
| V4 | Try BMI 11, then BMI 24.5. | Both blocked: accepted BMI is an integer from 12 to 98. |
| V5 | Clear BMI and click Run. | Required-field validation blocks the run. |
| V6 | Try poor mental health days 31 and physical health days −1. | Both blocked; days must be integers 0–30. |
| V7 | Click Start a new profile and immediately click Run. | Incomplete fields block the run; blank dropdowns are not silently treated as No. |
| V8 | Load Example A after an invalid-input attempt. | The example restores valid values and a working prediction; the previous error clears. |
| V9 | Change an answer, navigate to another page, then return to Profile explorer. | The edited answer remains during this tab session; no outdated result is presented for it. Run the model again if prompted. |

## 5. Explanation and local export

1. Load Example C with Random forest.
2. Look under **What changes this output?** It should list up to five one-feature changes, reference values, and signed changes in percentage points.
3. Verify the page explains that this is model sensitivity, not proof that changing a behavior causes a health outcome.
4. Click **Download this result**. A file named `glucoscope-profile-result.json` should download.
5. Open it in a text editor. Confirm it contains the model name, input values, probability, threshold, classification, and educational notice.
6. Check Example C has BMI 35 and probability approximately 0.669 (the UI displays a rounded percentage). After you edit BMI and rerun, a new download must match the new inputs and probability.

## 6. Model evidence

1. Open **Model evidence**: https://glucoscope-mushtahir-ml.mushtahirkabir.chatgpt.site/#compare.
2. The table must show the four full models, Short-form logistic, and the always-negative baseline. Random forest should be marked **Selected on validation**.
3. For Random forest, verify rounded metrics: accuracy 0.802; precision 0.374; recall 0.622; F1 0.467; AP 0.431; ROC-AUC 0.827; Brier 0.098.
4. In **Inspect model**, select Random forest. The confusion matrix should show true negatives 36,314, false positives 7,353, false negatives 2,672, and true positives 4,397. They sum to 50,736.
5. Switch the dropdown through all five models. The confusion matrix, PR curve, calibration curve, importance panel, and subgroup rows should reflect the selected model. The comparison table intentionally stays unchanged: it always compares all models.
6. Check the always-negative baseline: accuracy is approximately 0.861, but recall and F1 are zero. This is expected, not evidence that the baseline is better overall.
7. Click **Download CSV**. Open `metrics.csv` in Excel or a text editor. Its numbers should agree with the main model rows (the dummy is displayed separately, not included in this CSV).
8. Scroll to subgroup results. The table should have two recorded-sex groups and three age groups, with sample counts and metrics. Bootstrap intervals beneath it apply to the selected model overall, not each subgroup.

## 7. Capacity simulator

1. Open **Capacity simulator**: https://glucoscope-mushtahir-ml.mushtahirkabir.chatgpt.site/#capacity.
2. Set the slider to 20%. All strategies should select **10,148** records.
3. Verify captured positive labels: Logistic regression **3,907**; Decision tree **3,919**; Random forest **4,001**; Gradient boosting **4,021**; Short-form logistic **3,843**.
4. The text should state approximately **2.4% more** captured labels for random forest versus logistic regression at this capacity. The comparison is retrospective, not improved patient outcomes.
5. Move to 5%, 50%, then 100%. Counts and metrics should update without page reload. Selected count should increase with capacity.
6. At 100%, all models should select **50,736** records, capture **7,069** positives, have **100% recall**, and about **13.9% precision**. The advantage over the baseline should disappear.
7. Use keyboard arrow keys on the slider. Capacity should change in 5-percentage-point steps.

## 8. Dataset and methods

1. Open **Dataset & methods**: https://glucoscope-mushtahir-ml.mushtahirkabir.chatgpt.site/#data.
2. Verify 253,680 records, about 13.9% positive labels, zero missing values, and 25,772 repeated input rows.
3. Verify split counts: training 152,208; calibration 25,368; validation 25,368; test 50,736.
4. In **Explore a feature**, choose Age group, Body mass index, Recorded sex, and High blood pressure. Each change should update values, record counts, frequency bars, and positive-label rates.
5. Age should show 13 labeled bands. Recorded sex and blood pressure should each have two categories. BMI should have many rows.
6. Scroll through the 21-row dictionary. Verify that age is explained as a category, mental/physical health as days, and income as historical bands.
7. Open the UCI source link from the sidebar. It should lead to dataset 891 in a new tab.

## 9. Learning and downloads

1. Open **Learn & reproduce**: https://glucoscope-mushtahir-ml.mushtahirkabir.chatgpt.site/#learn.
2. Open and close each of the ten teaching sections. Each should contain an explanation and a check-your-understanding question.
3. Download all four files: this testing checklist, complete source ZIP, research report, and teaching guide.
4. Extract the source ZIP. Confirm it includes README.md, src/, dist/, tests/, docs/, artifacts/, and the executed notebook under notebooks/.
5. Do not expect raw survey rows or Python joblib files in the ZIP; reproduction downloads the CSV and retrains models.

## 10. Browser, mobile, accessibility, and privacy checks

- Repeat a profile and a simulator test in a second browser.
- Open the site on your phone. Navigation becomes a horizontal row; swipe it if necessary. Forms and buttons should be usable without overlapping. Wide evidence tables may scroll sideways within their panels.
- At browser zoom 200%, text and controls should remain usable. No controls should disappear behind panels.
- Navigate with Tab and Shift+Tab. Focus should be visible. Enter activates links/buttons; Space toggles learning disclosures; form labels should be associated with their inputs.
- Refresh after editing a profile. The standard fictional profile should return; your custom answers should not be restored from persistent storage.
- Optional technical check: open Developer Tools → Network, let initial model files finish downloading, then clear the log. Change questionnaire inputs and run a model. There should be no prediction request containing your answers. Do not confuse static asset loading with data submission.
- Optional: after initial load, set the browser network to Offline without reloading. Existing profile inference and navigation should still work from loaded artifacts. Downloads or a full reload may fail offline; no offline-installable app is promised.
- Check that no output promises diagnosis, disease prevention, treatment, or future disease prediction.

## How to report a bug

Copy this template and attach a screenshot if useful:

```
Page/link:
Device and browser:
Model and example:
Exact steps:
Expected result:
Actual result:
Does it still happen after a hard refresh?
Screenshot / error text:
```

Automatic verification covers numerical model parity, DOM interactions, validation, and evidence consistency. This checklist is the manual acceptance pass, including visual layout and actual browser downloads, which were not screenshot-tested in the build environment.
