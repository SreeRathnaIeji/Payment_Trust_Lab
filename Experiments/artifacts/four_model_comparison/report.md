# Four-Model PaySim Comparison

## Method

This experiment uses the existing chronological PaySim splits. Training uses a deterministic 10% row sample from steps 1-450; validation is steps 451-600; test is steps 601-743. The test split is evaluated only after model and threshold selection.

A StandardScaler and OneHotEncoder are fit only on the sampled training rows. No oversampling is used. Random Forest and Logistic Regression use class weighting; XGBoost uses the sampled negative-to-positive ratio; Isolation Forest is fit on non-fraud training rows only.

Individual decision thresholds maximize validation F1, then precision, with higher threshold as a final deterministic tie-break. Average precision uses continuous probability scores for supervised models and anomaly scores for Isolation Forest.

The ensemble uses the four thresholded binary predictions: at least 3 fraud votes means fraud. A 2-2 tie is resolved as legitimate. This is strict majority voting, not score averaging. Ensemble average precision is computed from the discrete fraud-vote count (0-4).

## Test Results

Confusion matrices use `[[TN, FP], [FN, TP]]`.

| Model | Threshold | Precision | Recall | F1 | Average precision | Confusion matrix |
|---|---:|---:|---:|---:|---:|---|
| Random Forest | 0.316219 | 0.9820 | 0.7844 | 0.8721 | 0.9270 | `[[101950, 23], [345, 1255]]` |
| XGBoost | 0.932888 | 0.9708 | 0.8106 | 0.8835 | 0.9621 | `[[101934, 39], [303, 1297]]` |
| Logistic Regression | 0.931997 | 0.9171 | 0.6637 | 0.7701 | 0.8596 | `[[101877, 96], [538, 1062]]` |
| Isolation Forest | 0.626264 | 0.1601 | 0.2044 | 0.1795 | 0.1127 | `[[100257, 1716], [1273, 327]]` |
| Strict Majority Ensemble | strict majority (3/4) | 0.9954 | 0.6700 | 0.8009 | 0.8000 | `[[101968, 5], [528, 1072]]` |

Test rows: 103,573; fraud rows: 1,600. Ensemble 2-2 ties: 206.

## Limitations

PaySim is synthetic, and its fraud labels reflect the simulator rather than independently verified real-world outcomes. These results do not establish performance on UPI or other live payment data. The deterministic 10% training sample reduces compute and may add sampling variability. Thresholds are validation-F1 choices, not calibrated risk cutoffs. Isolation Forest produces anomaly rankings rather than fraud probabilities, and the ensemble AP is coarse because it ranks transactions by only five possible vote counts. Model comparisons remain sensitive to each estimator's inductive bias and the shared feature representation.

This experiment writes only to its new artifact directory; it does not change the live application's model or frontend.
