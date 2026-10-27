# Temporal Graph-Feature XGBoost Candidate

## Reproduction

From the project root:

```powershell
python Experiments/train_temporal_graph_features.py --self-test
python Experiments/train_temporal_graph_features.py
```

The first command checks the past-only feature-state rule with a small synthetic example. The second reads `Dataset (PaySim)/chronological_splits/{train,validation,test}.csv`, uses the saved transaction-type encoder and existing deterministic 10% training-row sampling pattern, selects a classification threshold on validation, and evaluates on the existing chronological test split. It writes new files under `Experiments/artifacts/temporal_graph_xgboost/` and does not overwrite existing experiment artifacts.

## Method and leakage controls

This is an XGBoost model with four approximate temporal account-degree features, not a GNN. A fixed-memory Count-Min Sketch estimates prior sender out-degree, sender in-degree, recipient in-degree, and recipient out-degree. For each PaySim hourly step, all rows receive features before any transaction in that step updates sketch state. `isFraud` is used only as the prediction target and is not used to update graph state or create features. Sketch collisions can overestimate degree.

## Observed results

| Model | Validation-selected threshold | Test precision | Test recall | Test F1 | Test average precision |
|---|---:|---:|---:|---:|---:|
| Existing chronological XGBoost | 0.95 | 0.9749 | 0.8025 | 0.8804 | 0.9639 |
| Temporal graph-feature XGBoost candidate | 0.88 | 0.9776 | 0.7925 | 0.8754 | 0.9605 |

Both rows use the same chronological test split. Each threshold was selected using its validation predictions. The graph-feature candidate did not improve test F1 or average precision. Precision increased slightly while recall declined. Do not describe this experiment as demonstrating graph improvement.

Full metrics, confusion counts, baseline comparison, and validation threshold sweep are recorded in `Experiments/artifacts/temporal_graph_xgboost/results.json` and `validation_thresholds.csv`.

# Temporal Transaction GNN

## Reproduction

From the project root, using the active TensorFlow environment:

```powershell
python Experiments/train_temporal_gnn.py
```

The experiment reads the existing chronological PaySim splits and writes new files only under `Experiments/artifacts/gnn_experiment/`. It refuses to overwrite an existing output directory. Focused tests run with:

```powershell
python -m unittest tests.test_temporal_gnn -v
```

## Graph and leakage policy

Transactions are graph nodes with the six numeric baseline fields and one-hot transaction type. Directed edges connect earlier transactions to a query transaction when sender/recipient account roles match; each query has up to four typed predecessors and a normalized log time-gap edge feature. Edges from the same PaySim step are excluded. Historical context uses a deterministic 10% sample to bound memory, while all validation and test transactions are scored. Fraud labels are used only as targets, never graph features or edge attributes. Preprocessing is fit on the sampled training rows only.

The Keras model performs edge-conditioned message projection, masked mean aggregation, and transaction-level binary classification. It is a real trainable GNN, distinct from the Count-Min-Sketch XGBoost candidate above and the fictional UI graph.

## Observed results

The 601,472 training targets include 495 fraud cases. Validation selected epoch 1 by loss and threshold 0.912939 by F1. On the chronological test period (103,573 transactions, 1,600 fraud cases), the GNN achieved precision **0.5147**, recall **0.3513**, F1 **0.4175**, and average precision **0.3686**. Its confusion matrix is `[[101443, 530], [1038, 562]]` in `[[TN, FP], [FN, TP]]` order.

The existing XGBoost baseline on the same test period achieved precision **0.9749**, recall **0.8025**, F1 **0.8804**, and average precision **0.9639**, with confusion matrix `[[101940, 33], [316, 1284]]`. The GNN is substantially worse and should not replace the live model. PaySim labels are simulator-generated, so neither result establishes real-world or UPI performance.

TensorFlow 2.21.0 was already installed in the selected Python 3.13 environment; no dependency was installed. Weights, preprocessing, configuration, metrics, and the detailed report are under `Experiments/artifacts/gnn_experiment/`.
