# Temporal Transaction GNN Experiment

## Graph and Leakage Policy

Each PaySim transaction is a node. Node features are the six numeric baseline features (step, amount, and four balance fields) plus one-hot transaction type. `isFraud` is the node classification target and is never a node, edge, or message feature. Account IDs are used only to construct edges; they are not embedded as model features.

Directed edges point from a prior transaction to the current transaction when they share an account. The four edge relations distinguish sender-sender, recipient-sender, sender-recipient, and recipient-recipient matches. Edge attributes contain a one-hot relation code and normalized log step gap. The graph keeps one deterministically selected sampled predecessor per role, for at most four incoming edges per query.

Transactions sharing the same PaySim step are featurized before any of that step is added to history. Therefore message sources have strictly earlier steps. Historical context events are sampled at the configured deterministic fraction; current validation/test queries are all evaluated. Labels are used only for training loss, validation threshold/epoch selection, and final metrics.

The model performs relation-aware message projection, masked mean aggregation, and a transaction-level fraud prediction. This is a trainable message-passing GNN, distinct from the XGBoost degree-feature candidate and the fictional UI graph.

## Training

Framework: TensorFlow 2.21.0 on CPU. Sampled training target rows: 601,472; fraud targets: 495. Context sampling fraction: 10%. Best validation-loss epoch: 1; selected decision threshold: 0.912939.

Preprocessing was fit on the deterministic sampled training rows only. No oversampling was used; binary cross-entropy used training-derived class weights. Early stopping monitored validation loss. The chronological test period was not used for model or threshold selection.

## Held-Out Test Results

Confusion matrices use `[[TN, FP], [FN, TP]]`.

| Model | Threshold | Precision | Recall | F1 | Average precision | Confusion matrix |
|---|---:|---:|---:|---:|---:|---|
| Temporal transaction GNN | 0.912939 | 0.5147 | 0.3513 | 0.4175 | 0.3686 | `[[101443, 530], [1038, 562]]` |
| Existing chronological XGBoost baseline | 0.950000 | 0.9749 | 0.8025 | 0.8804 | 0.9639 | `[[101940, 33], [316, 1284]]` |

Both models report the same chronological PaySim test period. The baseline numbers are read from its existing saved result file and were not recomputed or modified.

## Limitations

PaySim is synthetic, so labels represent simulator-generated fraud rather than independently verified payment fraud. The temporal graph uses a deterministic 10% sample for context to bound memory and retains one representative predecessor for each account-role relation; this sparsifies histories and can omit useful interactions. Its absolute scores are not evidence of UPI or real-world performance. The GNN uses the same tabular transaction fields as the baseline, so differences reflect the message-passing architecture and sampled graph context rather than new external information.

This experiment writes only to its new artifact directory; the live application and existing model artifacts are unchanged.
