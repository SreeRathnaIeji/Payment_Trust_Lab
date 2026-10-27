# Payment Trust Lab: Architecture and Research Boundaries

## Components

```text
Browser dashboard
  ├── Primary Transaction Assessment ── FastAPI ── saved chronological XGBoost + type encoder
  │                                      └── optional signed-in history record in SQLite
  ├── Experimental GNN Assessment ───── FastAPI ── saved node-only TensorFlow GNN + preprocessing
  │                                      └── authenticated, separate history record in SQLite
  ├── Account / Graph Analysis ── FastAPI ── this user's saved transaction edges only
  ├── QR / URL Check ──────────── FastAPI ── zxing-cpp decode + local URL text/hostname indicators
  └── History ─────────────────── FastAPI ── records filtered by authenticated user ID

Offline research:
Chronological PaySim splits ── past-only account degree features ── XGBoost candidate
Chronological PaySim splits ── prior-transaction message passing ── transaction GNN
```

The URL checker and account-history graph do not contribute to transaction scores. No combined model or fusion result is claimed.

## Transaction model

`backend/main.py` loads `xgboost_main_features.joblib` and the saved transaction-type encoder. It builds the same 11 input columns used during training. The threshold remains 0.95. Optional PaySim-style `nameOrig` and `nameDest` fields are metadata for account history; the inference vector is still built only from the six numeric fields and encoded transaction type.

## Experimental node-only GNN assessment

`backend/gnn_inference.py` lazily loads the `node_feature_only` v2 bundle from `Experiments/artifacts/gnn_experiment_v2/node_feature_only/`. The separate authenticated `POST /api/predict/gnn` endpoint reuses the saved scaler, one-hot encoder, model weights, and validation-selected threshold. It needs the same six numeric transaction fields and type accepted by the existing request model. It does not use account IDs, prior account history, future transactions, or the fictional graph. It is explicitly not graph-enabled inference.

The current XGBoost `/api/predict` endpoint, model, threshold, output, and frontend result remain the primary transaction assessment. GNN bundle loading is independent; a failed/missing GNN bundle disables only `/api/predict/gnn`. The node-only GNN test metrics were precision 0.9021, recall 0.5931, F1 0.7157, and average precision 0.7551, below the baseline AP 0.9639 and F1 0.8804. PaySim is synthetic, and the raw score is not calibrated as a real-world fraud probability. TensorFlow 2.21.0 is required to load the optional model; the model is not retrained during inference.

To run it, start the existing API, sign in, open Transaction Assessment, enter transaction details, and choose **Assess with experimental GNN**. API, inference, and frontend regression tests are `python -m unittest tests.test_gnn_inference tests.test_qr_scan -v` and `node --test tests/frontend_behavior.test.cjs`.

The UI's labeled example selector reads `frontend/paysim_labelled_examples.json`, a deterministic eight-row balanced sample from the chronological test split. Ground truth is displayed separately and never included in model request payloads. Each model is run independently, and any correct/mismatch message applies to that one example only; it is not an accuracy statistic.

## Accounts and data isolation

`backend/persistence.py` stores local demo users and assessments in SQLite. Password input is SHA-256 pre-hashed and then bcrypt-hashed; plaintext is not stored. Signed, time-limited session tokens identify a user. Assessment reads and graph-summary queries require authentication and include the authenticated user ID in the database predicate. Transaction and URL assessments include their inputs, timestamp, score where applicable, threshold where applicable, decision, and component version.

This local demo has no multi-factor authentication, account recovery, remote database, deployment-grade key management, or bank integration. Protect the SQLite file and do not reuse real passwords or real financial identifiers.

## Account relationship view

The graph page has two separate scopes. Its fictional graph and table use the six illustrative rows in `frontend/research_demo_transactions.json`; they are not labeled PaySim examples or model context. Its authenticated saved-activity inspector searches only prior transaction assessments submitted by that user, using the exact sender/recipient IDs entered. It can show source and recipient recurrence, repeated directed pairs, and shared neighbors, but is not complete PaySim history or an input to either model. No demo workflow presents these user records as complete PaySim history.

## Fictional research demonstration graph

`frontend/research_demo_transactions.json` is a separate static file with fictional account IDs and unlabeled PaySim-shaped transaction inputs. The dashboard builds a relationship graph and table from those same rows. Selecting an account shows connected accounts, incoming/outgoing counts, and totals; selecting a transaction highlights both participants. Selecting a scenario populates the existing transaction form; the score still comes from the saved transaction model API. This graph neither reads nor writes user histories and does not supply features to either XGBoost model.

## Offline temporal graph-feature experiment

`Experiments/train_temporal_graph_features.py` streams the existing chronological splits. For each hourly step, it first generates each transaction's four log degree features from graph state accumulated through strictly earlier steps, then updates sender out-degree and recipient in-degree counts using the current step's edges. It uses fixed-memory Count-Min Sketches to keep account-cardinality memory bounded; hash collisions can overestimate degree counts. This conservative hourly policy avoids inventing an ordering among transactions that share PaySim's coarse `step` timestamp. Labels are never read while updating graph state.

The candidate XGBoost uses the same 10% training-sample fraction, model family, validation threshold selection, and chronological test split as the existing main-feature experiment. The new model and result file are isolated under `Experiments/artifacts/temporal_graph_xgboost/`. Actual comparative results exist only after a full run completes.

The full run completed. At validation-selected threshold 0.88, the temporal graph-feature candidate scored test precision 0.9776, recall 0.7925, F1 0.8754, and average precision 0.9605. The existing main model at its validation-selected threshold of 0.95 scored 0.9749, 0.8025, 0.8804, and 0.9639, respectively, on the same test split. The candidate therefore did not improve F1 or average precision; results are at `Experiments/artifacts/temporal_graph_xgboost/results.json`.

This experiment is a temporal graph-feature baseline, not message passing or a GNN. At the time it was run, the inspected environment lacked PyTorch, PyTorch Geometric, and DGL; a separate TensorFlow/Keras GNN has since been implemented and evaluated below.

## Offline temporal transaction GNN experiment

`Experiments/train_temporal_gnn.py` implements a separate transaction-node message-passing model using the installed TensorFlow/Keras runtime. Nodes are transactions with the six numeric baseline features and one-hot transaction type. Account IDs are used only to create directed typed edges from prior transactions sharing sender/recipient roles; each current transaction receives at most four such messages with a normalized log step-gap feature. All events at a PaySim step are scored before that step enters graph history, and fraud labels are targets only. Preprocessing is fitted on sampled training rows, validation selects the epoch and threshold, and the later chronological test period is evaluated once.

The experiment completed on CPU with TensorFlow 2.21.0 and is isolated under `Experiments/artifacts/gnn_experiment/`. Test precision was 0.5147, recall 0.3513, F1 0.4175, and average precision 0.3686, versus 0.9749, 0.8025, 0.8804, and 0.9639 for the existing chronological XGBoost baseline on the same test rows. This GNN is not connected to the live app and does not support replacing the baseline. See its `report.md` and `results.json` for full results and constraints.

## URL indicators

`backend/main.py` accepts a size-limited QR image at `POST /api/url/scan-qr` and decodes it with `zxing-cpp`. URL content is sent to `backend/url_analysis.py`, which parses the text without network access and checks a small documented set of suspicious indicators. Plain QR text is returned without URL analysis. For images with multiple codes, only the first decoded code is used. This is a heuristic review aid, not a dataset-trained classifier or QR Shield ML implementation. No warning signs found does not mean a URL is safe.

## Research contributions and limitations

1. An existing chronological transaction XGBoost baseline with saved validation-selected threshold and held-out test metrics.
2. A reproducible XGBoost candidate with past-only temporal account-degree features.
3. A separate transaction-level temporal message-passing GNN experiment, evaluated against the same chronological baseline and test period.
4. A local demonstration of password-hashed accounts and user-scoped assessment history.
5. A separate URL text-indicator workflow that avoids visiting untrusted links.

PaySim is synthetic. Account and history features in the interactive app are user-entered local demo records, not external payment history. Neither temporal graph experiment improved on the current baseline's F1/AP in the reported runs; the GNN result is materially worse. The URL rules have no corpus evaluation. The app's components are not fused.
