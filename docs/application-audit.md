# Payment Trust Lab application audit

## Current working functionality

- **Overview:** Starts the transaction, fictional relationship, URL/QR, and research workflows from the existing hamburger dashboard.
- **Transaction assessment:** Uses the saved chronological XGBoost artifact and saved type encoder through `POST /api/predict`. The API returns the raw class score, threshold, and threshold decision. Six unlabeled fictional scenarios in `frontend/research_demo_transactions.json` populate the form and require a real prediction submission.
- **Account relationships:** The fictional graph and six-row table use the same JSON source. Account, table, transaction, clear-selection, and empty/error states are implemented. The graph is a demonstration visualization, not a GNN or fraud predictor. A separate authenticated graph endpoint summarizes only that user's saved transaction assessments.
- **QR and URL:** `POST /api/url/scan-qr` uses `zxing-cpp` to decode an uploaded QR. URL contents go through local rules in `backend/url_analysis.py`; text contents are displayed without URL analysis. Neither path opens submitted URLs. Direct URL checks remain available.
- **Accounts and history:** SQLite-backed registration, bcrypt password hashing, signed sessions with server-side revocation, per-user histories, and stale-request protection are implemented.
- **Research:** The saved chronological baseline and temporal graph-feature candidate are compared using their existing result files. The candidate did not improve F1 or average precision.

## Problems identified and actions taken

1. The README, demo guide, and architecture page had stale QR availability statements. They now describe the installed decoder, upload endpoint, first-code behavior, and heuristic scope.
2. The URL result panel exposed internal risk-level values. It now uses plain-language headings and explicitly says that no listed warning signs do not establish safety.
3. The Overview repeated the synthetic-data disclaimer, and required versus optional transaction inputs were unclear. The Overview now has three direct entry actions with concise scope text; the transaction form labels six model fields as required and account IDs as optional.
4. History headings and results were technical or ambiguous. The table now uses plain-language assessment and decision labels while retaining account isolation and model score provenance.
5. QR scanning uses the first decoded QR when an image contains multiple codes. The interface and docs now state this limitation.

## Research and technical limits to communicate

- PaySim is synthetic mobile-money data, not real UPI or bank transaction history.
- The transaction score is the saved model's `predict_proba` output. The model uses `binary:logistic`, but no calibration procedure or calibration evaluation was found. It must be presented as a model score, not a calibrated real-world fraud probability.
- The transaction threshold is 0.95, selected on validation. It is unchanged.
- The graph-feature candidate is XGBoost with past-only graph-derived features, not a GNN; saved results show lower test F1 and average precision than the main baseline.
- The account graph is based on the current user's locally saved assessments. The fictional graph is separate from it and from the offline PaySim graph experiment.
- URL checks are heuristic indicators with no trained URL classifier or URL-corpus evaluation. QR decoding plus those rules is not the QR Shield ML method. No URL result guarantees safety.
- Browser layout/responsiveness has not been assessed as part of this file audit.

## Verification and remaining work

- `node --test tests/frontend_behavior.test.cjs`: PASS, 5 tests / 5 passed. Covers history loading/records/empty/error and account switching, URL/QR presentation, raw model-score formatting, and graph selection/related rows/clear/error states.
- `node --check tests/frontend_behavior.test.cjs`: PASS.
- Python AST syntax check for `backend/main.py`, `backend/persistence.py`, `backend/url_analysis.py`, `backend/verify_backend.py`, and `tests/test_qr_scan.py`: PASS. The installed Python launcher prints a “Failed to find real location” warning but returned success.
- `python Experiments/train_temporal_graph_features.py --self-test`: PASS, “no same-step or label leakage.” The launcher prints the same location warning but returned success.
- `python backend/verify_backend.py`: PASS. It used and cleaned a unique temporary SQLite database. Health, saved model/encoder load, prediction, validation, password hashing, URL checks, graph, history isolation, restart persistence, and server-side logout revocation all passed. The run reported model artifact `Experiments/artifacts/chronological/xgboost/xgboost_main_features.joblib`, encoder categories `CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER`, input width 11, raw score `4.82163102333e-07`, and threshold `0.95`.
- `python -m unittest discover -s tests -p test_qr_scan.py -v`: PASS, 3 tests / 3 passed (URL QR, plain-text QR, invalid image). A non-fatal SQLite `ResourceWarning` about an unclosed connection was emitted by module import; all temporary test resources were cleaned.
- No browser visual/responsive check was performed. The app still requires an actual browser review for visual correctness at target screen sizes.

The implementation work in this audit is complete. Further work should be driven by browser review or a concrete defect; no model, threshold, dataset, split, experiment result, database, or backend behavior was changed for these UX/documentation fixes.
