# Payment Trust Lab — review and verification notes

## Applied changes

- Improved typography, contrast, disabled button states, drawer labels, form controls, result panels, and responsive spacing with a final CSS override. The brand mark remains hidden as requested.
- Kept the home page focused on a short introduction and workflow cards; navigation remains in the hamburger drawer.
- Kept QR & Link Safety navigation pointed at the URL/QR section.
- Changed the network graph from fictional account labels/transactions to the included selected PaySim chronological test-split sample. It contains four account nodes and three selected transactions.
- Added a separate sample relationship inspector. It reports incoming/outgoing counts, directed pair count, reverse count, and shared connected accounts based only on the selected sample. Unknown IDs are rejected with a clear scope explanation.
- Kept a separate authenticated private-history inspector and clarified that it only searches that user's saved assessments.
- Updated the optional GNN error state to show the backend's actual reason. Guest users must sign in; a missing TensorFlow dependency or failed artifact load can still make the model unavailable.
- Added a broader offline model-comparison table and clarified that offline graph-enabled GNN results are not the live node-only endpoint.
- Updated README commands for Windows PowerShell and explained why npm commands are not used.
- Updated frontend behavior tests to match the real PaySim sample source and current result labels.

## Checks run in this environment

- `node --check` on the inline JavaScript: passed.
- `node --test tests/frontend_behavior.test.cjs`: 7 tests passed.
- Full Python test collection was attempted but could not complete in this environment because `bcrypt` and `tensorflow` were not installed. Install `requirements.txt` and run the Python tests locally.
- A full browser-based visual pass and live TensorFlow GNN inference were not performed here. Confirm these on the target machine before submission.

## Remaining known limitations

- The PaySim network view is a tiny selected sample, not a full-dataset graph query. Sample counts must always be interpreted in that scope.
- The authenticated relationship summary is still based on the user's saved assessments only.
- The live GNN is node-feature-only; graph-enabled GNN and graph-feature XGBoost are offline experiments.
- QR/URL checking uses heuristics and does not fetch websites or verify UPI recipients.
- SQLite auth/history is a demo feature, not production security.
