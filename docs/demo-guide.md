# Payment Trust Lab reproducible demonstration

## Fictional transaction and relationship data

`frontend/research_demo_transactions.json` is an explicitly fictional dataset of five demo account identifiers and six PaySim-shaped transaction inputs. It contains no fraud labels, claims of verified ground truth, or real account information. The sample selector loads one row into the existing transaction form. The displayed score is obtained only after the user submits that form to the unchanged `POST /api/predict` endpoint.

The Account Connections page renders the graph and transaction table from those same six rows. Selecting a node or sender/recipient name shows the account's unique connections, incoming and outgoing transaction counts, and total amounts. Selecting a transaction or connection line highlights its participants and shows its details. Repeated connections represent multiple fictional transactions, not proof of fraud. The view is a demonstration, not a trained GNN or fraud predictor. This demo graph is independent of:

1. The authenticated account-history endpoint, which returns only the current user's saved assessment edges.
2. The offline temporal graph-feature experiment, which derives past-only PaySim account-degree features for an XGBoost candidate.

Neither graph is a GNN, and the demo graph does not contribute to transaction predictions.

## Reproduce actual model responses

From the project root, run:

```powershell
python demo/run_model_scenarios.py
```

The script checks its configured model path, creates a uniquely named temporary SQLite database, initializes the actual FastAPI app, checks `/api/health`, and sends each fictional row through `POST /api/predict`. Requests are anonymous, so they do not create demo users or history records. The temporary database is removed when the run completes. The script also checks the seven documented URL rules locally without visiting any submitted URL. It writes the complete returned prediction objects to `demo/scenario_outputs.json`; this is a generated demo output, not an experiment artifact.

The following outputs were returned by the saved chronological model during the recorded run. The model's text decision is shown verbatim. These are predictions for fictional feature inputs, **not fraud labels**, and they do not verify whether any transaction is legitimate. The model score has not been established as a calibrated probability.

| Scenario | Returned score | Returned decision | Threshold |
|---|---:|---|---:|
| Aster → Birch · transfer 1 | 7.481133707187837e-06 | Likely legitimate | 0.95 |
| Aster → Birch · transfer 2 | 0.0001156571161118336 | Likely legitimate | 0.95 |
| Aster → Cedar · cash out | 9.145236617769115e-06 | Likely legitimate | 0.95 |
| Cedar → Dune · payment | 3.856411422020756e-05 | Likely legitimate | 0.95 |
| Dune → Birch · transfer | 0.00022319110576063395 | Likely legitimate | 0.95 |
| Elm → Aster · cash in | 9.210170901496895e-06 | Likely legitimate | 0.95 |

These returned outputs are not performance statistics. The offline test metrics are in the saved experiment result files and summarized separately in Research Results.

## Offline result comparison

The comparison displayed in the dashboard is sourced from:

- `Experiments/artifacts/chronological/xgboost/main_features_results.json`
- `Experiments/artifacts/temporal_graph_xgboost/results.json`

Both candidates were evaluated on `Dataset (PaySim)/chronological_splits/test.csv`. Each selected its threshold on its own validation predictions: 0.95 for main XGBoost and 0.88 for the graph-feature candidate.

| Metric | Main XGBoost | Temporal graph-feature XGBoost | Difference |
|---|---:|---:|---:|
| Test precision | 97.4943% | 97.7641% | +0.2698 percentage points |
| Test recall | 80.2500% | 79.2500% | −1.0000 percentage point |
| Test F1 | 88.0357% | 87.5388% | −0.4968 percentage points |
| Test average precision | 96.3920% | 96.0520% | −0.3399 percentage points |

The candidate's precision is slightly higher at its selected operating threshold, while recall, F1, and average precision are lower. It did not improve the baseline. Average precision summarizes ranking across thresholds; the other three displayed classification metrics use each model's validation-selected threshold. These are offline metrics on synthetic PaySim data, not live demo performance or real banking results. The graph candidate is XGBoost with past-only temporal graph features, not a GNN.

## URL indicator rule cases

The app checks URL text locally, without DNS resolution, fetching, or following redirects. `demo/run_model_scenarios.py` confirms these examples trigger the named rule codes:

| Example text | Expected indicator code |
|---|---|
| `http://example.test/path` | `non_https` |
| `https://demo:demo-pass@example.test/` | `userinfo` |
| `https://192.0.2.10/` | `ip_host` |
| `https://xn--bcher-kva.example/` | `punycode` |
| `https://bit.ly/demo` | `shortener` |
| `https://example.test:8443/` | `unusual_port` |
| `https://example.test/login?verify=payment` | `sensitive_terms` |

These strings are examples only; the script never opens them. The checker is a small heuristic ruleset, not a trained classifier. No listed indicator does not establish that a URL is safe.

## QR decoding and limits

The URL Risk Check page accepts direct URL text or an image upload. The API limits uploads to 5 MB and 20 megapixels, validates the image with Pillow, and uses `zxing-cpp` (Python module `zxingcpp`) to decode QR codes. If multiple QR codes are present, the first decoded code is used. A URL payload is sent to the existing local URL rules; plain text is displayed without URL analysis. Invalid images, missing QR codes, and undecodable content return readable errors.

The service never resolves, opens, or fetches decoded URLs. QR decoding plus heuristic URL checks is not QR Shield's separate machine-learning approach: this project has no labelled URL corpus, trained QR/URL classifier, or evaluation for such a model. A clean indicator result does not guarantee safety.

## Safety and scope

- The demo inputs are separate from the original PaySim CSV, chronological splits, and model artifacts.
- The integration script uses an isolated temporary database and anonymous prediction requests; it never touches the normal demo database or an existing user's history.
- The transaction model, 0.95 live threshold, saved accounts, logout revocation, private history, and endpoint contracts are unchanged.
- All three graph-related surfaces have distinct scopes: static fictional demo edges, authenticated per-user saved history, and offline past-only PaySim graph features.
- No component fusion, real UPI connection, real account relationship, calibrated probability claim, or GNN is included. QR decoding is implemented, but the URL rules remain untrained heuristics without corpus evaluation.
