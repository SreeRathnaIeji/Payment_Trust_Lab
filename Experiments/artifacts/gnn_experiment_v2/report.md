# Temporal GNN v2: Controlled Ablations

## Protocol

Both variants use the same chronological PaySim train/validation/test periods, sampled training targets, train-only preprocessing, class weights, optimizer, hidden dimension, batch size, early-stopping policy, and validation-only F1 threshold selection. The baseline is the existing saved chronological XGBoost result on the same test split; it was not retrained.

The graph variant uses transaction nodes and directed edges from strictly earlier sampled transactions sharing one of four sender/recipient endpoint roles. It keeps up to the configured number of recent neighbors per relation with typed edge and log time-gap attributes. Same-step and future edges are excluded; labels are targets only.

## Test Results

Confusion matrices are `[[TN, FP], [FN, TP]]`.

| Variant | Selected epoch | Validation threshold | Precision | Recall | F1 | Average precision | Confusion matrix | Training seconds | Available graph coverage* | Mean available edges/transaction* |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| node_feature_only | 18 | 0.902306 | 0.9021 | 0.5931 | 0.7157 | 0.7551 | `[[101870, 103], [651, 949]]` | 35.1 | 0.4046 | 0.9699 |
| graph_enabled | 12 | 0.981630 | 0.8956 | 0.7025 | 0.7874 | 0.7775 | `[[101842, 131], [476, 1124]]` | 93.2 | 0.4046 | 0.9699 |
| Existing chronological XGBoost | validation-selected | 0.950000 | 0.9749 | 0.8025 | 0.8804 | 0.9639 | `[[101940, 33], [316, 1284]]` | not retrained | n/a | n/a |

*Coverage describes sampled graph context available for the shared test rows; the node-feature-only model does not consume graph messages.*

## Training and Resource Details

Validation AP controlled checkpoint selection with patience 3; thresholds maximize validation F1, then precision, then higher threshold. Epoch histories and selected epochs are saved in each variant's `results.json`.

Preprocessing and graph staging time: 247.3 s (graph staging alone: 107.3 s). Memory-mapped temporary staging size: 1.00 GiB. Peak process working set: not captured for the completed training process.

## Limitations

PaySim labels are simulator-generated, not independently verified real-world fraud. The XGBoost comparison uses the same held-out period but a separately trained model and its own selected threshold, so it is a practical baseline comparison rather than a controlled architecture-only comparison. Context sampling and a finite recent-neighbor window still omit graph history. The node-only and graph-enabled variants are controlled against one another, but single-seed results do not establish robustness. Do not integrate a GNN unless results support it and repeated-seed checks confirm stability.
