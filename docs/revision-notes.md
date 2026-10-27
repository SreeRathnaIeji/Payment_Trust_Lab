# Graph Project — Revision Notes for Step 1

## What we are building
The project already has a tabular baseline. XGBoost is the strongest individual baseline in the completed validation experiments. The next research question is whether account relationships and transaction timing provide useful information beyond the transaction's own features.

## Why keep the existing FYP folder?
`Dataset (PaySim)` contains the original dataset and prepared splits. `Experiments` contains baseline scripts and model artifacts. Keeping them avoids duplicating data and preserves completed work.

The new `graph_model` folder is for graph-learning implementation. `backend` and `frontend` will hold the API/application and dashboard when we reach those stages.

## What is a graph in this project?
- **Node:** an account identifier.
- **Directed edge:** a transaction from a sender account to a receiver account.
- **Edge features:** transaction details such as amount, type and time step.
- **Edge label:** `isFraud`, the transaction label used for supervised training.
- **Temporal information:** `step` helps preserve when transactions happened and their order.

Example: if `nameOrig = A`, `nameDest = B`, `step = 10`, and `amount = 5000`, this transaction becomes a directed edge `A → B` with transaction details attached.

## Why does PaySim support this?
PaySim includes `nameOrig` (sender ID) and `nameDest` (receiver ID), allowing sender–receiver edges to be formed. `step` provides transaction-time information. These are synthetic account identifiers and synthetic transactions, not real customer banking data.

## What the first script does
`graph_model/data/inspect_graph_structure.py`:
1. Reads only the columns needed for graph inspection from the existing training split.
2. Reads at most 100,000 rows, avoiding loading the entire source dataset.
3. Counts sender accounts, receiver accounts, unique accounts, fraud-labelled transactions and transaction types in the sample.
4. Treats `nameOrig` as the source node and `nameDest` as the target node.
5. Saves the sample as `graph_model/data/graph_sample_100k.csv`.

This script does **not** train GraphSAGE and does **not** prove that a graph model improves fraud detection. It checks that we can prepare a manageable edge list.

## Exact command
Run from the FYP project root (the folder containing `Dataset (PaySim)` and `Experiments`):

```bash
python graph_model/data/inspect_graph_structure.py
```

## Important columns
| PaySim column | Meaning in graph work |
|---|---|
| `nameOrig` | Source/sender account node |
| `nameDest` | Target/receiver account node |
| `step` | Transaction time step |
| `amount` | Transaction amount |
| `type` | Transaction type |
| `isFraud` | Transaction label; target, not an input feature |

## What to learn
- A graph represents entities and their connections.
- A node is an account; an edge is a transaction.
- The graph is directed because sender and receiver have different roles.
- It is temporal because each transaction has a time step.
- This first step inspects graph data; it does not train the model.
- Do not use `isFraud` as an input feature. It is the label to predict.

## Results log
Fill this section only after running the script. Copy the actual output values; do not guess them.
- Rows inspected:
- Unique sender accounts:
- Unique receiver accounts:
- Unique accounts overall:
- Fraud-labelled transactions in the sample:
- Sender-equals-receiver transactions:
- Output CSV created:
- Any error or warning:

## Notes policy
After the command runs successfully, add the actual command, observed output, interpretation and what was learned to `docs/learning-notes.md` and `docs/experiment-log.md`. Do not record unobserved results.
