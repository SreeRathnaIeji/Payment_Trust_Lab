
from itertools import combinations
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from scipy.sparse import hstack
from scipy.stats import rankdata
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

project_dir = Path(__file__).resolve().parent.parent
split_dir = project_dir / "Dataset (PaySim)" / "splits"
artifact_dir = project_dir / "Experiments" / "artifacts"

scaler = joblib.load(artifact_dir / "numeric_scaler.joblib")
encoder = joblib.load(artifact_dir / "type_encoder.joblib")

models = {
    "Logistic Regression": joblib.load(
        artifact_dir / "logistic_regression_v2.joblib"
    ),
    "Random Forest": joblib.load(
        artifact_dir / "random_forest.joblib"
    ),
    "XGBoost": joblib.load(
        artifact_dir / "xgboost.joblib"
    ),
    "Isolation Forest": joblib.load(
        artifact_dir / "isolation_forest.joblib"
    ),
}

numeric_features = [
    "step",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
]
categorical_features = ["type"]
target_column = "isFraud"
chunk_size = 100_000

validation_path = split_dir / "validation.csv"

print("Step 1: Scoring the validation set...")

y_parts = []
raw_scores = {name: [] for name in models}

reader = pd.read_csv(
    validation_path,
    usecols=numeric_features
    + categorical_features
    + [target_column],
    chunksize=chunk_size,
)

for batch_number, chunk in enumerate(reader, start=1):
    X_num = scaler.transform(chunk[numeric_features])
    X_cat = encoder.transform(chunk[categorical_features])
    X = hstack([X_num, X_cat], format="csr")

    y_parts.append(chunk[target_column].to_numpy())

    for name, model in models.items():
        if name == "Isolation Forest":
            scores = -model.score_samples(X.toarray())
        else:
            scores = model.predict_proba(X)[:, 1]

        raw_scores[name].append(scores)

    if batch_number % 3 == 0:
        print(f"Scored {batch_number * chunk_size:,} rows...")

y_true = np.concatenate(y_parts)
n = len(y_true)

print(f"Validation rows: {n:,}")
print("Step 2: Converting scores to percentile ranks...")

rank_scores = {}

for name, parts in raw_scores.items():
    scores = np.concatenate(parts)
    rank_scores[name] = rankdata(scores) / n

# Use the known overall PaySim fraud rate to set a common
# predicted-positive proportion, without tuning on validation labels.
fraud_rate = 8213 / 6362620
positive_count = int(round(fraud_rate * n))
positive_count = max(1, positive_count)

print(f"Common predicted-fraud count: {positive_count:,}")

def evaluate(name, scores):
    threshold = np.partition(
        scores, n - positive_count
    )[n - positive_count]

    predictions = (scores >= threshold).astype(int)

    # Resolve ties deterministically if any scores are equal
    # at the cutoff, retaining exactly positive_count cases.
    if predictions.sum() > positive_count:
        tied = np.flatnonzero(scores == threshold)
        predictions[tied] = 0
        need = positive_count - int(predictions.sum())
        if need > 0:
            predictions[tied[:need]] = 1

    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()

    return {
        "Model": name,
        "Accuracy": accuracy_score(y_true, predictions),
        "Precision": precision_score(
            y_true, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_true, predictions, zero_division=0
        ),
        "F1": f1_score(y_true, predictions, zero_division=0),
        "AP": average_precision_score(y_true, scores),
        "FP": fp,
        "FN": fn,
        "TP": tp,
    }

results = []

print("Step 3: Evaluating individual models...")
for name, scores in rank_scores.items():
    results.append(evaluate(name, scores))

print("Step 4: Evaluating all six pairwise combinations...")
for first, second in combinations(rank_scores.keys(), 2):
    combined_scores = (
        rank_scores[first] + rank_scores[second]
    ) / 2

    results.append(
        evaluate(f"{first} + {second}", combined_scores)
    )

results_df = pd.DataFrame(results)

print("\n========== INDIVIDUAL AND PAIRWISE RESULTS ==========")
print(
    results_df.to_string(
        index=False,
        formatters={
            "Accuracy": "{:.6f}".format,
            "Precision": "{:.4f}".format,
            "Recall": "{:.4f}".format,
            "F1": "{:.4f}".format,
            "AP": "{:.6f}".format,
        },
    )
)

output_path = artifact_dir / "pairwise_ensemble_results.csv"
results_df.to_csv(output_path, index=False)

print(f"\nSaved comparison table to: {output_path}")
print("\nPairwise comparison complete.")
