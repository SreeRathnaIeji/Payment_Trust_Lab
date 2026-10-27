
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from scipy.sparse import hstack
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
    "Logistic Regression-style": joblib.load(
        artifact_dir / "logistic_regression_v2.joblib"
    ),
    "Random Forest": joblib.load(
        artifact_dir / "random_forest.joblib"
    ),
    "XGBoost": joblib.load(
        artifact_dir / "xgboost.joblib"
    ),
}

isolation_forest = joblib.load(
    artifact_dir / "isolation_forest.joblib"
)

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

print("Step 1: Scoring all models on validation data...")

y_parts = []
score_parts = {name: [] for name in models}
score_parts["Isolation Forest"] = []

reader = pd.read_csv(
    validation_path,
    usecols=numeric_features
    + categorical_features
    + [target_column],
    chunksize=chunk_size,
)

for batch_number, chunk in enumerate(reader, start=1):
    X_numeric = scaler.transform(chunk[numeric_features])
    X_categorical = encoder.transform(
        chunk[categorical_features]
    )
    X = hstack(
        [X_numeric, X_categorical],
        format="csr",
    )

    y_parts.append(chunk[target_column].to_numpy())

    for name, model in models.items():
        scores = model.predict_proba(X)[:, 1]
        score_parts[name].append(scores)

    # Isolation Forest: larger values mean more anomalous.
    anomaly_scores = -isolation_forest.score_samples(
        X.toarray()
    )
    score_parts["Isolation Forest"].append(anomaly_scores)

    if batch_number % 3 == 0:
        print(f"Scored {batch_number * chunk_size:,} rows...")

y_true = np.concatenate(y_parts)

all_scores = {
    name: np.concatenate(parts)
    for name, parts in score_parts.items()
}

print(f"\nValidation rows: {len(y_true):,}")

# Supervised models use threshold 0.5 for this comparison.
# Isolation Forest flags the top fraction matching the
# overall PaySim fraud prevalence.
fraud_rate = 8213 / 6362620
if_scores = all_scores["Isolation Forest"]
if_threshold = np.quantile(if_scores, 1 - fraud_rate)

print(f"Isolation Forest threshold: {if_threshold:.6f}")

print("\n========== MODEL COMPARISON ==========")
print(
    f"{'Model':27} {'Accuracy':>9} "
    f"{'Precision':>10} {'Recall':>9} "
    f"{'F1':>9} {'AP':>9} "
    f"{'FP':>9} {'FN':>9}"
)

for name, scores in all_scores.items():
    if name == "Isolation Forest":
        predictions = (scores >= if_threshold).astype(int)
    else:
        predictions = (scores >= 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()

    accuracy = accuracy_score(y_true, predictions)
    precision = precision_score(
        y_true, predictions, zero_division=0
    )
    recall = recall_score(
        y_true, predictions, zero_division=0
    )
    f1 = f1_score(
        y_true, predictions, zero_division=0
    )
    ap = average_precision_score(y_true, scores)

    print(
        f"{name:27} {accuracy:9.4f} "
        f"{precision:10.4f} {recall:9.4f} "
        f"{f1:9.4f} {ap:9.4f} "
        f"{fp:9,} {fn:9,}"
    )

print("\nComparison complete.")
