
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from scipy.sparse import hstack
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

project_dir = Path(__file__).resolve().parent.parent
split_dir = project_dir / "Dataset (PaySim)" / "splits"
artifact_dir = project_dir / "Experiments" / "artifacts"

model = joblib.load(
    artifact_dir / "logistic_regression_v2.joblib"
)
scaler = joblib.load(
    artifact_dir / "numeric_scaler.joblib"
)
encoder = joblib.load(
    artifact_dir / "type_encoder.joblib"
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

print("Step 1: Generating validation scores...")

y_true_parts = []
score_parts = []

reader = pd.read_csv(
    validation_path,
    usecols=numeric_features
    + categorical_features
    + [target_column],
    chunksize=chunk_size,
)

for chunk in reader:
    X_numeric = scaler.transform(
        chunk[numeric_features]
    )
    X_categorical = encoder.transform(
        chunk[categorical_features]
    )

    X = hstack(
        [X_numeric, X_categorical],
        format="csr",
    )

    scores = model.predict_proba(X)[:, 1]

    y_true_parts.append(
        chunk[target_column].to_numpy()
    )
    score_parts.append(scores)

y_true = np.concatenate(y_true_parts)
y_scores = np.concatenate(score_parts)

print(f"Validation rows: {len(y_true):,}")
print(
    "Score percentiles (50th, 90th, 95th, "
    "99th, 99.9th):"
)
print(
    np.percentile(
        y_scores,
        [50, 90, 95, 99, 99.9],
    )
)

thresholds = [
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
    0.95,
    0.99,
    0.995,
    0.999,
]

print("\nStep 2: Comparing thresholds...")
print(
    f"{'Threshold':>9} "
    f"{'Precision':>10} "
    f"{'Recall':>10} "
    f"{'F1':>10} "
    f"{'False Pos.':>12} "
    f"{'True Pos.':>10} "
    f"{'Flagged':>10}"
)

for threshold in thresholds:
    y_pred = (y_scores >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    precision = precision_score(
        y_true, y_pred, zero_division=0
    )
    recall = recall_score(
        y_true, y_pred, zero_division=0
    )
    f1 = f1_score(
        y_true, y_pred, zero_division=0
    )

    print(
        f"{threshold:9.3f} "
        f"{precision:10.4f} "
        f"{recall:10.4f} "
        f"{f1:10.4f} "
        f"{fp:12,} "
        f"{tp:10,} "
        f"{int(y_pred.sum()):10,}"
    )

print("\nThreshold evaluation complete.")
