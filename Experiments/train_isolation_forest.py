
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from scipy.sparse import hstack
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
)

project_dir = Path(__file__).resolve().parent.parent
split_dir = project_dir / "Dataset (PaySim)" / "splits"
artifact_dir = project_dir / "Experiments" / "artifacts"

scaler = joblib.load(artifact_dir / "numeric_scaler.joblib")
encoder = joblib.load(artifact_dir / "type_encoder.joblib")

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

train_path = split_dir / "train.csv"
validation_path = split_dir / "validation.csv"

print("Step 1: Sampling genuine training transactions...")

sample_parts = []

reader = pd.read_csv(
    train_path,
    usecols=numeric_features
    + categorical_features
    + [target_column],
    chunksize=chunk_size,
)

for batch_number, chunk in enumerate(reader, start=1):
    # Train this anomaly detector using genuine transactions only.
    genuine = chunk[chunk[target_column] == 0]

    sample = genuine.sample(
        frac=0.05,
        random_state=42 + batch_number,
    )
    sample_parts.append(sample)

    if batch_number % 10 == 0:
        print(f"Processed {batch_number * chunk_size:,} rows...")

train_sample = pd.concat(sample_parts, ignore_index=True)

print(f"Genuine training sample: {len(train_sample):,}")

print("\nStep 2: Transforming training sample...")

X_numeric = scaler.transform(train_sample[numeric_features])
X_categorical = encoder.transform(
    train_sample[categorical_features]
)
X_train = hstack(
    [X_numeric, X_categorical],
    format="csr",
).toarray()

del train_sample, sample_parts

print("\nStep 3: Training Isolation Forest...")

model = IsolationForest(
    n_estimators=100,
    max_samples=256,
    contamination="auto",
    random_state=42,
    n_jobs=-1,
)

model.fit(X_train)
del X_train

model_path = artifact_dir / "isolation_forest.joblib"
joblib.dump(model, model_path)
print(f"Saved model to: {model_path}")

print("\nStep 4: Scoring validation transactions...")

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
    X_numeric = scaler.transform(chunk[numeric_features])
    X_categorical = encoder.transform(
        chunk[categorical_features]
    )
    X = hstack(
        [X_numeric, X_categorical],
        format="csr",
    ).toarray()

    # Higher scores below indicate greater anomaly.
    anomaly_scores = -model.score_samples(X)

    y_true_parts.append(chunk[target_column].to_numpy())
    score_parts.append(anomaly_scores)

y_true = np.concatenate(y_true_parts)
anomaly_scores = np.concatenate(score_parts)

# Use the top 0.1291% of anomaly scores as predicted fraud.
# This matches the approximate fraud prevalence in the dataset.
fraud_rate = 8213 / 6362620
threshold = np.quantile(anomaly_scores, 1 - fraud_rate)
y_pred = (anomaly_scores >= threshold).astype(int)

print("\n========== ISOLATION FOREST RESULTS ==========")
print(f"Validation rows: {len(y_true):,}")
print(f"Anomaly threshold: {threshold:.6f}")
print(f"Accuracy: {accuracy_score(y_true, y_pred):.6f}")
print("Confusion matrix [ [TN, FP], [FN, TP] ]:")
print(confusion_matrix(y_true, y_pred, labels=[0, 1]))

print("\nClassification report:")
print(
    classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["Non-fraud", "Fraud"],
        digits=4,
        zero_division=0,
    )
)

print(
    "Average precision (AP): "
    f"{average_precision_score(y_true, anomaly_scores):.6f}"
)

print("\nIsolation Forest training and evaluation complete.")
