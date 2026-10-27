
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from scipy.sparse import hstack
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
)

project_dir = Path(__file__).resolve().parent.parent
split_dir = project_dir / "Dataset (PaySim)" / "splits"
artifact_dir = project_dir / "Experiments" / "artifacts"
artifact_dir.mkdir(parents=True, exist_ok=True)

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
sample_fraction = 0.10

train_path = split_dir / "train.csv"
validation_path = split_dir / "validation.csv"

print("Step 1: Sampling training data...")

sample_parts = []

reader = pd.read_csv(
    train_path,
    usecols=numeric_features
    + categorical_features
    + [target_column],
    chunksize=chunk_size,
)

for batch_number, chunk in enumerate(reader, start=1):
    sample = chunk.sample(
        frac=sample_fraction,
        random_state=42 + batch_number,
    )
    sample_parts.append(sample)

    if batch_number % 10 == 0:
        print(f"Processed {batch_number * chunk_size:,} rows...")

train_sample = pd.concat(sample_parts, ignore_index=True)

print(f"Sample rows: {len(train_sample):,}")
print(
    "Sample labels:",
    train_sample[target_column].value_counts().to_dict(),
)

print("\nStep 2: Transforming training sample...")

X_numeric = scaler.transform(train_sample[numeric_features])
X_categorical = encoder.transform(
    train_sample[categorical_features]
)
X_train = hstack(
    [X_numeric, X_categorical],
    format="csr",
)
y_train = train_sample[target_column].to_numpy()

print("\nStep 3: Training XGBoost...")

fraud_count = int((y_train == 1).sum())
nonfraud_count = int((y_train == 0).sum())

if fraud_count == 0:
    raise ValueError("Training sample contains no fraud cases.")

model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=1.0,
    objective="binary:logistic",
    eval_metric="aucpr",
    scale_pos_weight=nonfraud_count / fraud_count,
    tree_method="hist",
    random_state=42,
    n_jobs=-1,
)

model.fit(X_train, y_train)

model_path = artifact_dir / "xgboost.joblib"
joblib.dump(model, model_path)
print(f"Saved model to: {model_path}")

del train_sample, X_train, y_train, sample_parts

print("\nStep 4: Evaluating on full validation data...")

y_true_parts = []
y_pred_parts = []
y_score_parts = []

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
    )

    y = chunk[target_column].to_numpy()
    scores = model.predict_proba(X)[:, 1]
    predictions = (scores >= 0.5).astype(int)

    y_true_parts.append(y)
    y_pred_parts.append(predictions)
    y_score_parts.append(scores)

y_true = np.concatenate(y_true_parts)
y_pred = np.concatenate(y_pred_parts)
y_scores = np.concatenate(y_score_parts)

print("\n========== XGBOOST RESULTS ==========")
print(f"Validation rows: {len(y_true):,}")
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
    f"{average_precision_score(y_true, y_scores):.6f}"
)

print("\nXGBoost training and evaluation complete.")
