
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from scipy.sparse import hstack
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    average_precision_score,
    precision_recall_curve,
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

train_path = split_dir / "train.csv"
validation_path = split_dir / "validation.csv"

print("Step 1: Counting training labels...")

label_counts = pd.read_csv(
    train_path,
    usecols=[target_column],
)[target_column].value_counts()

n_nonfraud = int(label_counts.get(0, 0))
n_fraud = int(label_counts.get(1, 0))

# Balanced class weights: each class contributes approximately
# equally to the total training loss.
total = n_nonfraud + n_fraud
class_weights = {
    0: total / (2 * n_nonfraud),
    1: total / (2 * n_fraud),
}

print("Training labels:", dict(label_counts))
print("Class weights:", class_weights)

print("\nStep 2: Initialising logistic regression...")

model = SGDClassifier(
    loss="log_loss",
    penalty="l2",
    alpha=0.0001,
    class_weight=class_weights,
    random_state=42,
)


print("\nStep 3: Training for 3 epochs...")

classes = np.array([0, 1])
rows_seen = 0

for epoch in range(1, 4):
    print(f"\nEpoch {epoch}/3")

    reader = pd.read_csv(
        train_path,
        usecols=numeric_features
        + categorical_features
        + [target_column],
        chunksize=chunk_size,
    )

    for batch_number, chunk in enumerate(reader, start=1):
        # Shuffle rows within this chunk.
        chunk = chunk.sample(
            frac=1,
            random_state=42 + epoch * 1000 + batch_number,
        )

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
        y = chunk[target_column].to_numpy()

        if epoch == 1 and batch_number == 1:
            model.partial_fit(X, y, classes=classes)
        else:
            model.partial_fit(X, y)

        rows_seen += len(chunk)

        if batch_number % 10 == 0:
            print(
                f"Epoch {epoch}: processed "
                f"{batch_number * chunk_size:,} rows"
            )

model_path = artifact_dir / "logistic_regression_v2.joblib"
joblib.dump(model, model_path)

print(f"\nSaved improved model to: {model_path}")

print("\nStep 4: Evaluating on validation data...")

y_true_parts = []
y_pred_parts = []
y_score_parts = []

reader = pd.read_csv(
    validation_path,
    usecols=numeric_features + categorical_features + [target_column],
    chunksize=chunk_size,
)

for chunk in reader:
    X_numeric = scaler.transform(chunk[numeric_features])
    X_categorical = encoder.transform(chunk[categorical_features])
    X = hstack([X_numeric, X_categorical], format="csr")

    y = chunk[target_column].to_numpy()
    scores = model.predict_proba(X)[:, 1]
    predictions = (scores >= 0.5).astype(int)

    y_true_parts.append(y)
    y_pred_parts.append(predictions)
    y_score_parts.append(scores)

y_true = np.concatenate(y_true_parts)
y_pred = np.concatenate(y_pred_parts)
y_scores = np.concatenate(y_score_parts)

print("\n========== VALIDATION RESULTS ==========")
print(f"Validation rows: {len(y_true):,}")
print(f"Accuracy: {accuracy_score(y_true, y_pred):.6f}")
print("Confusion matrix [ [TN, FP], [FN, TP] ]:")
print(confusion_matrix(y_true, y_pred, labels=[0, 1]))

print("\nClassification report:")
print(classification_report(
    y_true,
    y_pred,
    labels=[0, 1],
    target_names=["Non-fraud", "Fraud"],
    digits=4,
    zero_division=0,
))

print(
    "Average precision (PR-AUC summary): "
    f"{average_precision_score(y_true, y_scores):.6f}"
)

print("\nTraining and validation evaluation complete.")
