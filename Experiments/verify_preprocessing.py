
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack

# Project folders
project_dir = Path(__file__).resolve().parent.parent
split_dir = project_dir / "Dataset (PaySim)" / "splits"
artifact_dir = project_dir / "Experiments" / "artifacts"

# Load the components fitted using training data.
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

datasets = {
    "Training": split_dir / "train.csv",
    "Validation": split_dir / "validation.csv",
    "Test": split_dir / "test.csv",
}

print("Verifying saved preprocessing components...")

for dataset_name, dataset_path in datasets.items():
    total_rows = 0
    fraud_count = 0
    invalid_numeric_rows = 0
    transformed_rows = 0
    feature_count = None

    reader = pd.read_csv(
        dataset_path,
        usecols=numeric_features + categorical_features + [target_column],
        chunksize=chunk_size,
    )

    for chunk in reader:
        # Separate input features from the target label.
        X_numeric = chunk[numeric_features]
        X_categorical = chunk[categorical_features]
        y = chunk[target_column]

        # Apply the already-fitted transformations.
        numeric_array = scaler.transform(X_numeric)
        encoded_array = encoder.transform(X_categorical)

        # Check numerical values for NaN or infinity.
        valid_rows = np.isfinite(numeric_array).all(axis=1)
        invalid_numeric_rows += int((~valid_rows).sum())

        # Combine numerical and categorical features.
        X_transformed = hstack(
            [numeric_array, encoded_array],
            format="csr",
        )

        if feature_count is None:
            feature_count = X_transformed.shape[1]

        total_rows += len(chunk)
        fraud_count += int(y.sum())
        transformed_rows += X_transformed.shape[0]

    print(f"\n{dataset_name} set:")
    print(f"  Rows checked: {total_rows:,}")
    print(f"  Fraud labels: {fraud_count:,}")
    print(f"  Transformed rows: {transformed_rows:,}")
    print(f"  Input feature count: {feature_count}")
    print(f"  Rows with invalid scaled numeric values: {invalid_numeric_rows}")

    if transformed_rows != total_rows:
        raise ValueError(f"Row count mismatch in {dataset_name} set.")

    if feature_count != 11:
        raise ValueError(f"Expected 11 features in {dataset_name} set.")

    if invalid_numeric_rows > 0:
        raise ValueError(
            f"Invalid numerical values found in {dataset_name} set."
        )

print("\nPreprocessing verification completed successfully!")
