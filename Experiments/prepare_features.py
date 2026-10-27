
from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Project paths
project_dir = Path(__file__).resolve().parent.parent
split_dir = project_dir / "Dataset (PaySim)" / "splits"
output_dir = project_dir / "Experiments" / "artifacts"
output_dir.mkdir(parents=True, exist_ok=True)

train_path = split_dir / "train.csv"

# Features used by the initial tabular models
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

print("Step 1: Fitting numerical scaler on training data...")

scaler = StandardScaler()

# Incrementally learn numerical means and standard deviations.
reader = pd.read_csv(
    train_path,
    usecols=numeric_features,
    chunksize=chunk_size,
)

total_rows = 0

for chunk in reader:
    scaler.partial_fit(chunk)
    total_rows += len(chunk)

print(f"Rows used to fit scaler: {total_rows:,}")

print("\nStep 2: Fitting transaction-type encoder...")

# These are the five transaction types found in our dataset.
known_types = [
    "CASH_IN",
    "CASH_OUT",
    "DEBIT",
    "PAYMENT",
    "TRANSFER",
]

encoder = OneHotEncoder(
    categories=[known_types],
    handle_unknown="ignore",
    sparse_output=True,
)

# Fit the encoder using training data only.
training_types = pd.read_csv(
    train_path,
    usecols=categorical_features,
    nrows=100_000,
)

encoder.fit(training_types)

print("Transaction types:", list(encoder.categories_[0]))

print("\nStep 3: Saving preprocessing components...")

joblib.dump(scaler, output_dir / "numeric_scaler.joblib")
joblib.dump(encoder, output_dir / "type_encoder.joblib")

configuration = {
    "numeric_features": numeric_features,
    "categorical_features": categorical_features,
    "target_column": target_column,
    "known_transaction_types": known_types,
    "chunk_size": chunk_size,
}

with open(output_dir / "feature_config.json", "w") as file:
    json.dump(configuration, file, indent=4)

print("Saved scaler, encoder and feature configuration.")

print("\nStep 4: Testing transformations on a small sample...")

sample_columns = numeric_features + categorical_features

sample = pd.read_csv(
    train_path,
    usecols=sample_columns,
    nrows=1_000,
)

scaled_numeric = scaler.transform(sample[numeric_features])
encoded_type = encoder.transform(sample[categorical_features])

print("Input sample rows:", len(sample))
print("Scaled numerical shape:", scaled_numeric.shape)
print("Encoded transaction-type shape:", encoded_type.shape)
print("Encoded output is sparse:", hasattr(encoded_type, "tocsr"))

print("\nNumerical feature means learned by scaler:")
for name, mean in zip(numeric_features, scaler.mean_):
    print(f"{name}: {mean:.4f}")

print("\nPreprocessing components saved successfully!")
print("Artifact folder:", output_dir)
