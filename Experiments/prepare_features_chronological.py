"""
Fit chronological-split preprocessing components on training data only.

Run from the FYP project root:
    python Experiments/prepare_features_chronological.py

Writes to Experiments/artifacts/chronological/. Does not modify the random-split
artifacts or Dataset (PaySim)/splits/.
"""
from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

project_dir = Path(__file__).resolve().parent.parent
split_dir = project_dir / "Dataset (PaySim)" / "chronological_splits"
output_dir = project_dir / "Experiments" / "artifacts" / "chronological"
output_dir.mkdir(parents=True, exist_ok=True)

train_path = split_dir / "train.csv"

numeric_features = [
    "step",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
]
ablation_numeric_features = ["step", "amount"]
categorical_features = ["type"]
target_column = "isFraud"
chunk_size = 100_000
known_types = [
    "CASH_IN",
    "CASH_OUT",
    "DEBIT",
    "PAYMENT",
    "TRANSFER",
]

excluded_from_inputs = [
    "isFraud",
    "isFlaggedFraud",
    "nameOrig",
    "nameDest",
]


def main():
    if not train_path.exists():
        raise FileNotFoundError(
            f"Chronological training split not found: {train_path}"
        )

    print("Fitting chronological preprocessing on training data only...")
    print(f"Training file: {train_path}")
    print(f"Output folder: {output_dir}")
    print(
        "Excluded from model inputs: "
        + ", ".join(excluded_from_inputs)
    )
    print(
        "Existing random-split artifacts in Experiments/artifacts/ "
        "are not modified."
    )

    main_scaler = StandardScaler()
    ablation_scaler = StandardScaler()
    total_rows = 0

    print("\nStep 1: Fitting numerical scalers...")
    reader = pd.read_csv(
        train_path,
        usecols=numeric_features,
        chunksize=chunk_size,
    )
    for chunk in reader:
        main_scaler.partial_fit(chunk[numeric_features])
        ablation_scaler.partial_fit(chunk[ablation_numeric_features])
        total_rows += len(chunk)

    print(f"Rows used to fit scalers: {total_rows:,}")

    print("\nStep 2: Fitting transaction-type encoder...")
    encoder = OneHotEncoder(
        categories=[known_types],
        handle_unknown="ignore",
        sparse_output=True,
    )
    training_types = pd.read_csv(
        train_path,
        usecols=categorical_features,
        nrows=100_000,
    )
    encoder.fit(training_types)
    print("Transaction types:", list(encoder.categories_[0]))

    print("\nStep 3: Saving chronological preprocessing components...")
    joblib.dump(main_scaler, output_dir / "numeric_scaler.joblib")
    joblib.dump(
        ablation_scaler,
        output_dir / "numeric_scaler_no_balance.joblib",
    )
    joblib.dump(encoder, output_dir / "type_encoder.joblib")

    configuration = {
        "split_directory": "Dataset (PaySim)/chronological_splits",
        "numeric_features": numeric_features,
        "ablation_numeric_features": ablation_numeric_features,
        "categorical_features": categorical_features,
        "target_column": target_column,
        "excluded_from_inputs": excluded_from_inputs,
        "known_transaction_types": known_types,
        "chunk_size": chunk_size,
        "rows_used_to_fit_scalers": total_rows,
        "notes": [
            "Main feature set matches the existing tabular baseline.",
            "Balance columns are omitted only in the separate no-balance ablation.",
            "step is retained in both sets; chronological val/test steps are outside the training range.",
        ],
    }
    with open(output_dir / "feature_config.json", "w", encoding="utf-8") as file:
        json.dump(configuration, file, indent=4)

    print("Saved:")
    print(f"  {output_dir / 'numeric_scaler.joblib'}")
    print(f"  {output_dir / 'numeric_scaler_no_balance.joblib'}")
    print(f"  {output_dir / 'type_encoder.joblib'}")
    print(f"  {output_dir / 'feature_config.json'}")

    print("\nMain scaler training means:")
    for name, mean in zip(numeric_features, main_scaler.mean_):
        print(f"  {name}: {mean:.4f}")

    print("\nNo-balance scaler training means:")
    for name, mean in zip(ablation_numeric_features, ablation_scaler.mean_):
        print(f"  {name}: {mean:.4f}")

    print("\nChronological preprocessing complete.")


if __name__ == "__main__":
    main()
