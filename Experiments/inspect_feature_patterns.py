
from pathlib import Path
import pandas as pd

project_dir = Path(__file__).resolve().parent.parent
dataset_path = (
    project_dir
    / "Dataset (PaySim)"
    / "splits"
    / "train.csv"
)

chunk_size = 100_000

columns = [
    "type",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud",
]

# Accumulate weighted sums and counts for each group.
stats = {}

print("Inspecting feature patterns in the training set...")

reader = pd.read_csv(
    dataset_path,
    usecols=columns,
    chunksize=chunk_size,
)

for chunk_number, chunk in enumerate(reader, start=1):
    numeric_columns = [
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
    ]

    grouped = chunk.groupby(["type", "isFraud"])

    for (transaction_type, label), group in grouped:
        key = (transaction_type, int(label))

        if key not in stats:
            stats[key] = {
                "count": 0,
                **{
                    column: 0.0
                    for column in numeric_columns
                },
            }

        stats[key]["count"] += len(group)

        for column in numeric_columns:
            stats[key][column] += group[column].sum()

    if chunk_number % 10 == 0:
        print(
            f"Processed approximately "
            f"{chunk_number * chunk_size:,} rows..."
        )

print("\n========== FEATURE PATTERNS ==========")

for (transaction_type, label), values in sorted(stats.items()):
    count = values["count"]
    description = "Fraud" if label == 1 else "Non-fraud"

    print(f"\nType: {transaction_type} | Class: {description}")
    print(f"Count: {count:,}")

    for column in [
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
    ]:
        mean_value = values[column] / count
        print(f"Mean {column}: {mean_value:,.2f}")

print("\nFeature-pattern inspection complete!")
