
from pathlib import Path
from collections import Counter
import pandas as pd

# Locate the original dataset.
dataset_path = (
    Path(__file__).resolve().parent.parent
    / "Dataset (PaySim)"
    / "PS_20174392719_1491204439457_log.csv"
)

chunk_size = 100_000

# Counters collect information across all chunks.
transaction_types = Counter()
fraud_by_type = Counter()
flagged_by_label = Counter()

# Store amount statistics without retaining every transaction.
amount_stats = {}

print("Reading a sample of transactions...\n")

sample = pd.read_csv(dataset_path, nrows=10)
print(sample.to_string(index=False))

print("\nProfiling the complete dataset...")

reader = pd.read_csv(dataset_path, chunksize=chunk_size)

for chunk_number, chunk in enumerate(reader, start=1):

    # Count transactions by type.
    transaction_types.update(
        chunk["type"].value_counts().to_dict()
    )

    # Count fraud and non-fraud transactions within each type.
    counts = chunk.groupby(["type", "isFraud"]).size()

    for (transaction_type, label), count in counts.items():
        key = (transaction_type, int(label))
        fraud_by_type[key] += int(count)

    # Compare actual fraud labels with the existing flag.
    counts = chunk.groupby(
        ["isFraud", "isFlaggedFraud"]
    ).size()

    for (label, flag), count in counts.items():
        key = (int(label), int(flag))
        flagged_by_label[key] += int(count)

    # Calculate amount statistics separately for each label.
    for label, group in chunk.groupby("isFraud")["amount"]:
        label = int(label)

        if label not in amount_stats:
            amount_stats[label] = {
                "count": 0,
                "sum": 0.0,
                "min": float("inf"),
                "max": float("-inf"),
            }

        stats = amount_stats[label]
        stats["count"] += len(group)
        stats["sum"] += group.sum()
        stats["min"] = min(stats["min"], group.min())
        stats["max"] = max(stats["max"], group.max())

    if chunk_number % 10 == 0:
        print(f"Processed {chunk_number * chunk_size:,} rows...")

# Display the results.
print("\n========== TRANSACTION PROFILE ==========")

print("\n1. Transactions by type:")
for transaction_type, count in transaction_types.most_common():
    print(f"{transaction_type}: {count:,}")

print("\n2. Fraud distribution by transaction type:")
for transaction_type in sorted(transaction_types):
    print(f"\n{transaction_type}:")

    for label in [0, 1]:
        count = fraud_by_type.get((transaction_type, label), 0)
        description = "Fraud" if label == 1 else "Non-fraud"
        print(f"  {description}: {count:,}")

print("\n3. Transaction amount statistics:")
for label in sorted(amount_stats):
    stats = amount_stats[label]
    mean_amount = stats["sum"] / stats["count"]
    description = "Fraud" if label == 1 else "Non-fraud"

    print(f"\n{description}:")
    print(f"  Count: {stats['count']:,}")
    print(f"  Mean amount: {mean_amount:,.2f}")
    print(f"  Minimum amount: {stats['min']:,.2f}")
    print(f"  Maximum amount: {stats['max']:,.2f}")

print("\n4. Fraud label versus existing flag:")
for (label, flag), count in sorted(flagged_by_label.items()):
    print(
        f"isFraud={label}, isFlaggedFraud={flag}: "
        f"{count:,} transactions"
    )

print("\nProfiling complete!")
