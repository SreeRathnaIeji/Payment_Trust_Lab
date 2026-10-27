
from pathlib import Path
import pandas as pd

dataset_path = (
    Path(__file__).resolve().parent.parent
    / "Dataset (PaySim)"
    / "PS_20174392719_1491204439457_log.csv"
)

# Define five time ranges.
ranges = [
    (1, 150),
    (151, 300),
    (301, 450),
    (451, 600),
    (601, 743),
]

results = {
    label: {"transactions": 0, "fraud": 0}
    for label in ranges
}

reader = pd.read_csv(
    dataset_path,
    usecols=["step", "isFraud"],
    chunksize=100_000,
)

print("Analysing fraud distribution across time ranges...")

for chunk in reader:
    for start, end in ranges:
        mask = chunk["step"].between(start, end)
        selected = chunk.loc[mask, "isFraud"]

        results[(start, end)]["transactions"] += len(selected)
        results[(start, end)]["fraud"] += int(selected.sum())

print("\n========== FRAUD BY TIME RANGE ==========")

for (start, end), counts in results.items():
    total = counts["transactions"]
    fraud = counts["fraud"]
    non_fraud = total - fraud
    fraud_percentage = (fraud / total * 100) if total else 0

    print(f"\nSteps {start}-{end}")
    print(f"Transactions: {total:,}")
    print(f"Fraud: {fraud:,}")
    print(f"Non-fraud: {non_fraud:,}")
    print(f"Fraud percentage: {fraud_percentage:.4f}%")

print("\nAnalysis complete!")
