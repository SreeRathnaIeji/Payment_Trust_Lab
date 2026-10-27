
from pathlib import Path
from collections import Counter
import pandas as pd

# Locate the original dataset.
dataset_path = (
    Path(__file__).resolve().parent.parent
    / "Dataset (PaySim)"
    / "PS_20174392719_1491204439457_log.csv"
)

# Read the large CSV in smaller chunks.
chunk_size = 100_000

total_rows = 0
missing_values = None
fraud_counts = Counter()
column_names = []
column_types = None

print("Starting dataset inspection...")
print("Dataset:", dataset_path)

reader = pd.read_csv(dataset_path, chunksize=chunk_size)

for chunk_number, chunk in enumerate(reader, start=1):

    # Collect column information from the first chunk.
    if chunk_number == 1:
        column_names = list(chunk.columns)
        column_types = chunk.dtypes.astype(str)
        missing_values = pd.Series(
            0, index=chunk.columns, dtype="int64"
        )

    # Count rows and missing values.
    total_rows += len(chunk)
    missing_values += chunk.isna().sum()

    # Count fraud and non-fraud labels.
    if "isFraud" in chunk.columns:
        fraud_counts.update(
            chunk["isFraud"].value_counts().to_dict()
        )

    # Display progress periodically.
    if chunk_number % 10 == 0:
        print(f"Processed {total_rows:,} rows...")

# Display the final report.
print("\n========== DATASET REPORT ==========")

print(f"\nTotal rows: {total_rows:,}")
print(f"Total columns: {len(column_names)}")

print("\nColumn names:")
for column in column_names:
    print(f"- {column}")

print("\nColumn data types (first chunk):")
print(column_types)

print("\nFraud distribution:")
for label, count in sorted(fraud_counts.items()):
    print(f"Label {label}: {count:,} rows")

print("\nMissing values per column:")
print(missing_values[missing_values > 0])

print("\nTotal missing values:", int(missing_values.sum()))

print("\nInspection complete!")