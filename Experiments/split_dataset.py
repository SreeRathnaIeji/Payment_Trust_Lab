
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Locate the original dataset and output folder.
project_dir = Path(__file__).resolve().parent.parent

dataset_path = (
    project_dir
    / "Dataset (PaySim)"
    / "PS_20174392719_1491204439457_log.csv"
)

output_dir = project_dir / "Dataset (PaySim)" / "splits"
output_dir.mkdir(parents=True, exist_ok=True)

# Fixed random seed makes the split reproducible.
random_seed = 42
chunk_size = 100_000

print("Step 1: Reading the target column...")

# Read only the fraud labels to reduce memory usage.
labels = pd.read_csv(
    dataset_path,
    usecols=["isFraud"],
)["isFraud"]

total_rows = len(labels)
row_indices = np.arange(total_rows)

print(f"Total rows: {total_rows:,}")
print("Creating stratified splits...")

# First, reserve 15% for the final test set.
train_val_idx, test_idx = train_test_split(
    row_indices,
    test_size=0.15,
    random_state=random_seed,
    stratify=labels,
)

# Split the remaining 85% into 70% training and 15% validation.
train_idx, val_idx = train_test_split(
    train_val_idx,
    test_size=(0.15 / 0.85),
    random_state=random_seed,
    stratify=labels.iloc[train_val_idx],
)

# Store the split membership for each original row.
split_codes = np.empty(total_rows, dtype=np.uint8)
split_codes[train_idx] = 0
split_codes[val_idx] = 1
split_codes[test_idx] = 2

del labels, row_indices, train_val_idx
del train_idx, val_idx, test_idx

print("Step 2: Writing the split files...")
print("This may take some time because the CSV is large.")

train_path = output_dir / "train.csv"
val_path = output_dir / "validation.csv"
test_path = output_dir / "test.csv"

# Remove old output files so repeated runs do not append duplicates.
for path in [train_path, val_path, test_path]:
    if path.exists():
        path.unlink()

row_offset = 0
first_chunk = True

reader = pd.read_csv(dataset_path, chunksize=chunk_size)

for chunk_number, chunk in enumerate(reader, start=1):
    end_offset = row_offset + len(chunk)
    codes = split_codes[row_offset:end_offset]

    # Write each chunk's rows to the correct output file.
    for code, path in [
        (0, train_path),
        (1, val_path),
        (2, test_path),
    ]:
        selected = chunk.loc[codes == code]

        if not selected.empty:
            selected.to_csv(
                path,
                index=False,
                mode="w" if first_chunk else "a",
                header=first_chunk,
            )

    row_offset = end_offset
    first_chunk = False

    if chunk_number % 10 == 0:
        print(f"Processed {row_offset:,} rows...")

del split_codes

print("\nStep 3: Checking the output files...")

for name, path in [
    ("Training", train_path),
    ("Validation", val_path),
    ("Test", test_path),
]:
    # Count rows in chunks instead of loading the whole output file.
    row_count = 0
    fraud_count = 0

    for part in pd.read_csv(
        path,
        usecols=["isFraud"],
        chunksize=chunk_size,
    ):
        row_count += len(part)
        fraud_count += int(part["isFraud"].sum())

    non_fraud_count = row_count - fraud_count
    fraud_rate = fraud_count / row_count * 100

    print(f"\n{name} set:")
    print(f"  Total rows: {row_count:,}")
    print(f"  Fraud: {fraud_count:,}")
    print(f"  Non-fraud: {non_fraud_count:,}")
    print(f"  Fraud percentage: {fraud_rate:.4f}%")

print("\nDataset splitting complete!")
print(f"Output folder: {output_dir}")
