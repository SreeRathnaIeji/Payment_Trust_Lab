
from pathlib import Path
import pandas as pd

dataset_path = (
    Path(__file__).resolve().parent.parent
    / "Dataset (PaySim)"
    / "PS_20174392719_1491204439457_log.csv"
)

chunk_size = 100_000

total_rows = 0
min_step = float("inf")
max_step = float("-inf")
fraud_by_step = {}
rows_by_step = {}

previous_last_step = None
order_violations = 0

print("Inspecting the time structure of PaySim...")

reader = pd.read_csv(
    dataset_path,
    chunksize=chunk_size,
    usecols=["step", "isFraud"],
)

for chunk_number, chunk in enumerate(reader, start=1):
    total_rows += len(chunk)

    min_step = min(min_step, chunk["step"].min())
    max_step = max(max_step, chunk["step"].max())

    # Check whether rows remain in non-decreasing step order.
    if previous_last_step is not None:
        if chunk["step"].iloc[0] < previous_last_step:
            order_violations += 1

    if not chunk["step"].is_monotonic_increasing:
        order_violations += 1

    previous_last_step = chunk["step"].iloc[-1]

    # Count total and fraud-labelled transactions per step.
    totals = chunk.groupby("step").size()
    frauds = chunk.groupby("step")["isFraud"].sum()

    for step, count in totals.items():
        rows_by_step[int(step)] = (
            rows_by_step.get(int(step), 0) + int(count)
        )

    for step, count in frauds.items():
        fraud_by_step[int(step)] = (
            fraud_by_step.get(int(step), 0) + int(count)
        )

    if chunk_number % 10 == 0:
        print(f"Processed {total_rows:,} rows...")

print("\n========== TIME STRUCTURE REPORT ==========")
print(f"Total rows: {total_rows:,}")
print(f"First step value: {int(min_step)}")
print(f"Last step value: {int(max_step)}")
print(f"Distinct step values: {len(rows_by_step)}")
print(f"Chunks with detected order issues: {order_violations}")

print("\nFirst 10 step values:")
for step in sorted(rows_by_step)[:10]:
    print(
        f"Step {step}: {rows_by_step[step]:,} transactions, "
        f"{fraud_by_step.get(step, 0):,} fraud"
    )

print("\nLast 10 step values:")
for step in sorted(rows_by_step)[-10:]:
    print(
        f"Step {step}: {rows_by_step[step]:,} transactions, "
        f"{fraud_by_step.get(step, 0):,} fraud"
    )

print("\nTime-structure inspection complete!")
