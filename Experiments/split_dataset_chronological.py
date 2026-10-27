"""
Create chronological PaySim splits from the original time-ordered CSV.

Run from the existing FYP project root:
    python Experiments/split_dataset_chronological.py

Train: steps 1-450
Validation: steps 451-600
Test: steps 601-743

This script does not modify Dataset (PaySim)/splits/ and does not train a model.
"""
from pathlib import Path

import pandas as pd

project_dir = Path(__file__).resolve().parent.parent

dataset_path = (
    project_dir
    / "Dataset (PaySim)"
    / "PS_20174392719_1491204439457_log.csv"
)

output_dir = project_dir / "Dataset (PaySim)" / "chronological_splits"

TRAIN_STEPS = (1, 450)
VALIDATION_STEPS = (451, 600)
TEST_STEPS = (601, 743)
CHUNK_SIZE = 100_000


def _in_range(steps, bounds):
    start, end = bounds
    return steps.between(start, end)


def main():
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Original PaySim CSV not found: {dataset_path}\n"
            "This script must read the original time-ordered dataset, "
            "not the random training split."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "train.csv"
    val_path = output_dir / "validation.csv"
    test_path = output_dir / "test.csv"
    split_paths = {
        "Training": train_path,
        "Validation": val_path,
        "Test": test_path,
    }

    for path in split_paths.values():
        if path.exists():
            path.unlink()

    print("Creating chronological splits from the original PaySim CSV...")
    print(f"Source file: {dataset_path}")
    print(f"Output folder: {output_dir}")
    print(f"Chunk size: {CHUNK_SIZE:,}")
    print("Existing random splits in Dataset (PaySim)/splits/ are not modified.")
    print(
        "Step ranges: train 1-450, validation 451-600, test 601-743."
    )

    stats = {
        name: {
            "rows": 0,
            "fraud": 0,
            "min_step": None,
            "max_step": None,
            "steps": set(),
        }
        for name in split_paths
    }

    original_rows = 0
    rows_in_1_to_743 = 0
    rows_outside_1_to_743 = 0
    write_header = {name: True for name in split_paths}

    reader = pd.read_csv(dataset_path, chunksize=CHUNK_SIZE)

    for chunk_number, chunk in enumerate(reader, start=1):
        original_rows += len(chunk)

        if "step" not in chunk.columns:
            raise ValueError("The source CSV does not contain a 'step' column.")
        if "isFraud" not in chunk.columns:
            raise ValueError("The source CSV does not contain an 'isFraud' column.")

        in_overall = chunk["step"].between(1, 743)
        rows_in_1_to_743 += int(in_overall.sum())
        rows_outside_1_to_743 += int((~in_overall).sum())

        masks = {
            "Training": _in_range(chunk["step"], TRAIN_STEPS),
            "Validation": _in_range(chunk["step"], VALIDATION_STEPS),
            "Test": _in_range(chunk["step"], TEST_STEPS),
        }

        if int(masks["Training"].sum() + masks["Validation"].sum() + masks["Test"].sum()) != int(
            in_overall.sum()
        ):
            raise ValueError(
                "A row with step between 1 and 743 was not assigned to exactly "
                "one chronological split."
            )

        for name, mask in masks.items():
            selected = chunk.loc[mask]
            if selected.empty:
                continue

            selected.to_csv(
                split_paths[name],
                index=False,
                mode="w" if write_header[name] else "a",
                header=write_header[name],
            )
            write_header[name] = False

            split_stats = stats[name]
            split_stats["rows"] += len(selected)
            split_stats["fraud"] += int(selected["isFraud"].sum())
            chunk_min = int(selected["step"].min())
            chunk_max = int(selected["step"].max())
            split_stats["min_step"] = (
                chunk_min
                if split_stats["min_step"] is None
                else min(split_stats["min_step"], chunk_min)
            )
            split_stats["max_step"] = (
                chunk_max
                if split_stats["max_step"] is None
                else max(split_stats["max_step"], chunk_max)
            )
            split_stats["steps"].update(selected["step"].astype(int).unique().tolist())

        if chunk_number % 10 == 0:
            print(f"Processed {original_rows:,} rows...")

    written_rows = sum(item["rows"] for item in stats.values())

    print("\nCHRONOLOGICAL SPLIT SUMMARY")
    print("---------------------------")
    for name in ("Training", "Validation", "Test"):
        item = stats[name]
        fraud_rate = (item["fraud"] / item["rows"] * 100) if item["rows"] else 0.0
        print(f"\n{name} set:")
        print(f"  Output file: {split_paths[name]}")
        print(f"  Total rows: {item['rows']:,}")
        print(f"  Fraud: {item['fraud']:,}")
        print(f"  Fraud percentage: {fraud_rate:.4f}%")
        print(f"  Minimum step: {item['min_step']}")
        print(f"  Maximum step: {item['max_step']}")
        print(f"  Distinct steps: {len(item['steps']):,}")

    train_steps = stats["Training"]["steps"]
    val_steps = stats["Validation"]["steps"]
    test_steps = stats["Test"]["steps"]
    overlap_train_val = train_steps & val_steps
    overlap_train_test = train_steps & test_steps
    overlap_val_test = val_steps & test_steps

    print("\nVERIFICATION")
    print(f"  Original CSV rows: {original_rows:,}")
    print(f"  Original rows with step 1-743: {rows_in_1_to_743:,}")
    print(f"  Original rows outside step 1-743: {rows_outside_1_to_743:,}")
    print(f"  Combined chronological-split rows: {written_rows:,}")
    print(
        "  Combined rows equal original rows with step 1-743: "
        f"{written_rows == rows_in_1_to_743}"
    )
    print(f"  Overlapping steps (train vs validation): {sorted(overlap_train_val)}")
    print(f"  Overlapping steps (train vs test): {sorted(overlap_train_test)}")
    print(f"  Overlapping steps (validation vs test): {sorted(overlap_val_test)}")

    if overlap_train_val or overlap_train_test or overlap_val_test:
        raise ValueError("Chronological splits contain overlapping steps.")
    if written_rows != rows_in_1_to_743:
        raise ValueError(
            "Combined chronological-split row count does not match the number "
            "of original rows with step between 1 and 743."
        )
    if stats["Training"]["min_step"] != TRAIN_STEPS[0] or stats["Training"]["max_step"] != TRAIN_STEPS[1]:
        print(
            "  Warning: training min/max step differs from the planned 1-450 range."
        )
    if (
        stats["Validation"]["min_step"] != VALIDATION_STEPS[0]
        or stats["Validation"]["max_step"] != VALIDATION_STEPS[1]
    ):
        print(
            "  Warning: validation min/max step differs from the planned 451-600 range."
        )
    if stats["Test"]["min_step"] != TEST_STEPS[0] or stats["Test"]["max_step"] != TEST_STEPS[1]:
        print(
            "  Warning: test min/max step differs from the planned 601-743 range."
        )

    print("\nChronological split files written.")
    print("Random splits, preprocessing artifacts and trained models were not modified.")


if __name__ == "__main__":
    main()
