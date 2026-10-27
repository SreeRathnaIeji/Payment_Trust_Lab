
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parent.parent
SPLITS = ROOT / "Dataset (PaySim)" / "chronological_splits"
OUT = ROOT / "Experiments" / "artifacts" / "chronological" / "xgboost"
OUT.mkdir(parents=True, exist_ok=True)

TARGET = "isFraud"
TYPE = "type"
MAIN_FEATURES = [
    "step", "amount", "oldbalanceOrg", "newbalanceOrig",
    "oldbalanceDest", "newbalanceDest",
]
NO_BALANCE_FEATURES = ["step", "amount"]
TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
CHUNK_SIZE = 100_000
SEED = 42


def load_data(path, features, encoder, sample=False):
    X_parts, y_parts = [], []
    columns = features + [TYPE, TARGET]

    for i, chunk in enumerate(
        pd.read_csv(path, usecols=columns, chunksize=CHUNK_SIZE)
    ):
        if sample:
            chunk = chunk.sample(frac=0.10, random_state=SEED + i)

        numeric = chunk[features].to_numpy(dtype=np.float32)
        categorical = encoder.transform(chunk[[TYPE]]).astype(np.float32)
        X_parts.append(np.concatenate([numeric, categorical], axis=1))
        y_parts.append(chunk[TARGET].to_numpy(dtype=np.int8))

    return np.vstack(X_parts), np.concatenate(y_parts)


def score(y, probabilities, threshold):
    predicted = (probabilities >= threshold).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(
        y, predicted, labels=[0, 1]
    ).ravel()

    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "average_precision": float(average_precision_score(y, probabilities)),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }


def run_experiment(name, features, encoder):
    print(f"\n========== {name} ==========")

    X_train, y_train = load_data(
        SPLITS / "train.csv", features, encoder, sample=True
    )
    positives = int(y_train.sum())
    negatives = len(y_train) - positives

    if positives == 0:
        raise ValueError("Training sample contains no fraud cases.")

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=1.0,
        objective="binary:logistic",
        eval_metric="aucpr",
        scale_pos_weight=negatives / positives,
        tree_method="hist",
        random_state=SEED,
        n_jobs=-1,
    )

    print(f"Training rows: {len(y_train):,}")
    print(f"Fraud cases in sample: {positives:,}")
    model.fit(X_train, y_train)

    model_path = OUT / f"xgboost_{name}.joblib"
    joblib.dump(model, model_path)
    del X_train, y_train

    # Validation: choose the threshold here, not on test data.
    X_val, y_val = load_data(
        SPLITS / "validation.csv", features, encoder
    )
    val_prob = model.predict_proba(X_val)[:, 1]
    del X_val

    thresholds = list(np.arange(0.01, 1.00, 0.01)) + [0.995, 0.999]
    candidates = [score(y_val, val_prob, t) for t in thresholds]
    best = max(candidates, key=lambda r: (r["f1"], r["precision"]))
    del y_val, val_prob

    pd.DataFrame(candidates).to_csv(
        OUT / f"{name}_validation_thresholds.csv", index=False
    )
    print("Best validation result:")
    print(json.dumps(best, indent=2))

    # Test: evaluated using the fixed validation threshold.
    X_test, y_test = load_data(
        SPLITS / "test.csv", features, encoder
    )
    test_prob = model.predict_proba(X_test)[:, 1]
    test_result = score(y_test, test_prob, best["threshold"])
    print("Test result:")
    print(json.dumps(test_result, indent=2))

    result = {
        "experiment": name,
        "features": features + [TYPE],
        "excluded_inputs": [
            "isFraud", "isFlaggedFraud", "nameOrig", "nameDest"
        ],
        "training_sample_fraction": 0.10,
        "validation_selected_threshold": best["threshold"],
        "validation_metrics": best,
        "test_metrics": test_result,
        "model_file": str(model_path.relative_to(ROOT)),
    }

    with open(OUT / f"{name}_results.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    del X_test, y_test, test_prob
    return result


def main():
    for filename in ("train.csv", "validation.csv", "test.csv"):
        if not (SPLITS / filename).exists():
            raise FileNotFoundError(f"Missing chronological split: {filename}")

    # The predefined categories are fitted using training rows only.
    encoder = OneHotEncoder(
        categories=[TYPES],
        handle_unknown="ignore",
        sparse_output=False,
        dtype=np.float32,
    )
    first_rows = pd.read_csv(
        SPLITS / "train.csv", usecols=[TYPE], nrows=100_000
    )
    encoder.fit(first_rows[[TYPE]])
    joblib.dump(encoder, OUT / "type_encoder.joblib")

    results = [
        run_experiment("main_features", MAIN_FEATURES, encoder),
        run_experiment("no_balance", NO_BALANCE_FEATURES, encoder),
    ]

    summary_rows = []
    for result in results:
        for split_name in ("validation_metrics", "test_metrics"):
            row = {
                "experiment": result["experiment"],
                "split": split_name,
            }
            row.update(result[split_name])
            summary_rows.append(row)

    pd.DataFrame(summary_rows).to_csv(
        OUT / "chronological_xgboost_summary.csv", index=False
    )
    print(f"\nAll results saved under: {OUT}")


if __name__ == "__main__":
    main()
