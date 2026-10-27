"""Train and compare four PaySim models with a strict-majority ensemble.

Run from the project root:
    python Experiments/compare_four_models.py

The experiment reads the existing chronological splits, samples training rows
deterministically, fits preprocessing on that sample only, chooses individual
model thresholds on validation data, and evaluates once on the later test data.
It refuses to overwrite its output directory.
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPLIT_DIR = PROJECT_ROOT / "Dataset (PaySim)" / "chronological_splits"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "Experiments" / "artifacts" / "four_model_comparison"

NUMERIC_FEATURES = (
    "step",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
)
CATEGORICAL_FEATURES = ("type",)
TARGET_COLUMN = "isFraud"
INPUT_COLUMNS = list(NUMERIC_FEATURES + CATEGORICAL_FEATURES)
MODEL_NAMES = (
    "Random Forest",
    "XGBoost",
    "Logistic Regression",
    "Isolation Forest",
)
MODEL_FILES = {
    "Random Forest": "random_forest.joblib",
    "XGBoost": "xgboost.joblib",
    "Logistic Regression": "logistic_regression.joblib",
    "Isolation Forest": "isolation_forest.joblib",
}
CHUNK_SIZE = 100_000
SEED = 42


def fit_preprocessing(training_frame):
    """Fit numeric scaling and category encoding using training rows only."""
    scaler = StandardScaler()
    scaler.fit(training_frame[list(NUMERIC_FEATURES)])

    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
        dtype=np.float32,
    )
    encoder.fit(training_frame[list(CATEGORICAL_FEATURES)])
    return scaler, encoder


def transform_features(frame, scaler, encoder):
    numeric = scaler.transform(frame[list(NUMERIC_FEATURES)]).astype(
        np.float32, copy=False
    )
    categorical = encoder.transform(frame[list(CATEGORICAL_FEATURES)]).astype(
        np.float32, copy=False
    )
    return np.concatenate((numeric, categorical), axis=1)


def majority_vote(binary_predictions):
    """Return strict-majority predictions, vote counts, and 2-2 tie mask.

    Input rows are transactions and the four columns are model votes. A fraud
    result requires at least three fraud votes. A 2-2 tie is resolved as
    legitimate, so a tie never counts as a fraud majority.
    """
    votes = np.asarray(binary_predictions)
    if votes.ndim != 2 or votes.shape[1] != 4:
        raise ValueError("Expected a 2D array with exactly four model votes per row.")
    if not np.isin(votes, (0, 1)).all():
        raise ValueError("Model votes must contain only binary values 0 and 1.")

    vote_counts = votes.sum(axis=1).astype(np.int8)
    ties = vote_counts == 2
    predictions = (vote_counts >= 3).astype(np.int8)
    return predictions, vote_counts, ties


def build_models(positive_weight, n_estimators=300):
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=min(n_estimators, 100),
            max_depth=20,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=SEED,
            n_jobs=-1,
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=1.0,
            objective="binary:logistic",
            eval_metric="aucpr",
            scale_pos_weight=positive_weight,
            tree_method="hist",
            random_state=SEED,
            n_jobs=-1,
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=500,
            class_weight="balanced",
            solver="lbfgs",
            random_state=SEED,
        ),
        "Isolation Forest": IsolationForest(
            n_estimators=min(n_estimators, 100),
            max_samples=256,
            contamination="auto",
            random_state=SEED,
            n_jobs=-1,
        ),
    }


def _sample_training_rows(train_path, fraction, seed=SEED, chunksize=CHUNK_SIZE):
    chunks = []
    required = INPUT_COLUMNS + [TARGET_COLUMN]
    reader = pd.read_csv(train_path, usecols=required, chunksize=chunksize)
    for chunk_number, chunk in enumerate(reader):
        chunks.append(
            chunk.sample(frac=fraction, random_state=seed + chunk_number)
        )
    if not chunks:
        raise ValueError(f"Training split is empty: {train_path}")
    return pd.concat(chunks, ignore_index=True)


def _predict_scores(models, features):
    scores = {}
    for name in MODEL_NAMES:
        model = models[name]
        if name == "Isolation Forest":
            scores[name] = -model.score_samples(features)
        else:
            fraud_class_index = int(np.flatnonzero(model.classes_ == 1)[0])
            scores[name] = model.predict_proba(features)[:, fraud_class_index]
    return scores


def _score_split(path, scaler, encoder, models, chunksize=CHUNK_SIZE):
    labels = []
    score_parts = {name: [] for name in MODEL_NAMES}
    reader = pd.read_csv(
        path,
        usecols=INPUT_COLUMNS + [TARGET_COLUMN],
        chunksize=chunksize,
    )
    for chunk in reader:
        features = transform_features(chunk, scaler, encoder)
        labels.append(chunk[TARGET_COLUMN].to_numpy(dtype=np.int8))
        for name, values in _predict_scores(models, features).items():
            score_parts[name].append(values)

    if not labels:
        raise ValueError(f"Evaluation split is empty: {path}")
    return (
        np.concatenate(labels),
        {name: np.concatenate(parts) for name, parts in score_parts.items()},
    )


def select_f1_threshold(labels, scores):
    """Select an F1-maximizing threshold, breaking ties by precision."""
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    if thresholds.size == 0:
        return float(scores[0])
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(
        precision[:-1] + recall[:-1], np.finfo(float).eps
    )
    best_index = max(
        range(len(thresholds)),
        key=lambda index: (f1[index], precision[index], thresholds[index]),
    )
    return float(thresholds[best_index])


def _classification_metrics(labels, predictions, scores):
    true_positive = int(np.sum((labels == 1) & (predictions == 1)))
    false_positive = int(np.sum((labels == 0) & (predictions == 1)))
    false_negative = int(np.sum((labels == 1) & (predictions == 0)))
    true_negative = int(np.sum((labels == 0) & (predictions == 0)))
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "average_precision": float(average_precision_score(labels, scores)),
        "confusion_matrix": [[true_negative, false_positive], [false_negative, true_positive]],
    }


def load_artifacts(artifact_dir):
    """Load a saved comparison bundle and reject incompatible feature schemas."""
    artifact_dir = Path(artifact_dir)
    with (artifact_dir / "config.json").open(encoding="utf-8") as file:
        config = json.load(file)
    if config.get("numeric_features") != list(NUMERIC_FEATURES):
        raise ValueError("Saved numeric feature schema is incompatible.")
    if config.get("categorical_features") != list(CATEGORICAL_FEATURES):
        raise ValueError("Saved categorical feature schema is incompatible.")

    scaler = joblib.load(artifact_dir / "numeric_scaler.joblib")
    encoder = joblib.load(artifact_dir / "type_encoder.joblib")
    models = {
        name: joblib.load(artifact_dir / filename)
        for name, filename in MODEL_FILES.items()
    }
    expected_feature_count = len(NUMERIC_FEATURES) + sum(
        len(categories) for categories in encoder.categories_
    )
    if scaler.n_features_in_ != len(NUMERIC_FEATURES):
        raise ValueError("Saved scaler feature count is incompatible.")
    if any(model.n_features_in_ != expected_feature_count for model in models.values()):
        raise ValueError("A saved model feature count is incompatible.")
    return {"scaler": scaler, "encoder": encoder, "models": models, "config": config}


def predict_scores(frame, bundle):
    features = transform_features(frame, bundle["scaler"], bundle["encoder"])
    return _predict_scores(bundle["models"], features)


def _save_results(output_dir, scaler, encoder, models, config, results, test_details):
    output_dir.mkdir(parents=True, exist_ok=False)
    joblib.dump(scaler, output_dir / "numeric_scaler.joblib")
    joblib.dump(encoder, output_dir / "type_encoder.joblib")
    for name, filename in MODEL_FILES.items():
        joblib.dump(models[name], output_dir / filename)

    with (output_dir / "config.json").open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)
    with (output_dir / "results.json").open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    rows = []
    for name, metrics in results["test_metrics"].items():
        rows.append({"model": name, **metrics})
    pd.DataFrame(rows).to_csv(output_dir / "comparison.csv", index=False)
    (output_dir / "report.md").write_text(
        _build_report(results, test_details), encoding="utf-8"
    )


def _build_report(results, test_details):
    lines = [
        "# Four-Model PaySim Comparison",
        "",
        "## Method",
        "",
        "This experiment uses the existing chronological PaySim splits. Training uses a deterministic 10% row sample from steps 1-450; validation is steps 451-600; test is steps 601-743. The test split is evaluated only after model and threshold selection.",
        "",
        "A StandardScaler and OneHotEncoder are fit only on the sampled training rows. No oversampling is used. Random Forest and Logistic Regression use class weighting; XGBoost uses the sampled negative-to-positive ratio; Isolation Forest is fit on non-fraud training rows only.",
        "",
        "Individual decision thresholds maximize validation F1, then precision, with higher threshold as a final deterministic tie-break. Average precision uses continuous probability scores for supervised models and anomaly scores for Isolation Forest.",
        "",
        "The ensemble uses the four thresholded binary predictions: at least 3 fraud votes means fraud. A 2-2 tie is resolved as legitimate. This is strict majority voting, not score averaging. Ensemble average precision is computed from the discrete fraud-vote count (0-4).",
        "",
        "## Test Results",
        "",
        "Confusion matrices use `[[TN, FP], [FN, TP]]`.",
        "",
        "| Model | Threshold | Precision | Recall | F1 | Average precision | Confusion matrix |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for name, metrics in results["test_metrics"].items():
        threshold = metrics.get("threshold", "strict majority (3/4)")
        threshold_text = f"{threshold:.6f}" if isinstance(threshold, float) else threshold
        cm = metrics["confusion_matrix"]
        lines.append(
            f"| {name} | {threshold_text} | {metrics['precision']:.4f} | {metrics['recall']:.4f} | {metrics['f1']:.4f} | {metrics['average_precision']:.4f} | `{cm}` |"
        )
    lines.extend(
        [
            "",
            f"Test rows: {test_details['rows']:,}; fraud rows: {test_details['fraud_rows']:,}. Ensemble 2-2 ties: {test_details['ensemble_tie_rows']:,}.",
            "",
            "## Limitations",
            "",
            "PaySim is synthetic, and its fraud labels reflect the simulator rather than independently verified real-world outcomes. These results do not establish performance on UPI or other live payment data. The deterministic 10% training sample reduces compute and may add sampling variability. Thresholds are validation-F1 choices, not calibrated risk cutoffs. Isolation Forest produces anomaly rankings rather than fraud probabilities, and the ensemble AP is coarse because it ranks transactions by only five possible vote counts. Model comparisons remain sensitive to each estimator's inductive bias and the shared feature representation.",
            "",
            "This experiment writes only to its new artifact directory; it does not change the live application's model or frontend.",
            "",
        ]
    )
    return "\n".join(lines)


def run_experiment(
    split_dir=SPLIT_DIR,
    output_dir=DEFAULT_OUTPUT_DIR,
    sample_fraction=0.10,
    chunksize=CHUNK_SIZE,
    n_estimators=300,
):
    split_dir = Path(split_dir)
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing experiment output: {output_dir}"
        )
    split_paths = {
        name: split_dir / f"{name}.csv"
        for name in ("train", "validation", "test")
    }
    missing = [str(path) for path in split_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing chronological split(s): " + ", ".join(missing))
    if not 0 < sample_fraction <= 1:
        raise ValueError("sample_fraction must be in (0, 1].")

    print("Sampling chronological training rows...")
    training = _sample_training_rows(
        split_paths["train"], sample_fraction, chunksize=chunksize
    )
    y_train = training[TARGET_COLUMN].to_numpy(dtype=np.int8)
    if np.unique(y_train).size != 2:
        raise ValueError("Sampled training rows must contain both fraud classes.")

    scaler, encoder = fit_preprocessing(training)
    X_train = transform_features(training, scaler, encoder)
    positives = int(y_train.sum())
    negatives = len(y_train) - positives
    models = build_models(negatives / positives, n_estimators=n_estimators)

    print(f"Training rows: {len(training):,}; fraud rows: {positives:,}")
    models["Random Forest"].fit(X_train, y_train)
    models["XGBoost"].fit(X_train, y_train)
    models["Logistic Regression"].fit(X_train, y_train)
    models["Isolation Forest"].fit(X_train[y_train == 0])
    del X_train, training

    print("Selecting individual thresholds on chronological validation data...")
    validation_labels, validation_scores = _score_split(
        split_paths["validation"], scaler, encoder, models, chunksize
    )
    thresholds = {
        name: select_f1_threshold(validation_labels, validation_scores[name])
        for name in MODEL_NAMES
    }

    print("Evaluating once on the later chronological test data...")
    test_labels, test_scores = _score_split(
        split_paths["test"], scaler, encoder, models, chunksize
    )
    test_metrics = {}
    binary_test_predictions = []
    for name in MODEL_NAMES:
        predictions = (test_scores[name] >= thresholds[name]).astype(np.int8)
        metrics = _classification_metrics(
            test_labels, predictions, test_scores[name]
        )
        metrics["threshold"] = thresholds[name]
        test_metrics[name] = metrics
        binary_test_predictions.append(predictions)

    ensemble_predictions, vote_counts, tie_mask = majority_vote(
        np.column_stack(binary_test_predictions)
    )
    test_metrics["Strict Majority Ensemble"] = _classification_metrics(
        test_labels, ensemble_predictions, vote_counts
    )
    test_details = {
        "rows": int(len(test_labels)),
        "fraud_rows": int(test_labels.sum()),
        "ensemble_tie_rows": int(tie_mask.sum()),
    }

    config = {
        "dataset": "PaySim synthetic transaction simulator",
        "split_directory": str(split_dir.relative_to(PROJECT_ROOT))
        if split_dir.is_relative_to(PROJECT_ROOT)
        else str(split_dir),
        "split_step_ranges": {
            "train": [1, 450],
            "validation": [451, 600],
            "test": [601, 743],
        },
        "numeric_features": list(NUMERIC_FEATURES),
        "categorical_features": list(CATEGORICAL_FEATURES),
        "target_column": TARGET_COLUMN,
        "excluded_inputs": ["isFraud", "isFlaggedFraud", "nameOrig", "nameDest"],
        "training_sample_fraction": sample_fraction,
        "training_sample_rows": int(len(y_train)),
        "training_sample_fraud_rows": positives,
        "validation_selected_thresholds": thresholds,
        "threshold_selection": "maximize validation F1, then precision, then threshold",
        "preprocessing_fit_scope": "sampled training rows only",
        "resampling": "none; imbalance handled with model class weights / positive weight",
        "isolation_forest_fit_scope": "non-fraud sampled training rows only",
        "ensemble_rule": "fraud with at least 3 of 4 binary fraud votes; 2-2 tie resolves as legitimate",
        "random_seed": SEED,
        "chunk_size": chunksize,
        "estimators": n_estimators,
    }
    results = {
        "config": config,
        "test_details": test_details,
        "test_metrics": test_metrics,
        "validation_metrics_at_selected_thresholds": {
            name: _classification_metrics(
                validation_labels,
                (validation_scores[name] >= thresholds[name]).astype(np.int8),
                validation_scores[name],
            )
            for name in MODEL_NAMES
        },
    }
    _save_results(output_dir, scaler, encoder, models, config, results, test_details)

    print("\nTest results [confusion matrix [[TN, FP], [FN, TP]]]:")
    for name, metrics in test_metrics.items():
        print(
            f"{name}: precision={metrics['precision']:.4f}, "
            f"recall={metrics['recall']:.4f}, f1={metrics['f1']:.4f}, "
            f"AP={metrics['average_precision']:.4f}, "
            f"confusion={metrics['confusion_matrix']}"
        )
    print(f"Ensemble tie rows: {test_details['ensemble_tie_rows']:,}")
    print(f"New artifacts and report: {output_dir}")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-fraction", type=float, default=0.10)
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--estimators", type=int, default=300)
    args = parser.parse_args()
    run_experiment(
        output_dir=args.output_dir,
        sample_fraction=args.sample_fraction,
        chunksize=args.chunk_size,
        n_estimators=args.estimators,
    )


if __name__ == "__main__":
    main()