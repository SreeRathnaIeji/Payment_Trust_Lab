"""Train an XGBoost candidate with past-only transaction graph features.

This is a graph-feature baseline, not a graph neural network. It computes
historical sender/recipient degrees using transactions from earlier PaySim
steps only. All transactions in an hour receive features before any edge from
that hour updates the graph state, preventing within-step look-ahead.

Run from the project root:
    python Experiments/train_temporal_graph_features.py
Smoke-test temporal state logic only:
    python Experiments/train_temporal_graph_features.py --self-test
"""

from __future__ import annotations

import argparse
import zlib
import json
from pathlib import Path

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


ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "Dataset (PaySim)" / "chronological_splits"
BASELINE_DIR = ROOT / "Experiments" / "artifacts" / "chronological" / "xgboost"
OUT = ROOT / "Experiments" / "artifacts" / "temporal_graph_xgboost"
MODEL_VERSION = "temporal-graph-xgboost-v1"
NUMERIC = ["step", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]
GRAPH_NAMES = [
    "log_source_prior_out_degree",
    "log_source_prior_in_degree",
    "log_target_prior_in_degree",
    "log_target_prior_out_degree",
]
SKETCH_WIDTH = 1 << 22
SKETCH_SEEDS = (0x13579BDF, 0x2468ACE1, 0x5A17C9E3)
USECOLS = NUMERIC + ["type", "nameOrig", "nameDest", "isFraud"]
CHUNK_SIZE = 100_000


class TemporalGraph:
    def __init__(self):
        # Fixed-size sketches bound memory independently of account cardinality.
        self.out_degree = np.zeros((len(SKETCH_SEEDS), SKETCH_WIDTH), dtype=np.uint32)
        self.in_degree = np.zeros((len(SKETCH_SEEDS), SKETCH_WIDTH), dtype=np.uint32)

    @staticmethod
    def _indices(value: str) -> tuple[int, ...]:
        encoded = value.encode("utf-8")
        return tuple(zlib.crc32(encoded, seed) & (SKETCH_WIDTH - 1) for seed in SKETCH_SEEDS)

    @classmethod
    def estimate(cls, sketch: np.ndarray, value: str) -> int:
        return min(int(sketch[row, column]) for row, column in enumerate(cls._indices(value)))

    @classmethod
    def _add_counts(cls, sketch: np.ndarray, counts: dict) -> None:
        for value, count in counts.items():
            for row, column in enumerate(cls._indices(str(value))):
                sketch[row, column] += np.uint32(count)

    def feature_matrix(self, frame: pd.DataFrame, encoder) -> np.ndarray:
        source = frame["nameOrig"].astype(str).to_numpy()
        target = frame["nameDest"].astype(str).to_numpy()
        graph_values = np.column_stack(
            [
                np.log1p(np.fromiter((self.estimate(self.out_degree, key) for key in source), dtype=np.float32, count=len(source))),
                np.log1p(np.fromiter((self.estimate(self.in_degree, key) for key in source), dtype=np.float32, count=len(source))),
                np.log1p(np.fromiter((self.estimate(self.in_degree, key) for key in target), dtype=np.float32, count=len(target))),
                np.log1p(np.fromiter((self.estimate(self.out_degree, key) for key in target), dtype=np.float32, count=len(target))),
            ]
        ).astype(np.float32, copy=False)
        numeric = frame[NUMERIC].to_numpy(dtype=np.float32, copy=True)
        categorical = encoder.transform(frame[["type"]]).astype(np.float32)
        base = np.concatenate((numeric, categorical), axis=1)
        return np.concatenate((base, graph_values), axis=1)

    def observe_step(self, frame: pd.DataFrame) -> None:
        # Labels are intentionally never read here.
        self._add_counts(self.out_degree, frame["nameOrig"].value_counts().to_dict())
        self._add_counts(self.in_degree, frame["nameDest"].value_counts().to_dict())


def score(y_true, probabilities, threshold):
    predicted = (probabilities >= threshold).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "average_precision": float(average_precision_score(y_true, probabilities)),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }


def run_split(path: Path, graph: TemporalGraph, encoder, *, training: bool):
    """Stream a chronological CSV; hold each step intact before updating graph."""
    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    pending = pd.DataFrame()
    previous_step = None
    previous_chunk_last = None

    def consume(frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        if not frame["step"].is_monotonic_increasing:
            raise ValueError(f"Non-chronological rows found in {path}")
        for _, step_frame in frame.groupby("step", sort=False):
            x_all = graph.feature_matrix(step_frame, encoder)
            y_all = step_frame["isFraud"].to_numpy(dtype=np.int8)
            if training:
                selected = step_frame["__sample"].to_numpy(dtype=bool)
                if selected.any():
                    x_parts.append(x_all[selected])
                    y_parts.append(y_all[selected])
            else:
                x_parts.append(x_all)
                y_parts.append(y_all)
            graph.observe_step(step_frame)

    reader = pd.read_csv(path, usecols=USECOLS, chunksize=CHUNK_SIZE)
    for chunk_index, chunk in enumerate(reader):
        if previous_chunk_last is not None and int(chunk["step"].iloc[0]) < previous_chunk_last:
            raise ValueError(f"Step order moves backwards in {path}")
        previous_chunk_last = int(chunk["step"].iloc[-1])
        if training:
            chunk["__sample"] = False
            selected_indices = chunk.sample(frac=0.10, random_state=42 + chunk_index).index
            chunk.loc[selected_indices, "__sample"] = True
        combined = pd.concat((pending, chunk), ignore_index=True)
        last_step = combined["step"].iloc[-1]
        complete = combined.loc[combined["step"] != last_step]
        pending = combined.loc[combined["step"] == last_step].copy()
        consume(complete)
    consume(pending)
    if not x_parts:
        raise ValueError(f"No rows were prepared from {path}")
    return np.concatenate(x_parts), np.concatenate(y_parts)


def self_test() -> None:
    class Encoder:
        def transform(self, frame):
            return np.zeros((len(frame), 5), dtype=np.float32)

    graph = TemporalGraph()
    columns = NUMERIC + ["type", "nameOrig", "nameDest", "isFraud"]
    first = pd.DataFrame(
        [[1, 2, 3, 1, 0, 4, "TRANSFER", "A", "B", 1],
         [1, 5, 6, 1, 0, 7, "TRANSFER", "A", "B", 0]],
        columns=columns,
    )
    first_features = graph.feature_matrix(first, Encoder())
    assert np.all(first_features[:, -4:] == 0), "Same-step transactions must not see each other."
    graph.observe_step(first)
    second = pd.DataFrame(
        [[2, 8, 9, 1, 0, 10, "TRANSFER", "A", "C", 1]], columns=columns
    )
    second_features = graph.feature_matrix(second, Encoder())
    assert np.allclose(second_features[0, -4:], [np.log1p(2), 0, 0, 0])
    graph.observe_step(second)
    print("Temporal graph smoke test passed: no same-step or label leakage.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return

    OUT.mkdir(parents=True, exist_ok=True)
    encoder = joblib.load(BASELINE_DIR / "type_encoder.joblib")
    graph = TemporalGraph()
    print("Preparing past-only graph features for chronological training rows...")
    x_train, y_train = run_split(SPLITS / "train.csv", graph, encoder, training=True)
    positives = int(y_train.sum())
    negatives = len(y_train) - positives
    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=8, learning_rate=0.1, subsample=0.8,
        colsample_bytree=1.0, objective="binary:logistic", eval_metric="aucpr",
        scale_pos_weight=negatives / positives, tree_method="hist", random_state=42, n_jobs=-1,
    )
    model.fit(x_train, y_train)
    del x_train, y_train
    joblib.dump(model, OUT / "temporal_graph_xgboost.joblib")

    print("Preparing chronological validation features...")
    x_val, y_val = run_split(SPLITS / "validation.csv", graph, encoder, training=False)
    val_probabilities = model.predict_proba(x_val)[:, 1]
    thresholds = list(np.arange(0.01, 1.00, 0.01)) + [0.95, 0.995, 0.999]
    candidates = [score(y_val, val_probabilities, value) for value in sorted(set(thresholds))]
    best = max(candidates, key=lambda row: (row["f1"], row["precision"]))
    pd.DataFrame(candidates).to_csv(OUT / "validation_thresholds.csv", index=False)
    del x_val, y_val, val_probabilities

    print("Preparing chronological test features...")
    x_test, y_test = run_split(SPLITS / "test.csv", graph, encoder, training=False)
    test_probabilities = model.predict_proba(x_test)[:, 1]
    test_result = score(y_test, test_probabilities, best["threshold"])
    baseline_path = BASELINE_DIR / "main_features_results.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))["test_metrics"]
    result = {
        "experiment": MODEL_VERSION,
        "model_type": "XGBoost with past-only temporal graph degree features; not a GNN",
        "graph_features": GRAPH_NAMES,
        "graph_count_estimator": {
            "method": "Count-Min Sketch",
            "width_per_row": SKETCH_WIDTH,
            "depth": len(SKETCH_SEEDS),
            "note": "Fixed-memory approximate degree counts may overestimate due to hash collisions.",
        },
        "label_used_as_input": False,
        "within_step_future_edges_excluded": True,
        "training_sample_fraction": 0.10,
        "validation_selected_threshold": best["threshold"],
        "validation_metrics": best,
        "test_metrics": test_result,
        "existing_main_xgboost_test_metrics": baseline,
        "comparison_period": "Dataset (PaySim)/chronological_splits/test.csv",
        "model_file": str((OUT / "temporal_graph_xgboost.joblib").relative_to(ROOT)),
    }
    (OUT / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved new experiment artifacts under {OUT}")


if __name__ == "__main__":
    main()
