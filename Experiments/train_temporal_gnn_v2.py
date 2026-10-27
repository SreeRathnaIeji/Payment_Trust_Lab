"""Run controlled node-only and temporal-graph GNN PaySim experiments.

Run from the project root:
    python Experiments/train_temporal_gnn_v2.py

This version writes only to Experiments/artifacts/gnn_experiment_v2/. It stages
expanded graph tensors as temporary memory-mapped files so training uses
bounded batches instead of loading all neighbors into RAM.
"""

from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
import json
import math
import os
import random
import tempfile
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "Dataset (PaySim)" / "chronological_splits"
BASELINE_RESULTS = (
    ROOT
    / "Experiments"
    / "artifacts"
    / "chronological"
    / "xgboost"
    / "main_features_results.json"
)
OUT = ROOT / "Experiments" / "artifacts" / "gnn_experiment_v2"
NUMERIC_FEATURES = (
    "step",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
)
CATEGORICAL_FEATURES = ("type",)
INPUT_COLUMNS = list(NUMERIC_FEATURES + CATEGORICAL_FEATURES)
GRAPH_COLUMNS = ("nameOrig", "nameDest")
TARGET_COLUMN = "isFraud"
USECOLS = INPUT_COLUMNS + list(GRAPH_COLUMNS) + [TARGET_COLUMN]
RELATIONS = (
    ("sender", "sender"),
    ("recipient", "sender"),
    ("sender", "recipient"),
    ("recipient", "recipient"),
)
EDGE_FEATURE_COUNT = len(RELATIONS) + 1
MAX_STEP_GAP_SCALE = math.log1p(743)
CHUNK_SIZE = 100_000
SEED = 42
TARGET_SAMPLE_FRACTION = 0.10
CONTEXT_SAMPLE_FRACTION = 0.25
RECENT_NEIGHBORS_PER_RELATION = 4
HIDDEN_DIM = 32
BATCH_SIZE = 2048
MAX_EPOCHS = 20
EARLY_STOPPING_PATIENCE = 3


def set_determinism(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError):
        pass


def _sample_mask(frame, fraction, random_state):
    if fraction >= 1:
        return np.ones(len(frame), dtype=bool)
    selected = frame.sample(frac=fraction, random_state=random_state).index
    return frame.index.isin(selected)


def fit_preprocessing(train_path, sample_fraction=TARGET_SAMPLE_FRACTION):
    numeric_parts = []
    type_parts = []
    reader = pd.read_csv(train_path, usecols=INPUT_COLUMNS, chunksize=CHUNK_SIZE)
    for chunk_index, chunk in enumerate(reader):
        selected = _sample_mask(chunk, sample_fraction, SEED + chunk_index)
        sample = chunk.loc[selected]
        if len(sample):
            numeric_parts.append(sample[list(NUMERIC_FEATURES)])
            type_parts.append(sample[list(CATEGORICAL_FEATURES)])
    if not numeric_parts:
        raise ValueError("No training rows selected for preprocessing.")
    numeric = pd.concat(numeric_parts, ignore_index=True)
    categories = pd.concat(type_parts, ignore_index=True)
    scaler = StandardScaler().fit(numeric)
    encoder = OneHotEncoder(
        handle_unknown="ignore", sparse_output=False, dtype=np.float32
    ).fit(categories)
    return scaler, encoder, len(numeric)


def transform_nodes(frame, scaler, encoder):
    numeric = scaler.transform(frame[list(NUMERIC_FEATURES)]).astype(
        np.float32, copy=False
    )
    categorical = encoder.transform(frame[list(CATEGORICAL_FEATURES)]).astype(
        np.float32, copy=False
    )
    return np.concatenate((numeric, categorical), axis=1)


class TemporalTransactionGraph:
    """Transaction nodes joined to bounded recent account-role histories."""

    def __init__(self, feature_count, neighbors_per_relation=RECENT_NEIGHBORS_PER_RELATION):
        self.feature_count = int(feature_count)
        self.neighbors_per_relation = int(neighbors_per_relation)
        if self.neighbors_per_relation < 1:
            raise ValueError("neighbors_per_relation must be at least one.")
        self.edge_count = len(RELATIONS) * self.neighbors_per_relation
        self.capacity = 1024
        self.size = 0
        self.node_features = np.zeros(
            (self.capacity, self.feature_count), dtype=np.float32
        )
        self.node_steps = np.full(self.capacity, -1, dtype=np.int32)
        self.source_history = {}
        self.recipient_history = {}

    def _ensure_capacity(self, additional):
        required = self.size + additional
        if required <= self.capacity:
            return
        capacity = self.capacity
        while capacity < required:
            capacity *= 2
        features = np.zeros((capacity, self.feature_count), dtype=np.float32)
        steps = np.full(capacity, -1, dtype=np.int32)
        features[: self.size] = self.node_features[: self.size]
        steps[: self.size] = self.node_steps[: self.size]
        self.node_features = features
        self.node_steps = steps
        self.capacity = capacity

    def process_step(self, frame, node_features, context_mask, target_mask):
        if frame.empty or frame["step"].nunique() != 1:
            raise ValueError("process_step requires rows from exactly one nonempty step.")
        if node_features.shape != (len(frame), self.feature_count):
            raise ValueError("Node feature matrix has an incompatible shape.")
        context_mask = np.asarray(context_mask, dtype=bool)
        target_mask = np.asarray(target_mask, dtype=bool)
        if context_mask.shape != (len(frame),) or target_mask.shape != (len(frame),):
            raise ValueError("Masks must contain one value for every transaction row.")

        step = int(frame["step"].iloc[0])
        sources = frame["nameOrig"].astype(str).to_numpy()
        recipients = frame["nameDest"].astype(str).to_numpy()
        neighbors = np.zeros(
            (len(frame), self.edge_count, self.feature_count), dtype=np.float32
        )
        edge_features = np.zeros(
            (len(frame), self.edge_count, EDGE_FEATURE_COUNT), dtype=np.float32
        )
        neighbor_mask = np.zeros((len(frame), self.edge_count), dtype=np.float32)

        histories = (self.source_history, self.recipient_history)
        for row, (source, recipient) in enumerate(zip(sources, recipients)):
            accounts = (source, source, recipient, recipient)
            for relation_index, ((prior_role, _), account) in enumerate(
                zip(RELATIONS, accounts)
            ):
                history = histories[0 if prior_role == "sender" else 1]
                prior_ids = history.get(account, ())
                recent_ids = list(prior_ids)[-self.neighbors_per_relation :]
                offset = relation_index * self.neighbors_per_relation
                for rank, prior_id in enumerate(reversed(recent_ids)):
                    prior_step = int(self.node_steps[prior_id])
                    if prior_step >= step:
                        continue
                    edge_index = offset + rank
                    neighbors[row, edge_index] = self.node_features[prior_id]
                    edge_features[row, edge_index, relation_index] = 1.0
                    edge_features[row, edge_index, -1] = (
                        np.log1p(step - prior_step) / MAX_STEP_GAP_SCALE
                    )
                    neighbor_mask[row, edge_index] = 1.0

        model_inputs = (
            node_features[target_mask],
            neighbors[target_mask],
            edge_features[target_mask],
            neighbor_mask[target_mask],
        )
        labels = frame.loc[target_mask, TARGET_COLUMN].to_numpy(dtype=np.int8)

        selected_rows = np.flatnonzero(context_mask)
        self._ensure_capacity(len(selected_rows))
        node_ids = np.arange(self.size, self.size + len(selected_rows))
        if len(selected_rows):
            self.node_features[node_ids] = node_features[selected_rows]
            self.node_steps[node_ids] = step
            self.size += len(selected_rows)
            row_ids = frame["__row_id"].to_numpy(dtype=np.int64)
            updates = {}
            for row_index, node_id in zip(selected_rows, node_ids):
                row_id = int(row_ids[row_index])
                for history, account in (
                    (self.source_history, sources[row_index]),
                    (self.recipient_history, recipients[row_index]),
                ):
                    key = (id(history), str(account))
                    updates.setdefault(key, []).append((row_id, int(node_id)))
            for (history_id, account), additions in updates.items():
                history = self.source_history if history_id == id(self.source_history) else self.recipient_history
                prior = list(history.get(account, ()))
                prior.extend(node_id for _, node_id in sorted(additions))
                history[account] = prior[-self.neighbors_per_relation :]
        return model_inputs, labels


def _iter_steps(path, target_fraction, context_fraction, split_seed):
    pending = pd.DataFrame()
    previous_last_step = None
    row_offset = 0
    reader = pd.read_csv(path, usecols=USECOLS, chunksize=CHUNK_SIZE)
    for chunk_index, chunk in enumerate(reader):
        if not chunk["step"].is_monotonic_increasing:
            raise ValueError(f"Rows are not chronological in {path}.")
        first_step = int(chunk["step"].iloc[0])
        if previous_last_step is not None and first_step < previous_last_step:
            raise ValueError(f"Step order moves backwards in {path}.")
        previous_last_step = int(chunk["step"].iloc[-1])
        chunk["__row_id"] = np.arange(row_offset, row_offset + len(chunk), dtype=np.int64)
        row_offset += len(chunk)
        chunk["__context"] = _sample_mask(
            chunk, context_fraction, SEED + split_seed + chunk_index
        )
        chunk["__target"] = _sample_mask(
            chunk, target_fraction, SEED + 500_000 + split_seed + chunk_index
        )
        combined = pd.concat((pending, chunk), ignore_index=True)
        final_step = combined["step"].iloc[-1]
        complete = combined.loc[combined["step"] != final_step]
        pending = combined.loc[combined["step"] == final_step].copy()
        for _, step_frame in complete.groupby("step", sort=False):
            yield step_frame
    if not pending.empty:
        yield pending


def count_targets(path, fraction, split_seed):
    total = 0
    reader = pd.read_csv(path, usecols=["step"], chunksize=CHUNK_SIZE)
    for chunk_index, chunk in enumerate(reader):
        total += int(
            _sample_mask(chunk, fraction, SEED + 500_000 + split_seed + chunk_index).sum()
        )
    return total


def _array_specs(count, feature_count, neighbors_per_relation, graph_enabled):
    specs = {
        "nodes": (count, feature_count),
        "labels": (count,),
    }
    if graph_enabled:
        edge_count = len(RELATIONS) * neighbors_per_relation
        specs.update(
            {
                "neighbors": (count, edge_count, feature_count),
                "edge_features": (count, edge_count, EDGE_FEATURE_COUNT),
                "neighbor_mask": (count, edge_count),
            }
        )
    return specs


def stage_split(
    path,
    split_name,
    staging_dir,
    graph,
    scaler,
    encoder,
    *,
    target_fraction,
    context_fraction,
    split_seed,
    graph_enabled,
):
    target_count = count_targets(path, target_fraction, split_seed)
    feature_count = graph.feature_count
    maps = {}
    for name, shape in _array_specs(
        target_count, feature_count, graph.neighbors_per_relation, graph_enabled
    ).items():
        maps[name] = np.lib.format.open_memmap(
            Path(staging_dir) / f"{split_name}_{name}.npy",
            mode="w+",
            dtype=np.int8 if name == "labels" else np.float32,
            shape=shape,
        )

    cursor = 0
    for frame in _iter_steps(path, target_fraction, context_fraction, split_seed):
        node_features = transform_nodes(frame, scaler, encoder)
        target_mask = frame["__target"].to_numpy(dtype=bool)
        context_mask = frame["__context"].to_numpy(dtype=bool)
        graph_inputs, labels = graph.process_step(
            frame, node_features, context_mask, target_mask
        )
        count = len(labels)
        if not count:
            continue
        end = cursor + count
        maps["nodes"][cursor:end] = graph_inputs[0]
        maps["labels"][cursor:end] = labels
        if graph_enabled:
            maps["neighbors"][cursor:end] = graph_inputs[1]
            maps["edge_features"][cursor:end] = graph_inputs[2]
            maps["neighbor_mask"][cursor:end] = graph_inputs[3]
        cursor = end
    if cursor != target_count:
        raise ValueError(
            f"Staged {cursor} {split_name} rows, but counted {target_count}."
        )
    edge_counts = (
        np.asarray(maps["neighbor_mask"]).sum(axis=1)
        if graph_enabled
        else np.zeros(target_count, dtype=np.float32)
    )
    for value in maps.values():
        value.flush()
    del maps
    return {
        "split": split_name,
        "rows": int(target_count),
        "fraud_rows": int(np.load(Path(staging_dir) / f"{split_name}_labels.npy", mmap_mode="r").sum()),
        "transactions_with_edges": int(np.sum(edge_counts > 0)),
        "edge_coverage": float(np.mean(edge_counts > 0)) if target_count else 0.0,
        "average_edges_per_transaction": float(edge_counts.mean()) if target_count else 0.0,
        "maximum_edges_observed": int(edge_counts.max()) if target_count else 0,
    }


class MemmapBatches(tf.keras.utils.PyDataset):
    def __init__(self, staging_dir, split_name, batch_size, graph_enabled, *, shuffle=False, seed=SEED, sample_weights=None):
        super().__init__(workers=1, use_multiprocessing=False, max_queue_size=1)
        self.staging_dir = Path(staging_dir)
        self.split_name = split_name
        self.batch_size = int(batch_size)
        self.graph_enabled = graph_enabled
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)
        self.labels = np.load(self.staging_dir / f"{split_name}_labels.npy", mmap_mode="r")
        self.nodes = np.load(self.staging_dir / f"{split_name}_nodes.npy", mmap_mode="r")
        self.neighbors = None
        self.edge_features = None
        self.neighbor_mask = None
        if graph_enabled:
            self.neighbors = np.load(self.staging_dir / f"{split_name}_neighbors.npy", mmap_mode="r")
            self.edge_features = np.load(self.staging_dir / f"{split_name}_edge_features.npy", mmap_mode="r")
            self.neighbor_mask = np.load(self.staging_dir / f"{split_name}_neighbor_mask.npy", mmap_mode="r")
        self.sample_weights = sample_weights
        self.indices = np.arange(len(self.labels), dtype=np.int64)
        self.on_epoch_end()

    def __len__(self):
        return math.ceil(len(self.labels) / self.batch_size)

    def __getitem__(self, batch_index):
        start = batch_index * self.batch_size
        selected = self.indices[start : start + self.batch_size]
        inputs = (self.nodes[selected],)
        if self.graph_enabled:
            inputs += (
                self.neighbors[selected],
                self.edge_features[selected],
                self.neighbor_mask[selected],
            )
        labels = np.asarray(self.labels[selected], dtype=np.float32)
        if self.sample_weights is not None:
            return inputs, labels, self.sample_weights[selected]
        return inputs, labels

    def close(self):
        for array in (
            self.labels,
            self.nodes,
            self.neighbors,
            self.edge_features,
            self.neighbor_mask,
        ):
            mapped_file = getattr(array, "_mmap", None)
            if mapped_file is not None:
                mapped_file.close()

    def on_epoch_end(self):
        if self.shuffle:
            self.rng.shuffle(self.indices)


class MaskedMeanAggregation(tf.keras.layers.Layer):
    def call(self, inputs):
        messages, mask = inputs
        mask = tf.expand_dims(mask, axis=-1)
        total = tf.reduce_sum(messages * mask, axis=1)
        count = tf.reduce_sum(mask, axis=1)
        return tf.math.divide_no_nan(total, count)


def build_model(feature_count, *, graph_enabled, neighbors_per_relation, hidden_dim=HIDDEN_DIM):
    edge_count = len(RELATIONS) * neighbors_per_relation
    node_input = tf.keras.Input(shape=(feature_count,), name="transaction_features")
    own = tf.keras.layers.Dense(hidden_dim, activation="relu", name="node_projection")(node_input)
    if graph_enabled:
        neighbor_input = tf.keras.Input(
            shape=(edge_count, feature_count), name="prior_transaction_features"
        )
        edge_input = tf.keras.Input(
            shape=(edge_count, EDGE_FEATURE_COUNT), name="typed_temporal_edges"
        )
        mask_input = tf.keras.Input(shape=(edge_count,), name="prior_edge_mask")
        messages = tf.keras.layers.Dense(
            hidden_dim, activation="relu", name="edge_message_projection"
        )(tf.keras.layers.Concatenate(axis=-1)([neighbor_input, edge_input]))
        aggregate = MaskedMeanAggregation(name="masked_mean_messages")(
            [messages, mask_input]
        )
        hidden = tf.keras.layers.Dense(hidden_dim, activation="relu", name="node_update")(
            tf.keras.layers.Concatenate()([own, aggregate])
        )
        inputs = [node_input, neighbor_input, edge_input, mask_input]
    else:
        hidden = own
        inputs = [node_input]
    hidden = tf.keras.layers.Dropout(0.2)(hidden)
    output = tf.keras.layers.Dense(1, activation="sigmoid", name="fraud_probability")(hidden)
    return tf.keras.Model(inputs=inputs, outputs=output)


def predict_memmap(model, dataset, batch_size):
    predictions = np.empty(len(dataset.labels), dtype=np.float32)
    for batch_index in range(len(dataset)):
        inputs, _ = dataset[batch_index]
        start = batch_index * dataset.batch_size
        end = start + len(inputs[0])
        predictions[start:end] = model(inputs, training=False).numpy().reshape(-1)
    return predictions


def select_f1_threshold(labels, probabilities):
    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    if not len(thresholds):
        return float(probabilities[0])
    f1_values = 2 * precision[:-1] * recall[:-1] / np.maximum(
        precision[:-1] + recall[:-1], np.finfo(float).eps
    )
    selected = max(
        range(len(thresholds)),
        key=lambda i: (f1_values[i], precision[i], thresholds[i]),
    )
    return float(thresholds[selected])


def evaluate(labels, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "average_precision": float(average_precision_score(labels, probabilities)),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


class ValidationAveragePrecisionEarlyStopping(tf.keras.callbacks.Callback):
    def __init__(self, validation_dataset, validation_labels, batch_size, patience):
        super().__init__()
        self.validation_dataset = validation_dataset
        self.validation_labels = validation_labels
        self.batch_size = batch_size
        self.patience = patience
        self.best_average_precision = -np.inf
        self.best_epoch = 0
        self.best_weights = None
        self.wait = 0
        self.epoch_history = []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs if logs is not None else {}
        probabilities = predict_memmap(
            self.model, self.validation_dataset, self.batch_size
        )
        validation_ap = float(
            average_precision_score(self.validation_labels, probabilities)
        )
        logs["val_average_precision"] = validation_ap
        record = {"epoch": int(epoch + 1)}
        for name, value in logs.items():
            if np.isscalar(value):
                record[name] = float(value)
        self.epoch_history.append(record)
        if validation_ap > self.best_average_precision:
            self.best_average_precision = validation_ap
            self.best_epoch = epoch + 1
            self.best_weights = [np.array(weight, copy=True) for weight in self.model.get_weights()]
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.model.stop_training = True

    def on_train_end(self, logs=None):
        if self.best_weights is None:
            raise RuntimeError("Validation AP callback did not observe a checkpoint.")
        self.model.set_weights(self.best_weights)


def select_best_epoch(epoch_history):
    if not epoch_history:
        raise ValueError("Epoch history cannot be empty.")
    return max(
        epoch_history,
        key=lambda entry: (entry["val_average_precision"], -entry["epoch"]),
    )["epoch"]


def _baseline_results():
    saved = json.loads(BASELINE_RESULTS.read_text(encoding="utf-8"))["test_metrics"]
    return {
        "name": "Existing chronological XGBoost baseline",
        "threshold": float(saved["threshold"]),
        "precision": float(saved["precision"]),
        "recall": float(saved["recall"]),
        "f1": float(saved["f1"]),
        "average_precision": float(saved["average_precision"]),
        "confusion_matrix": [
            [int(saved["TN"]), int(saved["FP"])],
            [int(saved["FN"]), int(saved["TP"])],
        ],
    }


def process_memory_mb():
    if os.name != "nt":
        return None

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    psapi = ctypes.WinDLL("Psapi.dll")
    kernel32 = ctypes.WinDLL("Kernel32.dll")
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(ProcessMemoryCounters),
        wintypes.DWORD,
    ]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    succeeded = psapi.GetProcessMemoryInfo(
        kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
    )
    return counters.PeakWorkingSetSize / (1024 * 1024) if succeeded else None


def _fit_variant(
    name,
    graph_enabled,
    staging_dir,
    scaler,
    encoder,
    config,
    variant_dir,
    *,
    batch_size,
    max_epochs,
    patience,
):
    train_labels = np.load(Path(staging_dir) / "train_labels.npy")
    validation_labels = np.load(Path(staging_dir) / "validation_labels.npy")
    test_labels = np.load(Path(staging_dir) / "test_labels.npy")
    positives = int(train_labels.sum())
    negatives = len(train_labels) - positives
    class_weights = {
        0: len(train_labels) / (2 * negatives),
        1: len(train_labels) / (2 * positives),
    }
    sample_weights = np.where(train_labels == 1, class_weights[1], class_weights[0]).astype(np.float32)
    training = MemmapBatches(
        staging_dir,
        "train",
        batch_size,
        graph_enabled,
        shuffle=True,
        seed=SEED,
        sample_weights=sample_weights,
    )
    validation = MemmapBatches(
        staging_dir, "validation", batch_size, graph_enabled, shuffle=False
    )
    test = MemmapBatches(staging_dir, "test", batch_size, graph_enabled, shuffle=False)
    feature_count = len(NUMERIC_FEATURES) + sum(
        len(categories) for categories in encoder.categories_
    )
    model = build_model(
        feature_count,
        graph_enabled=graph_enabled,
        neighbors_per_relation=config["neighbors_per_relation"],
        hidden_dim=config["hidden_dim"],
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.AUC(curve="PR", name="pr_auc")],
    )
    checkpoint = ValidationAveragePrecisionEarlyStopping(
        validation, validation_labels, batch_size, patience
    )
    started = time.perf_counter()
    try:
        model.fit(
            training,
            validation_data=validation,
            epochs=max_epochs,
            callbacks=[checkpoint],
            verbose=2,
        )
    except BaseException:
        training.close()
        validation.close()
        test.close()
        raise
    training_seconds = time.perf_counter() - started

    validation_probabilities = predict_memmap(model, validation, batch_size)
    threshold = select_f1_threshold(validation_labels, validation_probabilities)
    validation_metrics = evaluate(validation_labels, validation_probabilities, threshold)
    test_probabilities = predict_memmap(model, test, batch_size)
    test_metrics = evaluate(test_labels, test_probabilities, threshold)

    variant_dir.mkdir(parents=True, exist_ok=False)
    joblib.dump(scaler, variant_dir / "numeric_scaler.joblib")
    joblib.dump(encoder, variant_dir / "type_encoder.joblib")
    model.save_weights(variant_dir / "model.weights.h5")
    variant_config = {
        **config,
        "variant": name,
        "graph_enabled": graph_enabled,
        "selected_epoch": int(checkpoint.best_epoch),
        "selected_threshold": threshold,
        "validation_selected_metrics": validation_metrics,
        "class_weights": {str(key): float(value) for key, value in class_weights.items()},
    }
    results = {
        "training_seconds": training_seconds,
        "selected_epoch": int(checkpoint.best_epoch),
        "best_validation_average_precision": float(checkpoint.best_average_precision),
        "validation_metrics_at_selected_threshold": validation_metrics,
        "test_metrics": test_metrics,
        "epoch_history": checkpoint.epoch_history,
    }
    (variant_dir / "config.json").write_text(
        json.dumps(variant_config, indent=2), encoding="utf-8"
    )
    (variant_dir / "results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    training.close()
    validation.close()
    test.close()
    return results


def load_variant(variant_dir):
    variant_dir = Path(variant_dir)
    config = json.loads((variant_dir / "config.json").read_text(encoding="utf-8"))
    scaler = joblib.load(variant_dir / "numeric_scaler.joblib")
    encoder = joblib.load(variant_dir / "type_encoder.joblib")
    if config.get("numeric_features") != list(NUMERIC_FEATURES):
        raise ValueError("Saved numeric feature schema does not match.")
    feature_count = len(NUMERIC_FEATURES) + sum(
        len(categories) for categories in encoder.categories_
    )
    if scaler.n_features_in_ != len(NUMERIC_FEATURES):
        raise ValueError("Saved scaler feature count does not match.")
    model = build_model(
        feature_count,
        graph_enabled=bool(config["graph_enabled"]),
        neighbors_per_relation=config["neighbors_per_relation"],
        hidden_dim=config["hidden_dim"],
    )
    model.load_weights(variant_dir / "model.weights.h5")
    return {"model": model, "scaler": scaler, "encoder": encoder, "config": config}


def write_missing_report(output_dir=OUT):
    output_dir = Path(output_dir)
    report_path = output_dir / "report.md"
    if report_path.exists():
        raise FileExistsError(f"Refusing to overwrite report: {report_path}")
    results_path = output_dir / "results.json"
    if not results_path.is_file():
        raise FileNotFoundError(f"Completed results not found: {results_path}")
    results = json.loads(results_path.read_text(encoding="utf-8"))
    report_path.open("x", encoding="utf-8").write(_report(results))
    return report_path


def _report(results):
    lines = [
        "# Temporal GNN v2: Controlled Ablations",
        "",
        "## Protocol",
        "",
        "Both variants use the same chronological PaySim train/validation/test periods, sampled training targets, train-only preprocessing, class weights, optimizer, hidden dimension, batch size, early-stopping policy, and validation-only F1 threshold selection. The baseline is the existing saved chronological XGBoost result on the same test split; it was not retrained.",
        "",
        "The graph variant uses transaction nodes and directed edges from strictly earlier sampled transactions sharing one of four sender/recipient endpoint roles. It keeps up to the configured number of recent neighbors per relation with typed edge and log time-gap attributes. Same-step and future edges are excluded; labels are targets only.",
        "",
        "## Test Results",
        "",
        "Confusion matrices are `[[TN, FP], [FN, TP]]`.",
        "",
        "| Variant | Selected epoch | Validation threshold | Precision | Recall | F1 | Average precision | Confusion matrix | Training seconds | Available graph coverage* | Mean available edges/transaction* |",
        "|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|",
    ]
    for name in ("node_feature_only", "graph_enabled"):
        item = results["variants"][name]
        metrics = item["test_metrics"]
        coverage = item["graph_coverage"]["test"]
        lines.append(
            f"| {name} | {item['selected_epoch']} | {metrics['threshold']:.6f} | {metrics['precision']:.4f} | {metrics['recall']:.4f} | {metrics['f1']:.4f} | {metrics['average_precision']:.4f} | `{metrics['confusion_matrix']}` | {item['training_seconds']:.1f} | {coverage['edge_coverage']:.4f} | {coverage['average_edges_per_transaction']:.4f} |"
        )
    baseline = results["baseline"]
    lines.append(
        f"| Existing chronological XGBoost | validation-selected | {baseline['threshold']:.6f} | {baseline['precision']:.4f} | {baseline['recall']:.4f} | {baseline['f1']:.4f} | {baseline['average_precision']:.4f} | `{baseline['confusion_matrix']}` | not retrained | n/a | n/a |"
    )
    lines.append(
        "*Coverage describes sampled graph context available for the shared test rows; the node-feature-only model does not consume graph messages.*"
    )
    resources = results["resource_usage"]
    peak_working_set = resources.get("peak_working_set_mb")
    peak_text = (
        f"{peak_working_set:.0f} MiB"
        if peak_working_set is not None
        else "not captured for the completed training process"
    )
    lines.extend(
        [
            "",
            "## Training and Resource Details",
            "",
            f"Validation AP controlled checkpoint selection with patience {results['config']['early_stopping_patience']}; thresholds maximize validation F1, then precision, then higher threshold. Epoch histories and selected epochs are saved in each variant's `results.json`.",
            "",
            f"Preprocessing and graph staging time: {resources['preprocessing_and_graph_staging_seconds']:.1f} s (graph staging alone: {resources['graph_staging_seconds']:.1f} s). Memory-mapped temporary staging size: {resources['staging_bytes'] / (1024 ** 3):.2f} GiB. Peak process working set: {peak_text}.",
            "",
            "## Limitations",
            "",
            "PaySim labels are simulator-generated, not independently verified real-world fraud. The XGBoost comparison uses the same held-out period but a separately trained model and its own selected threshold, so it is a practical baseline comparison rather than a controlled architecture-only comparison. Context sampling and a finite recent-neighbor window still omit graph history. The node-only and graph-enabled variants are controlled against one another, but single-seed results do not establish robustness. Do not integrate a GNN unless results support it and repeated-seed checks confirm stability.",
            "",
        ]
    )
    return "\n".join(lines)


def run_experiment(
    output_dir=OUT,
    target_sample_fraction=TARGET_SAMPLE_FRACTION,
    context_fraction=CONTEXT_SAMPLE_FRACTION,
    neighbors_per_relation=RECENT_NEIGHBORS_PER_RELATION,
    hidden_dim=HIDDEN_DIM,
    batch_size=BATCH_SIZE,
    max_epochs=MAX_EPOCHS,
    patience=EARLY_STOPPING_PATIENCE,
):
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing v2 output: {output_dir}")
    for split in ("train", "validation", "test"):
        if not (SPLITS / f"{split}.csv").is_file():
            raise FileNotFoundError(f"Missing chronological split: {SPLITS / f'{split}.csv'}")
    if not 0 < target_sample_fraction <= 1 or not 0 < context_fraction <= 1:
        raise ValueError("Target and context fractions must be in (0, 1].")
    if neighbors_per_relation < 1 or batch_size < 1 or max_epochs < 1 or patience < 1:
        raise ValueError("Neighbor count, batch size, epochs, and patience must be positive.")
    if not BASELINE_RESULTS.is_file():
        raise FileNotFoundError(f"Baseline result file missing: {BASELINE_RESULTS}")

    set_determinism()
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = SPLITS / "train.csv"
    preprocessing_start = time.perf_counter()
    scaler, encoder, preprocessing_rows = fit_preprocessing(
        train_path, target_sample_fraction
    )
    feature_count = len(NUMERIC_FEATURES) + sum(
        len(categories) for categories in encoder.categories_
    )
    graph = TemporalTransactionGraph(feature_count, neighbors_per_relation)
    config = {
        "experiment": "temporal-transaction-gnn-v2",
        "framework": "TensorFlow/Keras",
        "tensorflow_version": tf.__version__,
        "compute_device": "CPU",
        "split_directory": "Dataset (PaySim)/chronological_splits",
        "split_step_ranges": {"train": [1, 450], "validation": [451, 600], "test": [601, 743]},
        "numeric_features": list(NUMERIC_FEATURES),
        "categorical_features": list(CATEGORICAL_FEATURES),
        "target_column": TARGET_COLUMN,
        "excluded_model_inputs": ["isFraud", "isFlaggedFraud", "nameOrig", "nameDest"],
        "edge_relations": [f"prior_{left}_to_current_{right}" for left, right in RELATIONS],
        "edge_features": [f"prior_{left}_to_current_{right}" for left, right in RELATIONS] + ["normalized_log_step_gap"],
        "neighbors_per_relation": neighbors_per_relation,
        "maximum_edges_per_transaction": len(RELATIONS) * neighbors_per_relation,
        "target_sample_fraction": target_sample_fraction,
        "context_sample_fraction": context_fraction,
        "preprocessing_fit_scope": "deterministic sampled training rows only",
        "label_policy": "fraud labels are targets only, never node/edge/model inputs",
        "same_step_and_future_edges": "excluded; build step examples before updating its graph history",
        "random_seed": SEED,
        "hidden_dim": hidden_dim,
        "batch_size": batch_size,
        "maximum_epochs": max_epochs,
        "early_stopping_monitor": "validation_average_precision (exact sklearn average_precision_score)",
        "early_stopping_patience": patience,
        "optimizer": "Adam(learning_rate=0.001)",
        "loss": "binary cross-entropy with train-derived inverse-frequency sample weights",
        "preprocessing_training_rows": int(preprocessing_rows),
    }

    with tempfile.TemporaryDirectory(prefix="paysim-gnn-v2-") as temporary_dir:
        staging_start = time.perf_counter()
        graph_coverage = {}
        for name, fraction, seed_offset in (
            ("train", target_sample_fraction, 0),
            ("validation", 1.0, 100_000),
            ("test", 1.0, 200_000),
        ):
            graph_coverage[name] = stage_split(
                SPLITS / f"{name}.csv",
                name,
                temporary_dir,
                graph,
                scaler,
                encoder,
                target_fraction=fraction,
                context_fraction=context_fraction,
                split_seed=seed_offset,
                graph_enabled=True,
            )
        graph_staging_seconds = time.perf_counter() - staging_start
        staging_bytes = sum(
            path.stat().st_size for path in Path(temporary_dir).glob("*.npy")
        )

        train_labels = np.load(Path(temporary_dir) / "train_labels.npy")
        validation_labels = np.load(Path(temporary_dir) / "validation_labels.npy")
        test_labels = np.load(Path(temporary_dir) / "test_labels.npy")
        for split_name, labels in (
            ("training", train_labels),
            ("validation", validation_labels),
            ("test", test_labels),
        ):
            if np.unique(labels).size != 2:
                raise ValueError(f"{split_name} rows must contain both classes.")
        config["training_target_rows"] = int(len(train_labels))
        config["training_fraud_rows"] = int(train_labels.sum())
        config["validation_rows"] = int(len(validation_labels))
        config["test_rows"] = int(len(test_labels))
        config["test_fraud_rows"] = int(test_labels.sum())
        variants = {}

        # Both variants consume the same staged node features and labels.
        for name, graph_enabled in (("node_feature_only", False), ("graph_enabled", True)):
            set_determinism(SEED)
            variant_config = {
                **config,
                "neighbors_per_relation": neighbors_per_relation,
            }
            variants[name] = _fit_variant(
                name,
                graph_enabled,
                temporary_dir,
                scaler,
                encoder,
                variant_config,
                output_dir / name,
                batch_size=batch_size,
                max_epochs=max_epochs,
                patience=patience,
            )
            variants[name]["graph_coverage"] = graph_coverage
            variant_results_path = output_dir / name / "results.json"
            saved_variant = json.loads(variant_results_path.read_text(encoding="utf-8"))
            saved_variant["graph_coverage"] = graph_coverage
            variant_results_path.write_text(json.dumps(saved_variant, indent=2), encoding="utf-8")

        results = {
            "config": config,
            "variants": variants,
            "graph_coverage": graph_coverage,
            "baseline": _baseline_results(),
            "resource_usage": {
                "preprocessing_and_graph_staging_seconds": time.perf_counter() - preprocessing_start,
                "graph_staging_seconds": graph_staging_seconds,
                "staging_bytes": int(staging_bytes),
                "peak_working_set_mb": process_memory_mb(),
                "temporary_memmaps_cleaned": True,
            },
        }
        (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        (output_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        (output_dir / "report.md").write_text(_report(results), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"Saved controlled GNN v2 experiment to {output_dir}")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--target-sample-fraction", type=float, default=TARGET_SAMPLE_FRACTION)
    parser.add_argument("--context-fraction", type=float, default=CONTEXT_SAMPLE_FRACTION)
    parser.add_argument("--neighbors-per-relation", type=int, default=RECENT_NEIGHBORS_PER_RELATION)
    parser.add_argument("--hidden-dim", type=int, default=HIDDEN_DIM)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=EARLY_STOPPING_PATIENCE)
    args = parser.parse_args()
    run_experiment(
        output_dir=args.output_dir,
        target_sample_fraction=args.target_sample_fraction,
        context_fraction=args.context_fraction,
        neighbors_per_relation=args.neighbors_per_relation,
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
        max_epochs=args.epochs,
        patience=args.patience,
    )


if __name__ == "__main__":
    main()