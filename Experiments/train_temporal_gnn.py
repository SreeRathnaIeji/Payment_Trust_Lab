"""Train a past-only transaction graph neural network on chronological PaySim.

Run from the project root with the active TensorFlow environment:
    python Experiments/train_temporal_gnn.py

Each transaction is a node. For a transaction query, up to four incoming
messages come from sampled earlier transactions sharing sender/recipient
accounts by endpoint role. All rows in one PaySim step are scored before any
of that step is added to graph history. Fraud labels are targets only.
"""

from __future__ import annotations

import argparse
import json
import math
import random
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
OUT = ROOT / "Experiments" / "artifacts" / "gnn_experiment"
MODEL_VERSION = "temporal-transaction-gnn-v1"
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
TARGET_COLUMN = "isFraud"
GRAPH_COLUMNS = ("nameOrig", "nameDest")
USECOLS = INPUT_COLUMNS + list(GRAPH_COLUMNS) + [TARGET_COLUMN]
RELATION_NAMES = (
    "prior_sender_to_current_sender",
    "prior_recipient_to_current_sender",
    "prior_sender_to_current_recipient",
    "prior_recipient_to_current_recipient",
)
EDGE_FEATURE_COUNT = len(RELATION_NAMES) + 1
MAX_STEP_GAP_SCALE = math.log1p(743)
CHUNK_SIZE = 100_000
SEED = 42
DEFAULT_SAMPLE_FRACTION = 0.10
DEFAULT_HIDDEN_DIM = 32
DEFAULT_BATCH_SIZE = 2048
DEFAULT_EPOCHS = 20


def set_determinism(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError):
        pass


def _sample_frame(frame, fraction, random_state):
    if fraction >= 1:
        return np.ones(len(frame), dtype=bool)
    chosen = frame.sample(frac=fraction, random_state=random_state).index
    return frame.index.isin(chosen)


def fit_preprocessing(training_path, sample_fraction=DEFAULT_SAMPLE_FRACTION):
    """Fit scaling and category encoding using a deterministic train sample."""
    numeric_parts = []
    category_parts = []
    reader = pd.read_csv(
        training_path,
        usecols=INPUT_COLUMNS,
        chunksize=CHUNK_SIZE,
    )
    for chunk_index, chunk in enumerate(reader):
        selected = _sample_frame(chunk, sample_fraction, SEED + chunk_index)
        sample = chunk.loc[selected]
        if not sample.empty:
            numeric_parts.append(sample[list(NUMERIC_FEATURES)])
            category_parts.append(sample[list(CATEGORICAL_FEATURES)])
    if not numeric_parts:
        raise ValueError("No training rows were selected to fit preprocessing.")

    numeric_training = pd.concat(numeric_parts, ignore_index=True)
    category_training = pd.concat(category_parts, ignore_index=True)
    scaler = StandardScaler().fit(numeric_training)
    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
        dtype=np.float32,
    ).fit(category_training)
    return scaler, encoder, len(numeric_training)


def transform_node_features(frame, scaler, encoder):
    numeric = scaler.transform(frame[list(NUMERIC_FEATURES)]).astype(
        np.float32, copy=False
    )
    categorical = encoder.transform(frame[list(CATEGORICAL_FEATURES)]).astype(
        np.float32, copy=False
    )
    return np.concatenate((numeric, categorical), axis=1)


class TemporalTransactionGraph:
    """Bounded temporal graph state with transaction nodes and typed edges."""

    def __init__(self, feature_count, initial_capacity=1024):
        self.feature_count = int(feature_count)
        self.capacity = max(1, int(initial_capacity))
        self.size = 0
        self.node_features = np.zeros(
            (self.capacity, self.feature_count), dtype=np.float32
        )
        self.node_steps = np.full(self.capacity, -1, dtype=np.int32)
        self.node_row_ids = np.full(self.capacity, -1, dtype=np.int64)
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
        row_ids = np.full(capacity, -1, dtype=np.int64)
        features[: self.size] = self.node_features[: self.size]
        steps[: self.size] = self.node_steps[: self.size]
        row_ids[: self.size] = self.node_row_ids[: self.size]
        self.node_features = features
        self.node_steps = steps
        self.node_row_ids = row_ids
        self.capacity = capacity

    def process_step(self, frame, node_features, context_mask, target_mask):
        """Create examples from past history, then add sampled current events.

        This two-phase operation excludes same-step and future edges. The
        isFraud column is read only after graph features have been constructed,
        to return labels separately from model inputs.
        """
        if frame.empty:
            raise ValueError("Cannot construct graph examples from an empty step.")
        if frame["step"].nunique() != 1:
            raise ValueError("process_step requires rows from exactly one PaySim step.")
        if node_features.shape != (len(frame), self.feature_count):
            raise ValueError("Node feature matrix does not match this step.")

        context_mask = np.asarray(context_mask, dtype=bool)
        target_mask = np.asarray(target_mask, dtype=bool)
        if context_mask.shape != (len(frame),) or target_mask.shape != (len(frame),):
            raise ValueError("Context and target masks must have one value per row.")

        row_count = len(frame)
        neighbors = np.zeros(
            (row_count, 4, self.feature_count), dtype=np.float32
        )
        edge_features = np.zeros(
            (row_count, 4, EDGE_FEATURE_COUNT), dtype=np.float32
        )
        neighbor_mask = np.zeros((row_count, 4), dtype=np.float32)
        step = int(frame["step"].iloc[0])
        sources = frame["nameOrig"].astype(str).to_numpy()
        recipients = frame["nameDest"].astype(str).to_numpy()

        for row_index, (source, recipient) in enumerate(zip(sources, recipients)):
            prior_ids = (
                self.source_history.get(source),
                self.recipient_history.get(source),
                self.source_history.get(recipient),
                self.recipient_history.get(recipient),
            )
            for relation_index, prior_id in enumerate(prior_ids):
                if prior_id is None:
                    continue
                prior_step = int(self.node_steps[prior_id])
                if prior_step >= step:
                    continue
                neighbors[row_index, relation_index] = self.node_features[prior_id]
                edge_features[row_index, relation_index, relation_index] = 1.0
                gap = step - prior_step
                edge_features[row_index, relation_index, -1] = (
                    np.log1p(gap) / MAX_STEP_GAP_SCALE
                )
                neighbor_mask[row_index, relation_index] = 1.0

        target_features = (
            node_features[target_mask],
            neighbors[target_mask],
            edge_features[target_mask],
            neighbor_mask[target_mask],
        )
        target_labels = frame.loc[target_mask, TARGET_COLUMN].to_numpy(dtype=np.int8)

        selected_rows = np.flatnonzero(context_mask)
        self._ensure_capacity(len(selected_rows))
        row_to_node_id = np.full(row_count, -1, dtype=np.int64)
        if len(selected_rows):
            node_ids = np.arange(self.size, self.size + len(selected_rows))
            row_to_node_id[selected_rows] = node_ids
            self.node_features[node_ids] = node_features[selected_rows]
            self.node_steps[node_ids] = step
            row_ids = frame["__row_id"].to_numpy(dtype=np.int64)
            self.node_row_ids[node_ids] = row_ids[selected_rows]
            self.size += len(selected_rows)

            updates = {}
            for row_index in selected_rows:
                node_id = int(row_to_node_id[row_index])
                row_id = int(row_ids[row_index])
                source = sources[row_index]
                recipient = recipients[row_index]
                for history, account in (
                    (self.source_history, source),
                    (self.recipient_history, recipient),
                ):
                    previous = updates.get((id(history), account))
                    if previous is None or row_id < previous[0]:
                        updates[(id(history), account)] = (row_id, node_id)
            for (history_id, account), (_, node_id) in updates.items():
                history = (
                    self.source_history
                    if history_id == id(self.source_history)
                    else self.recipient_history
                )
                history[account] = node_id

        return target_features, target_labels


def _step_groups(path, context_fraction, seed_offset):
    pending = pd.DataFrame()
    previous_chunk_last = None
    row_offset = 0
    reader = pd.read_csv(path, usecols=USECOLS, chunksize=CHUNK_SIZE)
    for chunk_index, chunk in enumerate(reader):
        if not chunk["step"].is_monotonic_increasing:
            raise ValueError(f"Non-chronological rows found in {path}.")
        first_step = int(chunk["step"].iloc[0])
        if previous_chunk_last is not None and first_step < previous_chunk_last:
            raise ValueError(f"Step order moves backwards in {path}.")
        previous_chunk_last = int(chunk["step"].iloc[-1])
        chunk["__row_id"] = np.arange(
            row_offset, row_offset + len(chunk), dtype=np.int64
        )
        row_offset += len(chunk)
        chunk["__context"] = _sample_frame(
            chunk, context_fraction, SEED + seed_offset + chunk_index
        )
        combined = pd.concat((pending, chunk), ignore_index=True)
        last_step = combined["step"].iloc[-1]
        complete = combined.loc[combined["step"] != last_step]
        pending = combined.loc[combined["step"] == last_step].copy()
        for _, step_frame in complete.groupby("step", sort=False):
            yield step_frame
    if not pending.empty:
        yield pending


def _empty_graph_examples(feature_count):
    return (
        np.empty((0, feature_count), dtype=np.float32),
        np.empty((0, 4, feature_count), dtype=np.float32),
        np.empty((0, 4, EDGE_FEATURE_COUNT), dtype=np.float32),
        np.empty((0, 4), dtype=np.float32),
    )


def _concatenate_parts(parts, feature_count):
    if not parts:
        return _empty_graph_examples(feature_count)
    return tuple(np.concatenate([part[index] for part in parts]) for index in range(4))


def prepare_split(path, graph, scaler, encoder, *, training, context_fraction=0.10, seed_offset=0):
    """Prepare graph examples while advancing only sampled past context nodes."""
    inputs_by_step = []
    labels_by_step = []
    feature_count = graph.feature_count
    for step_frame in _step_groups(path, context_fraction, seed_offset):
        node_features = transform_node_features(step_frame, scaler, encoder)
        context_mask = step_frame["__context"].to_numpy(dtype=bool)
        target_mask = context_mask if training else np.ones(len(step_frame), dtype=bool)
        graph_inputs, labels = graph.process_step(
            step_frame,
            node_features,
            context_mask,
            target_mask,
        )
        if len(labels):
            inputs_by_step.append(graph_inputs)
            labels_by_step.append(labels)
    if not labels_by_step:
        raise ValueError(f"No graph examples were produced from {path}.")
    return (
        _concatenate_parts(inputs_by_step, feature_count),
        np.concatenate(labels_by_step).astype(np.int8, copy=False),
    )


class TemporalMessagePassing(tf.keras.Model):
    """One relation-aware message-passing layer for transaction nodes."""

    def __init__(self, feature_count, hidden_dim=DEFAULT_HIDDEN_DIM, dropout=0.2):
        super().__init__()
        self.feature_count = int(feature_count)
        self.hidden_dim = int(hidden_dim)
        self.message_projection = tf.keras.layers.Dense(
            self.hidden_dim, activation="relu", name="edge_message"
        )
        self.self_projection = tf.keras.layers.Dense(
            self.hidden_dim, activation="relu", name="self_projection"
        )
        self.update_projection = tf.keras.layers.Dense(
            self.hidden_dim, activation="relu", name="node_update"
        )
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.classifier = tf.keras.layers.Dense(1, name="fraud_logit")

    def call(self, inputs, training=False):
        node_features, neighbor_features, edge_features, neighbor_mask = inputs
        message_input = tf.concat((neighbor_features, edge_features), axis=-1)
        messages = self.message_projection(message_input)
        expanded_mask = tf.expand_dims(neighbor_mask, axis=-1)
        summed_messages = tf.reduce_sum(messages * expanded_mask, axis=1)
        neighbor_count = tf.reduce_sum(expanded_mask, axis=1)
        aggregated = tf.math.divide_no_nan(summed_messages, neighbor_count)
        own_embedding = self.self_projection(node_features)
        updated = self.update_projection(tf.concat((own_embedding, aggregated), axis=-1))
        updated = self.dropout(updated, training=training)
        return tf.math.sigmoid(self.classifier(updated))


def build_model(feature_count, hidden_dim=DEFAULT_HIDDEN_DIM):
    model = TemporalMessagePassing(feature_count, hidden_dim=hidden_dim)
    model(
        [
            tf.zeros((1, feature_count), dtype=tf.float32),
            tf.zeros((1, 4, feature_count), dtype=tf.float32),
            tf.zeros((1, 4, EDGE_FEATURE_COUNT), dtype=tf.float32),
            tf.zeros((1, 4), dtype=tf.float32),
        ]
    )
    return model


def predict_probabilities(model, graph_inputs, batch_size=DEFAULT_BATCH_SIZE):
    return model.predict(list(graph_inputs), batch_size=batch_size, verbose=0).reshape(-1)


def select_f1_threshold(labels, probabilities):
    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    if thresholds.size == 0:
        return float(probabilities[0])
    f1_values = 2 * precision[:-1] * recall[:-1] / np.maximum(
        precision[:-1] + recall[:-1], np.finfo(float).eps
    )
    best = max(
        range(len(thresholds)),
        key=lambda index: (f1_values[index], precision[index], thresholds[index]),
    )
    return float(thresholds[best])


def evaluate_predictions(labels, probabilities, threshold):
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


def load_artifact(artifact_dir):
    artifact_dir = Path(artifact_dir)
    config = json.loads((artifact_dir / "config.json").read_text(encoding="utf-8"))
    scaler = joblib.load(artifact_dir / "numeric_scaler.joblib")
    encoder = joblib.load(artifact_dir / "type_encoder.joblib")
    if config.get("numeric_features") != list(NUMERIC_FEATURES):
        raise ValueError("Saved GNN numeric feature schema is incompatible.")
    if config.get("categorical_features") != list(CATEGORICAL_FEATURES):
        raise ValueError("Saved GNN categorical feature schema is incompatible.")
    feature_count = len(NUMERIC_FEATURES) + sum(
        len(categories) for categories in encoder.categories_
    )
    if scaler.n_features_in_ != len(NUMERIC_FEATURES):
        raise ValueError("Saved GNN scaler feature count is incompatible.")
    model = build_model(feature_count, hidden_dim=config["hidden_dim"])
    model.load_weights(artifact_dir / "transaction_gnn.weights.h5")
    return {
        "model": model,
        "scaler": scaler,
        "encoder": encoder,
        "config": config,
        "threshold": float(config["validation_selected_threshold"]),
    }


def _baseline_summary():
    baseline = json.loads(BASELINE_RESULTS.read_text(encoding="utf-8"))
    metrics = baseline["test_metrics"]
    return {
        "model": "Existing chronological XGBoost baseline",
        "threshold": float(metrics["threshold"]),
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "f1": float(metrics["f1"]),
        "average_precision": float(metrics["average_precision"]),
        "confusion_matrix": [
            [int(metrics["TN"]), int(metrics["FP"])],
            [int(metrics["FN"]), int(metrics["TP"])],
        ],
    }


def _build_report(config, results):
    gnn = results["gnn_test_metrics"]
    baseline = results["existing_xgboost_test_metrics"]
    context_fraction = config.get(
        "context_sample_fraction", config.get("context_sampling_fraction")
    )
    lines = [
        "# Temporal Transaction GNN Experiment",
        "",
        "## Graph and Leakage Policy",
        "",
        "Each PaySim transaction is a node. Node features are the six numeric baseline features (step, amount, and four balance fields) plus one-hot transaction type. `isFraud` is the node classification target and is never a node, edge, or message feature. Account IDs are used only to construct edges; they are not embedded as model features.",
        "",
        "Directed edges point from a prior transaction to the current transaction when they share an account. The four edge relations distinguish sender-sender, recipient-sender, sender-recipient, and recipient-recipient matches. Edge attributes contain a one-hot relation code and normalized log step gap. The graph keeps one deterministically selected sampled predecessor per role, for at most four incoming edges per query.",
        "",
        "Transactions sharing the same PaySim step are featurized before any of that step is added to history. Therefore message sources have strictly earlier steps. Historical context events are sampled at the configured deterministic fraction; current validation/test queries are all evaluated. Labels are used only for training loss, validation threshold/epoch selection, and final metrics.",
        "",
        "The model performs relation-aware message projection, masked mean aggregation, and a transaction-level fraud prediction. This is a trainable message-passing GNN, distinct from the XGBoost degree-feature candidate and the fictional UI graph.",
        "",
        "## Training",
        "",
        f"Framework: TensorFlow {config['tensorflow_version']} on CPU. Sampled training target rows: {config['training_target_rows']:,}; fraud targets: {config['training_fraud_rows']:,}. Context sampling fraction: {context_fraction:.0%}. Best validation-loss epoch: {config['best_epoch']}; selected decision threshold: {config['validation_selected_threshold']:.6f}.",
        "",
        "Preprocessing was fit on the deterministic sampled training rows only. No oversampling was used; binary cross-entropy used training-derived class weights. Early stopping monitored validation loss. The chronological test period was not used for model or threshold selection.",
        "",
        "## Held-Out Test Results",
        "",
        "Confusion matrices use `[[TN, FP], [FN, TP]]`.",
        "",
        "| Model | Threshold | Precision | Recall | F1 | Average precision | Confusion matrix |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for name, metrics in (
        ("Temporal transaction GNN", gnn),
        ("Existing chronological XGBoost baseline", baseline),
    ):
        cm = metrics["confusion_matrix"]
        lines.append(
            f"| {name} | {metrics['threshold']:.6f} | {metrics['precision']:.4f} | {metrics['recall']:.4f} | {metrics['f1']:.4f} | {metrics['average_precision']:.4f} | `{cm}` |"
        )
    lines.extend(
        [
            "",
            "Both models report the same chronological PaySim test period. The baseline numbers are read from its existing saved result file and were not recomputed or modified.",
            "",
            "## Limitations",
            "",
            "PaySim is synthetic, so labels represent simulator-generated fraud rather than independently verified payment fraud. The temporal graph uses a deterministic 10% sample for context to bound memory and retains one representative predecessor for each account-role relation; this sparsifies histories and can omit useful interactions. Its absolute scores are not evidence of UPI or real-world performance. The GNN uses the same tabular transaction fields as the baseline, so differences reflect the message-passing architecture and sampled graph context rather than new external information.",
            "",
            "This experiment writes only to its new artifact directory; the live application and existing model artifacts are unchanged.",
            "",
        ]
    )
    return "\n".join(lines)


def save_artifacts(artifact_dir, model, scaler, encoder, config, results):
    artifact_dir = Path(artifact_dir)
    report = _build_report(config, results)
    artifact_dir.mkdir(parents=True, exist_ok=False)
    joblib.dump(scaler, artifact_dir / "numeric_scaler.joblib")
    joblib.dump(encoder, artifact_dir / "type_encoder.joblib")
    model.save_weights(artifact_dir / "transaction_gnn.weights.h5")
    (artifact_dir / "config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    (artifact_dir / "results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    (artifact_dir / "report.md").write_text(report, encoding="utf-8")


def write_missing_report(artifact_dir):
    """Create only a missing report; refuse to replace any existing file."""
    artifact_dir = Path(artifact_dir)
    report_path = artifact_dir / "report.md"
    if report_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing report: {report_path}")
    config = json.loads((artifact_dir / "config.json").read_text(encoding="utf-8"))
    results = json.loads((artifact_dir / "results.json").read_text(encoding="utf-8"))
    report = _build_report(config, results)
    with report_path.open("x", encoding="utf-8") as file:
        file.write(report)
    return report_path


def run_experiment(
    output_dir=OUT,
    sample_fraction=DEFAULT_SAMPLE_FRACTION,
    context_fraction=DEFAULT_SAMPLE_FRACTION,
    hidden_dim=DEFAULT_HIDDEN_DIM,
    batch_size=DEFAULT_BATCH_SIZE,
    epochs=DEFAULT_EPOCHS,
):
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing GNN output: {output_dir}")
    if not 0 < sample_fraction <= 1 or not 0 < context_fraction <= 1:
        raise ValueError("Training and context fractions must be in (0, 1].")
    for split_name in ("train", "validation", "test"):
        if not (SPLITS / f"{split_name}.csv").is_file():
            raise FileNotFoundError(f"Missing chronological split: {SPLITS / f'{split_name}.csv'}")
    if not BASELINE_RESULTS.is_file():
        raise FileNotFoundError(f"Existing XGBoost baseline results missing: {BASELINE_RESULTS}")

    set_determinism()
    train_path = SPLITS / "train.csv"
    print("Fitting preprocessing on deterministic training rows only...")
    scaler, encoder, preprocessing_rows = fit_preprocessing(train_path, sample_fraction)
    feature_count = len(NUMERIC_FEATURES) + sum(
        len(categories) for categories in encoder.categories_
    )
    graph = TemporalTransactionGraph(feature_count)

    print("Building past-only training examples...")
    training_inputs, y_train = prepare_split(
        train_path,
        graph,
        scaler,
        encoder,
        training=True,
        context_fraction=context_fraction,
        seed_offset=0,
    )
    positives = int(y_train.sum())
    negatives = len(y_train) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("Sampled training targets must contain both classes.")

    print("Building validation examples from prior training/validation context...")
    validation_inputs, y_validation = prepare_split(
        SPLITS / "validation.csv",
        graph,
        scaler,
        encoder,
        training=False,
        context_fraction=context_fraction,
        seed_offset=100_000,
    )
    print("Building test examples from prior train/validation/test context...")
    test_inputs, y_test = prepare_split(
        SPLITS / "test.csv",
        graph,
        scaler,
        encoder,
        training=False,
        context_fraction=context_fraction,
        seed_offset=200_000,
    )

    model = build_model(feature_count, hidden_dim=hidden_dim)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.AUC(curve="PR", name="pr_auc")],
    )
    class_weights = {
        0: len(y_train) / (2 * negatives),
        1: len(y_train) / (2 * positives),
    }
    callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
        mode="min",
    )
    print("Training the TensorFlow message-passing GNN...")
    history = model.fit(
        list(training_inputs),
        y_train.astype(np.float32),
        validation_data=(list(validation_inputs), y_validation.astype(np.float32)),
        batch_size=batch_size,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=[callback],
        shuffle=True,
        verbose=2,
    )
    best_epoch = int(np.argmin(history.history["val_loss"]) + 1)
    validation_probabilities = predict_probabilities(model, validation_inputs, batch_size)
    threshold = select_f1_threshold(y_validation, validation_probabilities)
    validation_metrics = evaluate_predictions(
        y_validation, validation_probabilities, threshold
    )
    print("Evaluating the held-out chronological test period once...")
    test_probabilities = predict_probabilities(model, test_inputs, batch_size)
    test_metrics = evaluate_predictions(y_test, test_probabilities, threshold)
    baseline_metrics = _baseline_summary()

    config = {
        "experiment": MODEL_VERSION,
        "framework": "TensorFlow/Keras",
        "tensorflow_version": tf.__version__,
        "compute_device": "CPU",
        "dataset": "Synthetic PaySim",
        "split_directory": "Dataset (PaySim)/chronological_splits",
        "split_step_ranges": {"train": [1, 450], "validation": [451, 600], "test": [601, 743]},
        "numeric_features": list(NUMERIC_FEATURES),
        "categorical_features": list(CATEGORICAL_FEATURES),
        "target_column": TARGET_COLUMN,
        "excluded_model_inputs": ["isFraud", "isFlaggedFraud", "nameOrig", "nameDest"],
        "graph_node_definition": "one transaction event per node",
        "graph_edge_definition": "directed from prior transaction to current when endpoint accounts match",
        "edge_relations": list(RELATION_NAMES),
        "edge_features": list(RELATION_NAMES) + ["normalized_log_step_gap"],
        "incoming_edges_per_query_max": 4,
        "same_step_edges": "excluded; all queries at a step are built before step history updates",
        "context_sample_fraction": context_fraction,
        "training_target_sampling_fraction": sample_fraction,
        "preprocessing_fit_scope": "deterministic sampled training rows only",
        "label_leakage_policy": "labels are targets only and never enter graph nodes, edges, or model inputs",
        "random_seed": SEED,
        "hidden_dim": hidden_dim,
        "batch_size": batch_size,
        "maximum_epochs": epochs,
        "best_epoch": best_epoch,
        "early_stopping": "validation loss, patience 3, restore best weights",
        "loss": "binary cross entropy with inverse-frequency training class weights",
        "optimizer": "Adam, learning_rate=0.001",
        "preprocessing_training_rows": int(preprocessing_rows),
        "training_target_rows": int(len(y_train)),
        "training_fraud_rows": positives,
        "validation_rows": int(len(y_validation)),
        "test_rows": int(len(y_test)),
        "validation_selected_threshold": threshold,
    }
    results = {
        "validation_metrics": validation_metrics,
        "gnn_test_metrics": test_metrics,
        "existing_xgboost_test_metrics": baseline_metrics,
        "test_fraud_rows": int(y_test.sum()),
        "validation_fraud_rows": int(y_validation.sum()),
    }
    save_artifacts(output_dir, model, scaler, encoder, config, results)
    print(json.dumps(results, indent=2))
    print(f"New GNN weights, configuration, and results saved to {output_dir}")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--sample-fraction", type=float, default=DEFAULT_SAMPLE_FRACTION)
    parser.add_argument("--context-fraction", type=float, default=DEFAULT_SAMPLE_FRACTION)
    parser.add_argument("--hidden-dim", type=int, default=DEFAULT_HIDDEN_DIM)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument(
        "--write-report-only",
        action="store_true",
        help="Create a missing report from an existing completed results.json without modifying other artifacts.",
    )
    args = parser.parse_args()
    if args.write_report_only:
        print(write_missing_report(args.output_dir))
        return
    run_experiment(
        output_dir=args.output_dir,
        sample_fraction=args.sample_fraction,
        context_fraction=args.context_fraction,
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
        epochs=args.epochs,
    )


if __name__ == "__main__":
    main()