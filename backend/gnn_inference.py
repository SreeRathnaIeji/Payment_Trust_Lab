"""Inference helpers for the experimental node-feature-only PaySim GNN."""

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
NODE_ONLY_ARTIFACT_DIR = (
    PROJECT_ROOT
    / "Experiments"
    / "artifacts"
    / "gnn_experiment_v2"
    / "node_feature_only"
)
NUMERIC_FEATURES = (
    "step",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
)
CATEGORICAL_FEATURES = ("type",)
REQUIRED_ARTIFACTS = (
    "config.json",
    "numeric_scaler.joblib",
    "type_encoder.joblib",
    "model.weights.h5",
)


def load_node_only_bundle(artifact_dir: str | Path = NODE_ONLY_ARTIFACT_DIR) -> dict[str, Any]:
    """Load and validate the saved node-only bundle without importing TF early."""
    artifact_dir = Path(artifact_dir)
    missing = [name for name in REQUIRED_ARTIFACTS if not (artifact_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing node-only GNN artifact(s) in {artifact_dir}: {', '.join(missing)}"
        )

    # TensorFlow is imported only when the optional GNN bundle is loaded.
    from Experiments.train_temporal_gnn_v2 import load_variant

    bundle = load_variant(artifact_dir)
    config = bundle["config"]
    if config.get("variant") != "node_feature_only" or config.get("graph_enabled") is not False:
        raise ValueError("The experimental endpoint only accepts the node-feature-only GNN bundle.")
    if config.get("numeric_features") != list(NUMERIC_FEATURES):
        raise ValueError("Saved GNN numeric feature schema does not match the API schema.")
    if config.get("categorical_features") != list(CATEGORICAL_FEATURES):
        raise ValueError("Saved GNN categorical feature schema does not match the API schema.")

    threshold = float(config["selected_threshold"])
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Saved GNN decision threshold is outside [0, 1].")
    bundle["threshold"] = threshold
    bundle["variant"] = "experimental-node-feature-only-gnn-v2"
    return bundle


def prepare_node_features(
    transaction: Mapping[str, Any] | Any,
    bundle: Mapping[str, Any],
) -> np.ndarray:
    """Apply the v2 training scaler/encoder in the saved feature order."""
    if hasattr(transaction, "model_dump"):
        values = transaction.model_dump()
    elif isinstance(transaction, Mapping):
        values = transaction
    else:
        values = vars(transaction)

    required = list(NUMERIC_FEATURES + CATEGORICAL_FEATURES)
    missing = [name for name in required if name not in values]
    if missing:
        raise ValueError(f"Missing GNN input feature(s): {', '.join(missing)}")

    frame = pd.DataFrame([{name: values[name] for name in required}])
    from Experiments.train_temporal_gnn_v2 import transform_nodes

    features = transform_nodes(frame, bundle["scaler"], bundle["encoder"])
    expected_count = len(NUMERIC_FEATURES) + sum(
        len(categories) for categories in bundle["encoder"].categories_
    )
    if features.shape != (1, expected_count):
        raise ValueError("Prepared GNN node features do not match the saved model input shape.")
    if not np.isfinite(features).all():
        raise ValueError("Prepared GNN node features contain non-finite values.")
    return features


def predict_node_only(
    transaction: Mapping[str, Any] | Any,
    bundle: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a thresholded decision and explicitly uncalibrated raw score."""
    if bundle.get("variant") != "experimental-node-feature-only-gnn-v2":
        raise ValueError("A validated node-feature-only GNN bundle is required.")
    features = prepare_node_features(transaction, bundle)
    output = bundle["model"]((features,), training=False)
    raw_score = float(np.asarray(output).reshape(-1)[0])
    if not np.isfinite(raw_score):
        raise RuntimeError("The GNN returned a non-finite score.")
    threshold = float(bundle["threshold"])
    is_fraud = raw_score >= float(np.float32(threshold))
    return {
        "decision": "FRAUD" if is_fraud else "NOT FRAUD",
        "is_fraud": is_fraud,
        "raw_score": raw_score,
        "threshold": threshold,
    }