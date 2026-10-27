import json
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from Experiments.train_temporal_gnn_v2 import (
    CATEGORICAL_FEATURES,
    EDGE_FEATURE_COUNT,
    NUMERIC_FEATURES,
    RELATIONS,
    TemporalTransactionGraph,
    ValidationAveragePrecisionEarlyStopping,
    build_model,
    load_variant,
    select_best_epoch,
)


def graph_frame(steps, labels=None):
    labels = labels if labels is not None else [0] * len(steps)
    return pd.DataFrame(
        {
            "step": steps,
            "amount": np.arange(len(steps), dtype=np.float32) + 1,
            "oldbalanceOrg": np.arange(len(steps), dtype=np.float32) + 10,
            "newbalanceOrig": np.arange(len(steps), dtype=np.float32) + 9,
            "oldbalanceDest": np.arange(len(steps), dtype=np.float32) + 20,
            "newbalanceDest": np.arange(len(steps), dtype=np.float32) + 21,
            "type": ["TRANSFER"] * len(steps),
            "nameOrig": ["A"] * len(steps),
            "nameDest": [f"B{index}" for index in range(len(steps))],
            "isFraud": labels,
            "__row_id": np.arange(len(steps), dtype=np.int64),
        }
    )


class TemporalGNNV2Tests(unittest.TestCase):
    def setUp(self):
        tf.keras.utils.set_random_seed(37)

    def test_epoch_selection_uses_validation_ap_and_prefers_earlier_tie(self):
        history = [
            {"epoch": 1, "val_loss": 0.20, "val_average_precision": 0.40},
            {"epoch": 2, "val_loss": 0.10, "val_average_precision": 0.65},
            {"epoch": 3, "val_loss": 0.08, "val_average_precision": 0.65},
            {"epoch": 4, "val_loss": 0.30, "val_average_precision": 0.50},
        ]
        self.assertEqual(select_best_epoch(history), 2)

    def test_ap_callback_restores_best_weights_and_stops_after_patience(self):
        class ValidationDataset:
            labels = np.array([0, 1, 0, 1], dtype=np.int8)
            batch_size = 4

            def __len__(self):
                return 1

            def __getitem__(self, index):
                return (np.zeros((4, 1), dtype=np.float32),), self.labels

        class ModelStub:
            def __init__(self):
                self.predictions = np.array([0.9, 0.1, 0.9, 0.1], dtype=np.float32)
                self.weight = 0
                self.stop_training = False

            def __call__(self, inputs, training=False):
                return tf.convert_to_tensor(self.predictions[:, None])

            def get_weights(self):
                return [np.array([self.weight], dtype=np.float32)]

            def set_weights(self, weights):
                self.weight = int(weights[0][0])

        model = ModelStub()
        callback = ValidationAveragePrecisionEarlyStopping(
            ValidationDataset(), ValidationDataset.labels, batch_size=4, patience=1
        )
        callback.set_model(model)
        callback.on_epoch_end(0, {"val_loss": 0.1})
        model.weight = 2
        model.predictions = np.array([0.1, 0.9, 0.1, 0.9], dtype=np.float32)
        callback.on_epoch_end(1, {"val_loss": 0.5})
        model.weight = 3
        model.predictions = np.array([0.9, 0.1, 0.9, 0.1], dtype=np.float32)
        callback.on_epoch_end(2, {"val_loss": 0.01})
        callback.on_train_end()

        self.assertEqual(callback.best_epoch, 2)
        self.assertEqual(model.weight, 2)
        self.assertTrue(model.stop_training)
        self.assertGreater(callback.epoch_history[1]["val_average_precision"], callback.epoch_history[0]["val_average_precision"])

    def test_retains_multiple_recent_prior_events_and_never_same_step(self):
        frame = graph_frame([1, 2, 3, 4, 4])
        graph = TemporalTransactionGraph(feature_count=1, neighbors_per_relation=4)

        for row in range(3):
            one = frame.iloc[row : row + 1].copy()
            features = np.array([[float(row + 1)]], dtype=np.float32)
            graph.process_step(one, features, [True], [False])

        same_step = frame.iloc[3:5].copy()
        same_step_features = np.array([[4.0], [5.0]], dtype=np.float32)
        inputs, labels = graph.process_step(
            same_step,
            same_step_features,
            context_mask=[True, True],
            target_mask=[True, True],
        )
        self.assertEqual(len(labels), 2)
        self.assertEqual(graph.edge_count, 16)
        np.testing.assert_array_equal(inputs[3][:, :4], np.tile([1, 1, 1, 0], (2, 1)))
        np.testing.assert_array_equal(inputs[1][0, :3, 0], [3, 2, 1])
        np.testing.assert_array_equal(inputs[1][1, :3, 0], [3, 2, 1])
        self.assertEqual(float(inputs[3][:, 4:].sum()), 0.0)

    def test_future_step_transaction_is_not_used_as_predecessor(self):
        graph = TemporalTransactionGraph(feature_count=1, neighbors_per_relation=4)
        future = graph_frame([5])
        graph.process_step(
            future,
            np.array([[5.0]], dtype=np.float32),
            context_mask=[True],
            target_mask=[False],
        )
        earlier = graph_frame([4])
        inputs, _ = graph.process_step(
            earlier,
            np.array([[4.0]], dtype=np.float32),
            context_mask=[False],
            target_mask=[True],
        )
        self.assertEqual(float(inputs[3].sum()), 0.0)

    def test_label_changes_do_not_change_neighbor_features_or_edges(self):
        base = graph_frame([1, 2, 3], [0, 0, 0])
        changed = graph_frame([1, 2, 3], [1, 1, 1])
        features = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
        outputs = []
        for frame in (base, changed):
            graph = TemporalTransactionGraph(1, neighbors_per_relation=4)
            for row in range(2):
                graph.process_step(
                    frame.iloc[row : row + 1],
                    features[row : row + 1],
                    [True],
                    [False],
                )
            inputs, _ = graph.process_step(frame.iloc[2:], features[2:], [False], [True])
            outputs.append(inputs)
        for first, second in zip(outputs[0], outputs[1]):
            np.testing.assert_array_equal(first, second)

    def test_node_only_and_graph_models_train_and_saved_weights_reload(self):
        rng = np.random.default_rng(17)
        nodes = rng.normal(size=(12, len(NUMERIC_FEATURES) + 1)).astype(np.float32)
        labels = np.array([0, 1] * 6, dtype=np.float32)
        scaler = StandardScaler().fit(np.arange(24, dtype=np.float32).reshape(4, 6))
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore").fit(
            pd.DataFrame({"type": ["TRANSFER"]})
        )

        for graph_enabled in (False, True):
            model = build_model(
                nodes.shape[1],
                graph_enabled=graph_enabled,
                neighbors_per_relation=4,
                hidden_dim=8,
            )
            model.compile(optimizer="adam", loss="binary_crossentropy")
            if graph_enabled:
                inputs = [
                    nodes,
                    np.zeros((12, 16, nodes.shape[1]), dtype=np.float32),
                    np.zeros((12, 16, EDGE_FEATURE_COUNT), dtype=np.float32),
                    np.zeros((12, 16), dtype=np.float32),
                ]
            else:
                inputs = [nodes]
            model.fit(inputs, labels, epochs=1, batch_size=4, verbose=0)
            expected = model(inputs, training=False).numpy()

            with tempfile.TemporaryDirectory() as directory:
                variant_dir = Path(directory)
                config = {
                    "numeric_features": list(NUMERIC_FEATURES),
                    "categorical_features": list(CATEGORICAL_FEATURES),
                    "graph_enabled": graph_enabled,
                    "neighbors_per_relation": 4,
                    "hidden_dim": 8,
                }
                joblib.dump(scaler, variant_dir / "numeric_scaler.joblib")
                joblib.dump(encoder, variant_dir / "type_encoder.joblib")
                model.save_weights(variant_dir / "model.weights.h5")
                (variant_dir / "config.json").write_text(
                    json.dumps(config), encoding="utf-8"
                )
                loaded = load_variant(variant_dir)
                actual = loaded["model"](inputs, training=False).numpy()
                np.testing.assert_allclose(expected, actual, rtol=0, atol=1e-7)


if __name__ == "__main__":
    unittest.main()