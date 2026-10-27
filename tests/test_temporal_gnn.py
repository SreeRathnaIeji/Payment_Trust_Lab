import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from Experiments.train_temporal_gnn import (
    CATEGORICAL_FEATURES,
    EDGE_FEATURE_COUNT,
    NUMERIC_FEATURES,
    TemporalMessagePassing,
    TemporalTransactionGraph,
    build_model,
    fit_preprocessing,
    load_artifact,
    predict_probabilities,
    save_artifacts,
    transform_node_features,
)


def make_frame(labels=None):
    labels = labels or [0, 1, 0, 1]
    return pd.DataFrame(
        {
            "step": [1, 1, 2, 3],
            "amount": [10.0, 20.0, 30.0, 40.0],
            "oldbalanceOrg": [100.0, 200.0, 300.0, 400.0],
            "newbalanceOrig": [90.0, 180.0, 270.0, 360.0],
            "oldbalanceDest": [50.0, 60.0, 70.0, 80.0],
            "newbalanceDest": [60.0, 80.0, 100.0, 120.0],
            "type": ["TRANSFER", "PAYMENT", "PAYMENT", "TRANSFER"],
            "nameOrig": ["A", "A", "A", "B"],
            "nameDest": ["B", "C", "C", "A"],
            "isFraud": labels,
            "__row_id": [0, 1, 2, 3],
        }
    )


class TemporalGNNTests(unittest.TestCase):
    def setUp(self):
        tf.keras.utils.set_random_seed(23)

    def test_graph_excludes_same_step_edges_and_tracks_prior_account_roles(self):
        frame = make_frame()
        node_features = np.arange(len(frame) * 3, dtype=np.float32).reshape(4, 3)
        graph = TemporalTransactionGraph(feature_count=3)

        first_step = frame.iloc[:2].copy()
        first_inputs, first_labels = graph.process_step(
            first_step,
            node_features[:2],
            context_mask=[True, True],
            target_mask=[True, True],
        )
        self.assertEqual(len(first_labels), 2)
        np.testing.assert_array_equal(first_inputs[3], np.zeros((2, 4)))

        later_step = frame.iloc[2:3].copy()
        later_inputs, _ = graph.process_step(
            later_step,
            node_features[2:3],
            context_mask=[False],
            target_mask=[True],
        )
        np.testing.assert_array_equal(later_inputs[3][0], [1, 0, 0, 1])
        np.testing.assert_array_equal(later_inputs[1][0, 0], node_features[0])
        np.testing.assert_array_equal(later_inputs[1][0, 3], node_features[1])
        self.assertEqual(later_inputs[2][0, 0, 0], 1.0)
        self.assertAlmostEqual(
            float(later_inputs[2][0, 0, -1]),
            np.log1p(1) / np.log1p(743),
            places=6,
        )

    def test_changing_labels_does_not_change_graph_model_inputs(self):
        features = np.arange(12, dtype=np.float32).reshape(4, 3)
        first_frame = make_frame([0, 0, 0, 0])
        second_frame = make_frame([1, 1, 1, 1])
        prepared = []
        for frame in (first_frame, second_frame):
            graph = TemporalTransactionGraph(feature_count=3)
            first_step = frame.iloc[:2].copy()
            graph.process_step(
                first_step,
                features[:2],
                context_mask=[True, True],
                target_mask=[True, True],
            )
            later_step = frame.iloc[2:3].copy()
            inputs, labels = graph.process_step(
                later_step,
                features[2:3],
                context_mask=[False],
                target_mask=[True],
            )
            prepared.append((inputs, labels))

        for left, right in zip(prepared[0][0], prepared[1][0]):
            np.testing.assert_array_equal(left, right)
        self.assertNotEqual(prepared[0][1][0], prepared[1][1][0])

    def test_preprocessing_is_fit_on_training_file_and_handles_unseen_type(self):
        frame = make_frame()
        with tempfile.TemporaryDirectory() as directory:
            training_path = Path(directory) / "train.csv"
            frame.iloc[:2].drop(columns=["__row_id"]).to_csv(training_path, index=False)
            scaler, encoder, rows = fit_preprocessing(training_path, sample_fraction=1.0)

        self.assertEqual(rows, 2)
        expected_means = frame.iloc[:2][list(NUMERIC_FEATURES)].mean().to_numpy()
        np.testing.assert_allclose(scaler.mean_, expected_means)
        unseen = frame.iloc[2:3].copy()
        unseen["type"] = "CASH_IN"
        self.assertNotIn("CASH_IN", encoder.categories_[0])
        transformed = transform_node_features(unseen, scaler, encoder)
        self.assertEqual(transformed.shape[1], len(NUMERIC_FEATURES) + len(encoder.categories_[0]))
        self.assertTrue(np.isfinite(transformed).all())

    def test_model_training_inference_and_artifact_round_trip(self):
        frame = make_frame()
        with tempfile.TemporaryDirectory() as directory:
            training_path = Path(directory) / "train.csv"
            frame.drop(columns=["__row_id"]).to_csv(training_path, index=False)
            scaler, encoder, _ = fit_preprocessing(training_path, sample_fraction=1.0)
            nodes = transform_node_features(frame, scaler, encoder)
            feature_count = nodes.shape[1]
            model = build_model(feature_count, hidden_dim=8)
            inputs = [
                nodes,
                np.zeros((4, 4, feature_count), dtype=np.float32),
                np.zeros((4, 4, EDGE_FEATURE_COUNT), dtype=np.float32),
                np.zeros((4, 4), dtype=np.float32),
            ]
            labels = frame["isFraud"].to_numpy(dtype=np.float32)
            model.compile(optimizer="adam", loss="binary_crossentropy")
            model.fit(inputs, labels, epochs=1, batch_size=2, verbose=0)
            before = predict_probabilities(model, inputs, batch_size=2)

            config = {
                "numeric_features": list(NUMERIC_FEATURES),
                "categorical_features": list(CATEGORICAL_FEATURES),
                "hidden_dim": 8,
                "validation_selected_threshold": 0.5,
                "tensorflow_version": tf.__version__,
                "training_target_rows": 4,
                "training_fraud_rows": 2,
                "context_sample_fraction": 1.0,
                "best_epoch": 1,
            }
            metrics = {
                "threshold": 0.5,
                "precision": 0.5,
                "recall": 0.5,
                "f1": 0.5,
                "average_precision": 0.5,
                "confusion_matrix": [[1, 1], [1, 1]],
            }
            results = {
                "gnn_test_metrics": metrics,
                "existing_xgboost_test_metrics": metrics,
            }
            artifact_dir = Path(directory) / "artifact"
            save_artifacts(artifact_dir, model, scaler, encoder, config, results)
            loaded = load_artifact(artifact_dir)
            after = predict_probabilities(loaded["model"], inputs, batch_size=2)

            self.assertEqual(loaded["threshold"], 0.5)
            np.testing.assert_allclose(before, after, rtol=0, atol=1e-7)


if __name__ == "__main__":
    unittest.main()