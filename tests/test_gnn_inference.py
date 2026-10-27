import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from backend.gnn_inference import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    load_node_only_bundle,
    predict_node_only,
    prepare_node_features,
)


class FixedModel:
    def __init__(self, score):
        self.score = score

    def __call__(self, inputs, training=False):
        return np.asarray([[self.score]], dtype=np.float32)


def example_transaction():
    return {
        "step": 500,
        "amount": 25000.0,
        "oldbalanceOrg": 30000.0,
        "newbalanceOrig": 5000.0,
        "oldbalanceDest": 1000.0,
        "newbalanceDest": 26000.0,
        "type": "CASH_OUT",
    }


def test_bundle():
    training = pd.DataFrame(
        {
            "step": [1, 2, 3],
            "amount": [10.0, 20.0, 30.0],
            "oldbalanceOrg": [100.0, 200.0, 300.0],
            "newbalanceOrig": [90.0, 180.0, 270.0],
            "oldbalanceDest": [50.0, 60.0, 70.0],
            "newbalanceDest": [60.0, 80.0, 100.0],
        }
    )
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore").fit(
        pd.DataFrame({"type": ["CASH_IN", "CASH_OUT", "PAYMENT"]})
    )
    return {
        "scaler": StandardScaler().fit(training[list(NUMERIC_FEATURES)]),
        "encoder": encoder,
        "model": FixedModel(0.8),
        "threshold": 0.7,
        "variant": "experimental-node-feature-only-gnn-v2",
    }


class GNNInferenceTests(unittest.TestCase):
    def test_preprocessing_uses_saved_scaler_encoder_and_feature_order(self):
        bundle = test_bundle()
        features = prepare_node_features(example_transaction(), bundle)
        expected_width = len(NUMERIC_FEATURES) + len(bundle["encoder"].categories_[0])
        self.assertEqual(features.shape, (1, expected_width))
        self.assertTrue(np.isfinite(features).all())

    def test_threshold_decisions_use_validation_selected_cutoff(self):
        bundle = test_bundle()
        bundle["model"] = FixedModel(0.7)
        result = predict_node_only(example_transaction(), bundle)
        self.assertEqual(result["decision"], "FRAUD")
        self.assertTrue(result["is_fraud"])
        self.assertEqual(result["threshold"], 0.7)
        self.assertAlmostEqual(result["raw_score"], 0.7, places=6)

        bundle["model"] = FixedModel(0.69)
        result = predict_node_only(example_transaction(), bundle)
        self.assertEqual(result["decision"], "NOT FRAUD")
        self.assertFalse(result["is_fraud"])

    def test_inference_rejects_graph_enabled_bundle(self):
        bundle = test_bundle()
        bundle["variant"] = "graph-enabled-gnn"
        with self.assertRaises(ValueError):
            predict_node_only(example_transaction(), bundle)

    def test_missing_bundle_artifacts_fail_with_clear_error(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "Missing node-only GNN artifact"):
                load_node_only_bundle(Path(directory))


if __name__ == "__main__":
    unittest.main()