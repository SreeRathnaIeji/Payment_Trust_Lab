import json
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from Experiments.compare_four_models import (
    MODEL_FILES,
    build_models,
    fit_preprocessing,
    load_artifacts,
    majority_vote,
    predict_scores,
    transform_features,
)


def sample_frame():
    rng = np.random.default_rng(19)
    rows = 64
    return pd.DataFrame(
        {
            "step": rng.integers(1, 40, rows),
            "amount": rng.uniform(1, 1000, rows),
            "oldbalanceOrg": rng.uniform(0, 2000, rows),
            "newbalanceOrig": rng.uniform(0, 2000, rows),
            "oldbalanceDest": rng.uniform(0, 2000, rows),
            "newbalanceDest": rng.uniform(0, 2000, rows),
            "type": np.where(np.arange(rows) % 2, "PAYMENT", "TRANSFER"),
            "isFraud": (np.arange(rows) % 4 == 0).astype(np.int8),
        }
    )


class FourModelComparisonTests(unittest.TestCase):
    def test_preprocessing_is_consistent_and_fitted_only_on_training_rows(self):
        training = sample_frame().iloc[:48]
        holdout = sample_frame().iloc[48:].copy()
        holdout["type"] = "UNSEEN_TYPE"

        scaler, encoder = fit_preprocessing(training)
        first = transform_features(holdout, scaler, encoder)
        second = transform_features(holdout, scaler, encoder)

        np.testing.assert_array_equal(first, second)
        np.testing.assert_allclose(
            scaler.mean_, training[list(scaler.feature_names_in_)].mean().to_numpy()
        )
        self.assertNotIn("UNSEEN_TYPE", encoder.categories_[0])
        self.assertEqual(first.shape[1], len(scaler.feature_names_in_) + 2)

    def test_majority_vote_requires_three_votes_and_resolves_ties_as_legitimate(self):
        votes = np.array(
            [
                [1, 1, 0, 0],
                [1, 1, 1, 0],
                [1, 1, 1, 1],
                [0, 0, 1, 0],
            ],
            dtype=np.int8,
        )
        predictions, counts, ties = majority_vote(votes)
        np.testing.assert_array_equal(predictions, [0, 1, 1, 0])
        np.testing.assert_array_equal(counts, [2, 3, 4, 1])
        np.testing.assert_array_equal(ties, [True, False, False, False])

    def test_majority_vote_rejects_invalid_vote_shapes_and_values(self):
        with self.assertRaises(ValueError):
            majority_vote(np.zeros((2, 3), dtype=np.int8))
        with self.assertRaises(ValueError):
            majority_vote(np.array([[0, 1, 1, 2]]))

    def test_predictions_are_deterministic_and_artifacts_reload(self):
        training = sample_frame()
        labels = training["isFraud"].to_numpy()
        scaler, encoder = fit_preprocessing(training)
        features = transform_features(training, scaler, encoder)
        models = build_models(3.0, n_estimators=4)
        for name, model in models.items():
            if name == "Isolation Forest":
                model.fit(features[labels == 0])
            else:
                model.fit(features, labels)

        bundle = {"scaler": scaler, "encoder": encoder, "models": models}
        before = predict_scores(training, bundle)
        again = predict_scores(training, bundle)
        for name in before:
            np.testing.assert_array_equal(before[name], again[name])

        config = {
            "numeric_features": list(scaler.feature_names_in_),
            "categorical_features": ["type"],
        }
        with tempfile.TemporaryDirectory() as directory:
            artifact_dir = Path(directory)
            joblib.dump(scaler, artifact_dir / "numeric_scaler.joblib")
            joblib.dump(encoder, artifact_dir / "type_encoder.joblib")
            for name, filename in MODEL_FILES.items():
                joblib.dump(models[name], artifact_dir / filename)
            (artifact_dir / "config.json").write_text(
                json.dumps(config), encoding="utf-8"
            )

            loaded = load_artifacts(artifact_dir)
            after = predict_scores(training, loaded)
            for name in before:
                np.testing.assert_array_equal(before[name], after[name])


if __name__ == "__main__":
    unittest.main()