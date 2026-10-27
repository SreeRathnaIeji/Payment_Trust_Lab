import csv
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "frontend" / "paysim_labelled_examples.json"
TEST_SPLIT = ROOT / "Dataset (PaySim)" / "chronological_splits" / "test.csv"
TRANSACTION_FIELDS = (
    "step",
    "type",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
)


class LabelledPaySimExampleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
        cls.examples = cls.fixture["examples"]

    def test_fixture_is_balanced_and_ground_truth_is_not_a_transaction_feature(self):
        self.assertEqual(len(self.examples), 8)
        self.assertEqual(sum(example["ground_truth"] == 1 for example in self.examples), 4)
        self.assertEqual(sum(example["ground_truth"] == 0 for example in self.examples), 4)
        self.assertEqual(len({example["id"] for example in self.examples}), 8)
        for example in self.examples:
            self.assertIn(example["ground_truth"], (0, 1))
            self.assertEqual(set(example["transaction"]), set(TRANSACTION_FIELDS))
            self.assertNotIn("isFraud", example["transaction"])
            self.assertNotIn("ground_truth", example["transaction"])

    def test_examples_preserve_exact_test_split_features_and_labels(self):
        expected_by_index = {
            int(example["id"].rsplit("-", maxsplit=1)[1]): example
            for example in self.examples
        }
        found = set()
        with TEST_SPLIT.open(newline="", encoding="utf-8") as csv_file:
            for row_index, row in enumerate(csv.DictReader(csv_file)):
                example = expected_by_index.get(row_index)
                if example is None:
                    continue
                found.add(row_index)
                self.assertEqual(int(row["isFraud"]), example["ground_truth"])
                for field in TRANSACTION_FIELDS:
                    if field == "type":
                        actual = row[field]
                    elif field == "step":
                        actual = int(row[field])
                    else:
                        actual = float(row[field])
                    self.assertEqual(actual, example["transaction"][field])

        self.assertEqual(found, set(expected_by_index))


if __name__ == "__main__":
    unittest.main()