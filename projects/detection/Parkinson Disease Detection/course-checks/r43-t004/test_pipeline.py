"""
Unit tests for R43-T004 metric correctness.
NOTE: SYNTHETIC FIXTURE only. NOT clinical evidence.
"""
import unittest
from pipeline import precision, recall, accuracy, compute_metrics, FIXTURE_RECORDS


class TestExplicitMetricExample(unittest.TestCase):
    """Spec example: truth=[1,1,0,0] / pred=[1,0,1,0] -> precision=recall=0.5"""
    def test_precision_half(self):
        self.assertAlmostEqual(precision([1,1,0,0], [1,0,1,0]), 0.5)

    def test_recall_half(self):
        self.assertAlmostEqual(recall([1,1,0,0], [1,0,1,0]), 0.5)

    def test_accuracy_half(self):
        self.assertAlmostEqual(accuracy([1,1,0,0], [1,0,1,0]), 0.5)


class TestZeroDenominator(unittest.TestCase):
    def test_precision_no_positives_predicted(self):
        self.assertEqual(precision([0,0],[0,0], zero_division=0.0), 0.0)

    def test_recall_no_positives_in_truth(self):
        self.assertEqual(recall([0,0],[1,1], zero_division=0.0), 0.0)


class TestPerfectPredictions(unittest.TestCase):
    def test_perfect_precision_recall(self):
        truth = [1, 0, 1]; pred = [1, 0, 1]
        self.assertAlmostEqual(precision(truth, pred), 1.0)
        self.assertAlmostEqual(recall(truth, pred), 1.0)
        self.assertAlmostEqual(accuracy(truth, pred), 1.0)


class TestFixture(unittest.TestCase):
    def test_eight_records(self):
        self.assertEqual(len(FIXTURE_RECORDS), 8)

    def test_x1_values(self):
        self.assertEqual([r["x1"] for r in FIXTURE_RECORDS], list(range(8)))

    def test_label_balance(self):
        labels = [r["label"] for r in FIXTURE_RECORDS]
        self.assertEqual(labels.count(0), 4)
        self.assertEqual(labels.count(1), 4)


class TestSentinel(unittest.TestCase):
    def test_nonzero_test_count(self):
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromModule(__import__(__name__))
        self.assertGreater(suite.countTestCases(), 0)

    def test_deliberate_wrong_expected_fails(self):
        self.assertNotAlmostEqual(precision([1,1,0,0],[1,0,1,0]), 0.9999)


if __name__ == "__main__":
    unittest.main()
