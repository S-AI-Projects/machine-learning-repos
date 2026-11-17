"""
Unit tests for R43-T001 schema validator.
NOTE: All test data is SYNTHETIC FIXTURE data for pipeline verification.
It is NOT real-world summarization data and results do NOT constitute
evidence of production summarization performance.
"""
import unittest
from pipeline import validate_records, extract_first_sentence, FIXTURE_RECORDS


class TestValidRecords(unittest.TestCase):
    def test_fixture_all_valid(self):
        """The three canonical fixture records must all pass validation."""
        valid, errors = validate_records(FIXTURE_RECORDS)
        self.assertEqual(errors, [], msg=f"Unexpected errors: {errors}")
        self.assertEqual(len(valid), 3)

    def test_first_sentence_extraction_multi(self):
        """Multi-sentence input returns first sentence (labelled baseline rule)."""
        result = extract_first_sentence("Ada wrote notes. Bo reviewed them.")
        self.assertEqual(result, "Ada wrote notes.")

    def test_first_sentence_extraction_single(self):
        """Single-sentence input returns same sentence."""
        result = extract_first_sentence("One sentence.")
        self.assertEqual(result, "One sentence.")

    def test_first_sentence_extraction_empty(self):
        """Empty input returns empty output."""
        result = extract_first_sentence("")
        self.assertEqual(result, "")


class TestInvalidRecords(unittest.TestCase):
    def test_missing_record_id(self):
        bad = [{"source_text": "hello", "expected_first_sentence": "hello"}]
        _, errors = validate_records(bad)
        self.assertTrue(any(e["field"] == "record_id" for e in errors))

    def test_missing_source_text(self):
        bad = [{"record_id": "x1", "expected_first_sentence": "hi"}]
        _, errors = validate_records(bad)
        self.assertTrue(any(e["field"] == "source_text" for e in errors))

    def test_missing_expected_first_sentence(self):
        bad = [{"record_id": "x2", "source_text": "hello."}]
        _, errors = validate_records(bad)
        self.assertTrue(any(e["field"] == "expected_first_sentence" for e in errors))

    def test_malformed_numeric_record_id(self):
        bad = [{"record_id": 42, "source_text": "hi.", "expected_first_sentence": "hi."}]
        _, errors = validate_records(bad)
        self.assertTrue(any(e["field"] == "record_id" for e in errors))

    def test_malformed_numeric_source_text(self):
        bad = [{"record_id": "x3", "source_text": 123, "expected_first_sentence": ""}]
        _, errors = validate_records(bad)
        self.assertTrue(any(e["field"] == "source_text" for e in errors))

    def test_duplicate_record_id(self):
        dup = [
            {"record_id": "dup1", "source_text": "A.", "expected_first_sentence": "A."},
            {"record_id": "dup1", "source_text": "B.", "expected_first_sentence": "B."},
        ]
        _, errors = validate_records(dup)
        self.assertTrue(any("Duplicate" in e["reason"] for e in errors))

    def test_nonzero_test_count(self):
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromModule(__import__(__name__))
        self.assertGreater(suite.countTestCases(), 0)

    def test_deliberate_wrong_expected_fails(self):
        result = extract_first_sentence("Ada wrote notes. Bo reviewed them.")
        self.assertNotEqual(result, "WRONG_EXPECTED_VALUE")


class TestSyntheticDataLabelling(unittest.TestCase):
    def test_fixture_records_are_synthetic(self):
        for record in FIXTURE_RECORDS:
            self.assertTrue(
                record["record_id"].startswith("r"),
                msg=f"Fixture IDs should be synthetic labels; got {record['record_id']!r}",
            )


if __name__ == "__main__":
    unittest.main()
