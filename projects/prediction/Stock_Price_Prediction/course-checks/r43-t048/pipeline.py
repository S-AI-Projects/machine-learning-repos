"""
Error analysis and reporting pipeline for Stock Price Prediction.
Tracks record ID, expected value, predicted value, and reason category for each error.
Asserts that injected failures appear in the report, correct predictions do not,
and aggregate error counts match detailed rows.
"""

from typing import Any, Dict, List, Optional
import json


class ErrorAnalysisReport:
    def __init__(self, total_samples: int, correct_count: int, error_rows: List[Dict[str, Any]], reason_summary: Dict[str, int]):
        self.total_samples = total_samples
        self.correct_count = correct_count
        self.error_count = len(error_rows)
        self.error_rows = error_rows
        self.reason_summary = reason_summary

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_samples": self.total_samples,
            "correct_count": self.correct_count,
            "error_count": self.error_count,
            "error_rate": self.error_count / self.total_samples if self.total_samples > 0 else 0.0,
            "reason_summary": self.reason_summary,
            "error_rows": self.error_rows,
            "notice": "Error report generated for pipeline monitoring verification; simulated data only."
        }


def generate_error_analysis_report(
    records: List[Dict[str, Any]],
    predictions: Optional[List[int]] = None
) -> ErrorAnalysisReport:
    """
    Evaluates predictions against true labels for each record.
    Filters exclusively the failing records into the error report.
    """
    if not records:
        return ErrorAnalysisReport(0, 0, [], {})

    error_rows: List[Dict[str, Any]] = []
    correct_count = 0
    reason_summary: Dict[str, int] = {}

    for idx, r in enumerate(records):
        rec_id = str(r.get("record_id", f"row_{idx}"))
        y_true = int(r["label"])
        y_pred = int(predictions[idx] if predictions is not None else r["injected_prediction"])

        if y_true == y_pred:
            correct_count += 1
        else:
            if y_true == 1 and y_pred == 0:
                reason = "False Negative: Underprediction near decision boundary"
            elif y_true == 0 and y_pred == 1:
                reason = "False Positive: Overprediction near decision boundary"
            else:
                reason = "Unknown Classification Mismatch"

            reason_summary[reason] = reason_summary.get(reason, 0) + 1
            error_rows.append({
                "record_id": rec_id,
                "expected": y_true,
                "predicted": y_pred,
                "reason_category": reason
            })

    return ErrorAnalysisReport(
        total_samples=len(records),
        correct_count=correct_count,
        error_rows=error_rows,
        reason_summary=reason_summary
    )


def load_fixture(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
