"""
Explicit numeric preprocessor for Blood Glucose Level Prediction.
Fits scaling parameters exclusively on training rows and enforces feature ordering,
label set validation, and strict missing-value policies.
"""

from typing import Any, Dict, List, Optional, Tuple


class PreprocessingContaminationError(ValueError):
    """Raised when contamination or policy violations occur in preprocessing."""
    pass


class ExplicitNumericPreprocessor:
    """
    Fits min-max normalization on training records only.
    Prevents leakage into test/evaluation data through the public transform interface.
    """
    EXPECTED_FEATURES = ["x1", "x2"]
    ALLOWED_LABELS = {0, 1}

    def __init__(self, missing_value_policy: str = "reject"):
        self.missing_value_policy = missing_value_policy
        self.is_fitted = False
        self.params: Dict[str, Dict[str, float]] = {}
        self.train_record_ids: List[str] = []

    def validate_record_schema(self, r: Dict[str, Any], check_label: bool = False):
        for feat in self.EXPECTED_FEATURES:
            if feat not in r or r[feat] is None:
                raise PreprocessingContaminationError(
                    f"Missing required feature '{feat}' in record {r.get('record_id', 'unknown')}"
                )
            if not isinstance(r[feat], (int, float)) or isinstance(r[feat], bool):
                raise PreprocessingContaminationError(
                    f"Malformed numeric feature '{feat}' in record {r.get('record_id', 'unknown')}: got {type(r[feat]).__name__}"
                )

        if check_label and "label" in r:
            lbl = r["label"]
            if lbl not in self.ALLOWED_LABELS or isinstance(lbl, bool):
                raise PreprocessingContaminationError(
                    f"Invalid label {repr(lbl)} in record {r.get('record_id', 'unknown')}; expected {self.ALLOWED_LABELS}"
                )

    def fit(self, train_records: List[Dict[str, Any]]) -> "ExplicitNumericPreprocessor":
        if not train_records:
            raise ValueError("Training records cannot be empty.")

        for r in train_records:
            self.validate_record_schema(r, check_label=True)

        self.params = {}
        for feat in self.EXPECTED_FEATURES:
            vals = [float(r[feat]) for r in train_records]
            min_val = min(vals)
            max_val = max(vals)
            self.params[feat] = {
                "min": min_val,
                "max": max_val,
                "scale": (max_val - min_val) if max_val != min_val else 1.0
            }

        self.train_record_ids = [str(r["record_id"]) for r in train_records]
        self.is_fitted = True
        return self

    def transform(self, records: List[Dict[str, Any]]) -> List[List[float]]:
        """
        Transforms records using previously fitted parameters.
        Does NOT alter or refit internal parameters under any circumstance.
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted on training rows before calling transform().")

        transformed_matrix: List[List[float]] = []

        for r in records:
            self.validate_record_schema(r, check_label=False)
            row_features: List[float] = []
            for feat in self.EXPECTED_FEATURES:
                val = float(r[feat])
                f_min = self.params[feat]["min"]
                f_scale = self.params[feat]["scale"]
                scaled_val = (val - f_min) / f_scale
                row_features.append(float(scaled_val))
            transformed_matrix.append(row_features)

        return transformed_matrix

    def fit_transform(self, train_records: List[Dict[str, Any]]) -> List[List[float]]:
        self.fit(train_records)
        return self.transform(train_records)
