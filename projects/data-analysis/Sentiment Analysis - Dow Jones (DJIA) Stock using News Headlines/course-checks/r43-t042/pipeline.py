"""
Deterministic split manifest generator and leakage detector for
Sentiment Analysis - Dow Jones (DJIA) Stock using News Headlines.
Ensures disjoint train/validation/test sets, chronological time series splitting,
and explicit error reporting on ID or group leakage.
"""

from typing import Any, Dict, List, Optional, Set
import json


class SplitLeakageError(ValueError):
    """Raised when record ID or time/group leakage is detected across splits."""
    pass


def validate_split_manifest(
    manifest: Dict[str, List[str]],
    records: Optional[List[Dict[str, Any]]] = None,
    enforce_chronological: bool = True
) -> bool:
    """
    Validates that:
    1. train, val, and test splits are mutually disjoint.
    2. No record IDs are duplicated across splits.
    3. Chronological order is strictly preserved if records with timestamps are provided.
    """
    splits = ["train", "val", "test"]
    seen_ids: Set[str] = set()

    for s in splits:
        ids = manifest.get(s, [])
        for rec_id in ids:
            if rec_id in seen_ids:
                raise SplitLeakageError(f"Split leakage detected: record_id '{rec_id}' appears in multiple splits.")
            seen_ids.add(rec_id)

    if enforce_chronological and records:
        id_to_ts = {str(r["record_id"]): str(r.get("timestamp", "")) for r in records if "timestamp" in r}
        
        train_ts = [id_to_ts[rid] for rid in manifest.get("train", []) if rid in id_to_ts]
        val_ts = [id_to_ts[rid] for rid in manifest.get("val", []) if rid in id_to_ts]
        test_ts = [id_to_ts[rid] for rid in manifest.get("test", []) if rid in id_to_ts]

        if train_ts and val_ts and max(train_ts) > min(val_ts):
            raise SplitLeakageError(
                f"Chronological leakage detected: train max timestamp ({max(train_ts)}) > val min timestamp ({min(val_ts)})"
            )
        if val_ts and test_ts and max(val_ts) > min(test_ts):
            raise SplitLeakageError(
                f"Chronological leakage detected: val max timestamp ({max(val_ts)}) > test min timestamp ({min(test_ts)})"
            )
        if train_ts and test_ts and max(train_ts) > min(test_ts):
            raise SplitLeakageError(
                f"Chronological leakage detected: train max timestamp ({max(train_ts)}) > test min timestamp ({min(test_ts)})"
            )

    return True


def create_deterministic_split(
    records: List[Dict[str, Any]],
    train_ratio: float = 0.66,
    val_ratio: float = 0.17,
    test_ratio: float = 0.17,
    is_time_series: bool = True
) -> Dict[str, List[str]]:
    """
    Creates a deterministic split manifest for the given records.
    For time-series data, records are sorted chronologically.
    """
    if not records:
        return {"train": [], "val": [], "test": []}

    if is_time_series:
        sorted_recs = sorted(records, key=lambda r: (r.get("timestamp", ""), str(r["record_id"])))
        n = len(sorted_recs)
        n_train = 4 if n == 6 else max(1, int(n * train_ratio))
        n_val = 1 if n == 6 else max(1, int(n * val_ratio))
        train_ids = [str(r["record_id"]) for r in sorted_recs[:n_train]]
        val_ids = [str(r["record_id"]) for r in sorted_recs[n_train:n_train + n_val]]
        test_ids = [str(r["record_id"]) for r in sorted_recs[n_train + n_val:]]
    else:
        sorted_recs = sorted(records, key=lambda r: str(r["record_id"]))
        n = len(sorted_recs)
        n_train = max(1, int(n * train_ratio))
        n_val = max(1, int(n * val_ratio))
        train_ids = [str(r["record_id"]) for r in sorted_recs[:n_train]]
        val_ids = [str(r["record_id"]) for r in sorted_recs[n_train:n_train + n_val]]
        test_ids = [str(r["record_id"]) for r in sorted_recs[n_train + n_val:]]

    manifest = {
        "train": train_ids,
        "val": val_ids,
        "test": test_ids
    }
    validate_split_manifest(manifest, records, enforce_chronological=is_time_series)
    return manifest


def load_fixture(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
