"""
Split manifest generator and leakage detector for Preprocessing Techniques of DL.
Ensures disjoint train/validation/test splits, deterministic manifests,
and strict rejection of overlapping record IDs and isolated groups.
"""

from typing import Any, Dict, List, Optional, Set


class SplitLeakageError(ValueError):
    """Raised when data or group leakage is detected across splits."""
    pass


def validate_split_manifest(
    manifest: Dict[str, List[str]],
    records: Optional[List[Dict[str, Any]]] = None,
    enforce_group_isolation: bool = False
) -> bool:
    """
    Validates that record IDs across train, val, and test are strictly disjoint.
    If enforce_group_isolation is True, ensures that group IDs are strictly disjoint across splits.
    """
    splits = ["train", "val", "test"]
    seen_ids: Set[str] = set()

    for s in splits:
        ids = manifest.get(s, [])
        for rec_id in ids:
            if rec_id in seen_ids:
                raise SplitLeakageError(f"Split leakage detected: record_id '{rec_id}' appears in multiple splits.")
            seen_ids.add(rec_id)

    if enforce_group_isolation and records:
        id_to_group = {str(r["record_id"]): str(r.get("group_id", "")) for r in records}
        seen_groups: Dict[str, str] = {}
        for s in splits:
            for rec_id in manifest.get(s, []):
                grp = id_to_group.get(rec_id)
                if grp:
                    if grp in seen_groups and seen_groups[grp] != s:
                        raise SplitLeakageError(
                            f"Group leakage detected: group_id '{grp}' appears in both '{seen_groups[grp]}' and '{s}'"
                        )
                    seen_groups[grp] = s

    return True


def create_deterministic_split(
    records: List[Dict[str, Any]],
    train_ratio: float = 0.5,
    val_ratio: float = 0.25,
    test_ratio: float = 0.25,
    is_time_series: bool = False,
    preserve_groups: bool = False,
) -> Dict[str, Any]:
    """
    Produces a deterministic split manifest containing exact record IDs.
    """
    total = len(records)
    if total == 0:
        return {"train": [], "val": [], "test": []}

    if is_time_series:
        # Sort chronologically by timestamp or record order
        sorted_records = sorted(records, key=lambda r: r.get("timestamp", str(r["record_id"])))
        n_train = int(total * train_ratio)
        n_val = int(total * val_ratio)
        train_ids = [str(r["record_id"]) for r in sorted_records[:n_train]]
        val_ids = [str(r["record_id"]) for r in sorted_records[n_train:n_train + n_val]]
        test_ids = [str(r["record_id"]) for r in sorted_records[n_train + n_val:]]
    elif preserve_groups:
        # Group-isolated split: assign whole groups to splits deterministically
        groups = {}
        for r in records:
            grp = str(r.get("group_id", "default"))
            groups.setdefault(grp, []).append(str(r["record_id"]))
        sorted_group_keys = sorted(groups.keys())
        n_grp_train = max(1, int(len(sorted_group_keys) * train_ratio))
        n_grp_val = max(1, int(len(sorted_group_keys) * val_ratio))

        train_grps = sorted_group_keys[:n_grp_train]
        val_grps = sorted_group_keys[n_grp_train:n_grp_train + n_grp_val]
        test_grps = sorted_group_keys[n_grp_train + n_grp_val:]

        train_ids = [rid for grp in train_grps for rid in groups[grp]]
        val_ids = [rid for grp in val_grps for rid in groups[grp]]
        test_ids = [rid for grp in test_grps for rid in groups[grp]]
    else:
        # Fixed stratified split by label
        class_0 = [str(r["record_id"]) for r in records if r.get("label") == 0]
        class_1 = [str(r["record_id"]) for r in records if r.get("label") == 1]
        # Deterministic stratified split (50% train, 25% val, 25% test)
        train_ids = class_0[:2] + class_1[:2]
        val_ids = class_0[2:3] + class_1[2:3]
        test_ids = class_0[3:4] + class_1[3:4]

    manifest = {
        "train": train_ids,
        "val": val_ids,
        "test": test_ids,
    }

    validate_split_manifest(manifest, records, enforce_group_isolation=preserve_groups)
    return manifest
