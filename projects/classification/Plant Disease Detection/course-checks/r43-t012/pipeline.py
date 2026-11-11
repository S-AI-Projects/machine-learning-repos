"""Deterministic, leakage-aware split manifest generation."""

import json
import random
from collections import defaultdict
from pathlib import Path

SPLIT_NAMES = ("train", "validation", "test")
RATIOS = (0.5, 0.25, 0.25)


def allocate_counts(total):
    raw = [total * ratio for ratio in RATIOS]
    counts = [int(value) for value in raw]
    remainder = total - sum(counts)
    order = sorted(
        range(3),
        key=lambda i: (raw[i] - counts[i], -i),
        reverse=True,
    )
    for index in order[:remainder]:
        counts[index] += 1
    return counts


def validate_records(records):
    if not records:
        raise ValueError("records must not be empty")
    ids = [row.get("record_id") for row in records]
    if any(not value for value in ids):
        raise ValueError("every record needs a record_id")
    if len(ids) != len(set(ids)):
        raise ValueError("record_id values must be unique")
    if any("label" not in row for row in records):
        raise ValueError("every record needs a label")


def group_keys(row):
    return tuple(
        (field, row[field])
        for field in ("group_id", "user_id")
        if row.get(field) is not None
    )


def check_no_leakage(splits):
    seen_ids = set()
    group_owner = {}

    for split_name in SPLIT_NAMES:
        for row in splits[split_name]:
            record_id = row["record_id"]
            if record_id in seen_ids:
                raise ValueError(f"record ID leakage: {record_id}")
            seen_ids.add(record_id)

            for group in group_keys(row):
                owner = group_owner.get(group)
                if owner is not None and owner != split_name:
                    raise ValueError(f"group/user leakage: {group}")
                group_owner[group] = split_name


def stratified_group_split(records, seed):
    groups = defaultdict(list)
    for row in records:
        key = group_keys(row) or (("record_id", row["record_id"]),)
        groups[key].append(row)

    strata = defaultdict(list)
    for group_rows in groups.values():
        signature = tuple(sorted(str(row["label"]) for row in group_rows))
        strata[signature].append(group_rows)

    output = {name: [] for name in SPLIT_NAMES}
    rng = random.Random(seed)

    for group_list in strata.values():
        rng.shuffle(group_list)
        counts = allocate_counts(len(group_list))
        start = 0
        for name, count in zip(SPLIT_NAMES, counts):
            for group_rows in group_list[start:start + count]:
                output[name].extend(group_rows)
            start += count

    return output


def chronological_split(records, time_field):
    ordered = sorted(records, key=lambda row: row[time_field])
    counts = allocate_counts(len(ordered))
    first, second, _ = counts
    return {
        "train": ordered[:first],
        "validation": ordered[first:first + second],
        "test": ordered[first + second:],
    }


def create_split_manifest(records, seed=43, time_field=None):
    validate_records(records)

    if time_field:
        if any(time_field not in row for row in records):
            raise ValueError(f"missing time field: {time_field}")
        splits = chronological_split(records, time_field)
    else:
        splits = stratified_group_split(records, seed)

    check_no_leakage(splits)

    return {
        "seed": seed,
        "strategy": "chronological" if time_field else "stratified_group",
        "time_field": time_field,
        "splits": {
            name: [row["record_id"] for row in splits[name]]
            for name in SPLIT_NAMES
        },
    }


def load_fixture(path):
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def write_manifest(fixture_path, output_path):
    fixture = load_fixture(fixture_path)
    manifest = create_split_manifest(
        fixture["records"], seed=fixture.get("seed", 43)
    )
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)
        file.write("\n")
    return manifest


if __name__ == "__main__":
    folder = Path(__file__).resolve().parent
    result = write_manifest(
        folder / "fixture.json",
        folder / "split_manifest.json",
    )
    print(json.dumps(result, indent=2))
