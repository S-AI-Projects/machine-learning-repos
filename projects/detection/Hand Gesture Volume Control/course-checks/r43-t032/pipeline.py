from collections import defaultdict


def manifest(rows):
    groups = defaultdict(list)
    ids = set()
    if any('time' in row for row in rows) and not all('time' in row for row in rows):
        raise ValueError('time must be present on every row')
    for row in rows:
        record_id, group_id, label = row['record_id'], row['group_id'], row['label']
        if record_id in ids:
            raise ValueError('duplicate record ID')
        ids.add(record_id)
        if label not in (0, 1):
            raise ValueError('unexpected label')
        groups[group_id].append(row)
    if all('time' in row for row in rows):
        # A time-bearing fixture uses whole groups in chronological order.
        ordered = sorted(groups, key=lambda g: min(row['time'] for row in groups[g]))
        if any(max(row['time'] for row in groups[a]) > min(row['time'] for row in groups[b])
               for a, b in zip(ordered, ordered[1:])):
            raise ValueError('overlapping group time ranges')
        if len(ordered) < 3:
            raise ValueError('need three chronological groups')
        cuts = (len(ordered) // 2, len(ordered) - 1)
        parts = [ordered[:cuts[0]], ordered[cuts[0]:cuts[1]], ordered[cuts[1]:]]
    else:
        by_label = defaultdict(list)
        for group_id, items in groups.items():
            labels = {row['label'] for row in items}
            if len(labels) != 1:
                raise ValueError('mixed labels within group')
            by_label[items[0]['label']].append(group_id)
        if set(by_label) != {0, 1} or any(len(v) < 3 for v in by_label.values()):
            raise ValueError('need three groups per class')
        parts = [[], [], []]
        for label in (0, 1):
            ordered = sorted(by_label[label])
            parts[0].extend(ordered[:len(ordered)-2])
            parts[1].append(ordered[-2])
            parts[2].append(ordered[-1])
    result = {name: sorted(row['record_id'] for group in part for row in groups[group])
              for name, part in zip(('train', 'validation', 'test'), parts)}
    validate_manifest(rows, result)
    return result


def validate_manifest(rows, split):
    if set(split) != {'train', 'validation', 'test'}:
        raise ValueError('split names')
    all_ids = [row['record_id'] for row in rows]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError('duplicate record IDs')
    assigned = [record_id for part in split.values() for record_id in part]
    if len(assigned) != len(set(assigned)) or set(assigned) != set(all_ids):
        raise ValueError('overlapping or missing record IDs')
    owner = {record_id: name for name, part in split.items() for record_id in part}
    group_splits = defaultdict(set)
    for row in rows:
        group_splits[row['group_id']].add(owner[row['record_id']])
    if any(len(names) > 1 for names in group_splits.values()):
        raise ValueError('group leakage')
    if all('time' in row for row in rows):
        for a, b in (('train', 'validation'), ('validation', 'test')):
            a_times = [row['time'] for row in rows if owner[row['record_id']] == a]
            b_times = [row['time'] for row in rows if owner[row['record_id']] == b]
            if a_times and b_times and max(a_times) > min(b_times):
                raise ValueError('nonchronological split')
