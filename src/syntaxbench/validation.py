"""Validate official two-column agreement data without normalizing text."""
import hashlib
import json
from pathlib import Path

LABELS = {"VBZ": 0, "VBP": 1}
SPLITS = {"train": "numpred.train", "validation": "numpred.val",
          "test": "numpred.test"}
SPLITS.update({f"test.{i}": f"numpred.test.{i}" for i in range(6)})


def parse_record(line):
    fields = line.rstrip("\r\n").split("\t")
    if len(fields) != 2 or fields[0] not in LABELS:
        raise ValueError("Expected a known POS label and one tab-separated preamble")
    if not fields[1].strip():
        raise ValueError("Missing preamble")
    return LABELS[fields[0]], fields[1]


def inspect_split(path):
    checksum = hashlib.sha256()
    seen = set()
    counts = {"0": 0, "1": 0}
    duplicates = 0
    with Path(path).open("rb") as stream:
        for number, raw in enumerate(stream, 1):
            checksum.update(raw)
            try:
                record = parse_record(raw.decode("utf-8"))
            except ValueError as exc:
                raise ValueError(f"{Path(path).name}:{number}: {exc}") from exc
            counts[str(record[0])] += 1
            if record in seen:
                duplicates += 1
            seen.add(record)
    if not sum(counts.values()):
        raise ValueError(f"{Path(path).name}: empty split")
    return {"sha256": checksum.hexdigest(), "count": sum(counts.values()),
            "label_counts": counts, "duplicate_rows": duplicates}


def validate_dataset(project_root):
    data = Path(project_root) / "data"
    raw = data / "rnn_agr_simple"
    report = {"schema_version": 1, "label_mapping": LABELS,
              "splits": {name: {"file": filename, **inspect_split(raw / filename)}
                         for name, filename in SPLITS.items()}}
    return report


def write_manifest(project_root):
    report = validate_dataset(project_root)
    path = Path(project_root) / "data" / "validation-manifest.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")
    return path
