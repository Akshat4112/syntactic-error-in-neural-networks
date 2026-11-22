"""Load official homogeneous attractor files as separate evaluation cohorts."""
from collections import Counter
from pathlib import Path

from .validation import SPLITS, parse_record


def load_attractor_bins(project_root):
    """Return ordered records by official file membership, not inferred syntax."""
    raw = Path(project_root) / "data" / "rnn_agr_simple"
    bins = {}
    for count in range(6):
        path = raw / SPLITS[f"test.{count}"]
        with path.open(encoding="utf-8") as stream:
            rows = tuple(parse_record(line) for line in stream)
        if not rows:
            raise ValueError(f"Empty attractor bin: {count}")
        bins[count] = rows
    return bins


def membership_report(bins):
    """Report per-file denominators and overlaps without pooling cohorts."""
    if set(bins) != set(range(6)):
        raise ValueError("Expected exactly six official attractor bins")
    counters = {k: Counter(bins[k]) for k in range(6)}
    return {
        "bins": {
            str(k): {
                "count": sum(counters[k].values()),
                "unique_records": len(counters[k]),
                "duplicate_rows": sum(counters[k].values()) - len(counters[k]),
            }
            for k in range(6)
        },
        "cross_bin_shared_unique_records": {
            f"{i}:{j}": len(counters[i].keys() & counters[j].keys())
            for i in range(6) for j in range(i + 1, 6)
        },
        "unique_records_across_bins": len(set().union(*counters.values())),
    }
