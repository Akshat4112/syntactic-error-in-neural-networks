import hashlib
import json

import pytest

from syntaxbench.validation import SPLITS, inspect_split, parse_record, write_manifest


@pytest.mark.parametrize("line", ["NN\ttext", "VBZ\t", "VBP\t  ", "VBZ", "VBZ\ta\tb"])
def test_invalid_record(line):
    with pytest.raises(ValueError):
        parse_record(line)


def test_label_mapping_and_raw_text():
    assert parse_record("VBZ\t A, cat. \n") == (0, " A, cat. ")
    assert parse_record("VBP\tCats\r\n") == (1, "Cats")


def test_manifest(tmp_path):
    raw = tmp_path / "data" / "rnn_agr_simple"
    raw.mkdir(parents=True)
    payload = b"VBZ\tA cat\nVBZ\tA cat\nVBP\tCats\n"
    for name in SPLITS.values():
        (raw / name).write_bytes(payload)
    report = json.loads(write_manifest(tmp_path).read_text())
    assert set(report["splits"]) == set(SPLITS)
    for split in report["splits"].values():
        assert split["count"] == 3
        assert split["duplicate_rows"] == 1
        assert split["label_counts"] == {"0": 2, "1": 1}
        assert split["sha256"] == hashlib.sha256(payload).hexdigest()


def test_missing_empty_invalid(tmp_path):
    path = tmp_path / "split"
    with pytest.raises(FileNotFoundError):
        inspect_split(path)
    path.write_bytes(b"")
    with pytest.raises(ValueError, match="empty split"):
        inspect_split(path)
    path.write_bytes(b"VBZ\tcat\nBAD\tdog\n")
    with pytest.raises(ValueError, match="split:2:"):
        inspect_split(path)
