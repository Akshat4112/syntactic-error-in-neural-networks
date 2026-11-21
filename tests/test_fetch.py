import io
import tarfile

import pytest

from syntaxbench import fetch


def make_archive(path, name="rnn_agr_simple/numpred.train"):
    with tarfile.open(path, "w:gz") as tar:
        payload = b"VBZ\\tThe cat\\n"
        member = tarfile.TarInfo(name)
        member.size = len(payload)
        tar.addfile(member, io.BytesIO(payload))


def test_fetch_offline_idempotent(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    archive = data / "rnn_agr_simple.tar.gz"
    make_archive(archive)
    sha = fetch.digest(archive)
    dest = fetch.fetch_dataset(tmp_path, sha, offline=True)
    assert (dest / "numpred.train").is_file()
    assert fetch.fetch_dataset(tmp_path, sha, offline=True) == dest
    (dest / "numpred.train").write_text("corrupted")
    with pytest.raises(ValueError, match="checksum"):
        fetch.fetch_dataset(tmp_path, sha, offline=True)


def test_mock_download(tmp_path, monkeypatch):
    archive = tmp_path / "fixture.tar.gz"
    make_archive(archive)
    monkeypatch.setattr(fetch, "urlopen", lambda *a, **kw: archive.open("rb"))
    dest = fetch.fetch_dataset(tmp_path, fetch.digest(archive))
    assert (dest / "numpred.train").is_file()


def test_bad_digest(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    make_archive(data / "rnn_agr_simple.tar.gz")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        fetch.fetch_dataset(tmp_path, "0" * 64, offline=True)


def test_traversal(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    archive = data / "rnn_agr_simple.tar.gz"
    make_archive(archive, "rnn_agr_simple/../../escape")
    with pytest.raises(ValueError, match="Unsafe"):
        fetch.fetch_dataset(tmp_path, fetch.digest(archive), offline=True)
    assert not (tmp_path / "escape").exists()
    assert not (data / "rnn_agr_simple").exists()


def test_offline_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        fetch.fetch_dataset(tmp_path, "0" * 64, offline=True)
