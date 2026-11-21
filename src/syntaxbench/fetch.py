"""Explicit, checksum-gated retrieval of the Linzen simple dataset."""
import hashlib
import re
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from urllib.request import urlopen

SOURCE = "https://tallinzen.net/media/rnn_agreement/rnn_agr_simple.tar.gz"
# Upstream README revision: 0001b91aab94c0f834f354a04ed22ba15cefd8da.
# Observed from HTTPS upstream in Actions run 44922540789 (2026-11-21).
SOURCE_SHA256 = "3e0e0c194696dc53378791ac9a75a35d134f4de0dc9253de7cc0a5c8e48b3142"
MAX_BYTES = 2 * 1024 ** 3


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def copy_bounded(source, target, limit):
    total = 0
    while chunk := source.read(1024 * 1024):
        total += len(chunk)
        if total > limit:
            raise ValueError("Dataset size limit exceeded")
        target.write(chunk)


def extract(archive, destination):
    seen = set()
    total = 0
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            path = PurePosixPath(member.name)
            # Ignore only regular AppleDouble metadata in the expected tree.
            metadata_path = any((
                path.parts == ("._rnn_agr_simple",),
                all((path.parts[:1] == ("rnn_agr_simple",), path.name.startswith("._"))),
            ))
            if all((
                not path.is_absolute(),
                ".." not in path.parts,
                "\\" not in member.name,
                member.isfile(),
                metadata_path,
            )):
                continue
            unsafe = any((
                path.is_absolute(),
                ".." in path.parts,
                path.parts[:1] != ("rnn_agr_simple",),
                "\\" in member.name,
                path in seen,
            ))
            if unsafe:
                raise ValueError(f"Unsafe archive member: {member.name}")
            seen.add(path)
            target = destination.joinpath(*path.parts)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                total += member.size
                if total > MAX_BYTES:
                    raise ValueError("Extracted dataset size limit exceeded")
                target.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(member) as source, target.open("xb") as out:
                    copy_bounded(source, out, member.size)
            else:
                raise ValueError(f"Unsupported archive member: {member.name}")


def fetch_dataset(project_root, sha256=None, offline=False):
    if sha256 is None:
        sha256 = SOURCE_SHA256
    if not re.fullmatch(r"[a-fA-F0-9]{64}", sha256):
        raise ValueError("A trusted SHA-256 archive digest is required")
    sha256 = sha256.lower()
    data = Path(project_root).resolve() / "data"
    data.mkdir(parents=True, exist_ok=True)
    cache = data / "rnn_agr_simple.tar.gz"
    dest = data / "rnn_agr_simple"
    with tempfile.TemporaryDirectory(dir=data) as tmp:
        tmp = Path(tmp)
        archive = cache
        if cache.is_symlink():
            raise ValueError("Archive cache must not be a symlink")
        if not cache.exists():
            if offline:
                raise FileNotFoundError("Verified archive required in offline mode")
            archive = tmp / "download.tar.gz"
            with urlopen(SOURCE, timeout=60) as response, archive.open("xb") as out:
                copy_bounded(response, out, MAX_BYTES)
        if digest(archive) != sha256:
            raise ValueError("Archive SHA-256 mismatch")
        extract(archive, tmp)
        extracted = tmp / dest.name
        if not extracted.is_dir():
            raise ValueError("Dataset directory missing")
        if dest.exists() or dest.is_symlink():
            if dest.is_symlink() or not dest.is_dir():
                raise ValueError("Invalid existing dataset directory")
            expected = {p.relative_to(extracted) for p in extracted.rglob("*") if p.is_file()}
            actual = {p.relative_to(dest) for p in dest.rglob("*") if p.is_file()}
            if expected != actual or any(p.is_symlink() for p in dest.rglob("*")):
                raise ValueError("Existing dataset differs from verified archive")
            if any(digest(dest / p) != digest(extracted / p) for p in expected):
                raise ValueError("Existing dataset checksum mismatch")
        else:
            extracted.rename(dest)
        if archive != cache:
            archive.rename(cache)
    return dest
