"""Downloads with checks, streaming hashes, and archive extraction that can't escape its folder.

Standard library only. Downloads go through ``curl -L -C -`` so a broken transfer resumes; the size and
checksums are verified before a file is used. Archive members with absolute paths, ``..`` parts or links
are refused rather than extracted.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import tarfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import IO, Iterable

from . import paths

CHUNK = 8 * 1024 * 1024


class ChecksumError(RuntimeError):
    """A downloaded or extracted file doesn't match its recorded size or hash."""


class UnsafeArchiveError(ValueError):
    """An archive member would land outside the extraction folder."""


def hash_stream(stream: IO[bytes], algorithm: str = "sha256") -> str:
    """Hex digest of a binary stream, read in chunks."""
    digest = hashlib.new(algorithm)
    for block in iter(lambda: stream.read(CHUNK), b""):
        digest.update(block)
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    """SHA-256 of a file, streamed."""
    with open(path, "rb") as handle:
        return hash_stream(handle, "sha256")


def md5_file(path: Path) -> str:
    """MD5 of a file, streamed (only to check a publisher's md5; we record SHA-256)."""
    with open(path, "rb") as handle:
        return hash_stream(handle, "md5")


def hashes_file(path: Path) -> dict[str, str]:
    """MD5 and SHA-256 of a file in one pass."""
    md5, sha = hashlib.md5(), hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            md5.update(block)
            sha.update(block)
    return {"md5": md5.hexdigest(), "sha256": sha.hexdigest()}


def verify(path: Path, size: int | None = None, md5: str | None = None, sha256: str | None = None) -> dict:
    """Check a file's size and hashes; return ``{bytes, md5, sha256}`` or raise ``ChecksumError``."""
    path = Path(path)
    actual = path.stat().st_size
    if size is not None and actual != size:
        raise ChecksumError(f"{path.name}: {actual} bytes, expected {size}")
    found = hashes_file(path)
    if md5 and found["md5"] != md5.lower():
        raise ChecksumError(f"{path.name}: md5 {found['md5']}, expected {md5}")
    if sha256 and found["sha256"] != sha256.lower():
        raise ChecksumError(f"{path.name}: sha256 {found['sha256']}, expected {sha256}")
    return {"bytes": actual, **found}


def download(url: str, dest: Path, size: int | None = None, md5: str | None = None,
             sha256: str | None = None, floor: int = paths.FREE_SPACE_FLOOR) -> dict:
    """Download ``url`` to ``dest`` (resuming a partial file), then verify it.

    Refuses to start if the download would leave less than ``floor`` bytes free. Returns
    ``{bytes, md5, sha256}``.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    have = dest.stat().st_size if dest.exists() else 0
    if size is None or have < size:
        paths.require_free_space((size or 0) - have, floor=floor, path=dest.parent)
        subprocess.run(
            ["curl", "-L", "-C", "-", "--fail", "--retry", "10", "--retry-delay", "5", "-sS",
             "-o", str(dest), url],
            check=True,
        )
    return verify(dest, size=size, md5=md5, sha256=sha256)


def safe_member_path(name: str) -> PurePosixPath:
    """The relative path an archive member may be written to; ``UnsafeArchiveError`` if it would escape."""
    normal = name.replace("\\", "/")
    if not normal or normal.startswith("/") or (len(normal) > 1 and normal[1] == ":"):
        raise UnsafeArchiveError(f"absolute archive member refused: {name!r}")
    parts = [part for part in PurePosixPath(normal).parts if part not in ("", ".")]
    if not parts or any(part == ".." for part in parts):
        raise UnsafeArchiveError(f"archive member outside the folder refused: {name!r}")
    return PurePosixPath(*parts)


def _write(source: IO[bytes], target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".part")
    with open(tmp, "wb") as out:
        shutil.copyfileobj(source, out, CHUNK)
    tmp.replace(target)


def extract_zip(zip_path: Path, dest: Path, members: Iterable[str] | None = None,
                rename: dict[str, str] | None = None) -> list[Path]:
    """Extract ``members`` (default: all files) of a zip under ``dest``; return the written paths.

    ``rename`` maps a member name to a relative target path. Every target is checked with
    ``safe_member_path``; directories and links are skipped or refused.
    """
    dest = Path(dest)
    written = []
    with zipfile.ZipFile(zip_path) as archive:
        wanted = list(members) if members is not None else [i.filename for i in archive.infolist()]
        for name in wanted:
            info = archive.getinfo(name)
            if info.is_dir():
                continue
            if (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise UnsafeArchiveError(f"symlink in archive refused: {name!r}")
            relative = safe_member_path((rename or {}).get(name, name))
            target = dest / relative
            with archive.open(info) as source:
                _write(source, target)
            written.append(target)
    return written


def extract_tar(tar_path: Path, dest: Path, members: Iterable[str] | None = None,
                skip_links: bool = False) -> list[Path]:
    """Extract regular files of a tar (any compression) under ``dest``.

    Links and devices are refused, or with ``skip_links`` left unextracted.
    """
    dest = Path(dest)
    written = []
    with tarfile.open(tar_path) as archive:
        chosen = set(members) if members is not None else None
        for info in archive:
            if chosen is not None and info.name not in chosen:
                continue
            if info.isdir():
                safe_member_path(info.name)
                continue
            if not info.isfile():
                if skip_links and (info.issym() or info.islnk()):
                    continue
                raise UnsafeArchiveError(f"non-file tar member refused: {info.name!r}")
            target = dest / safe_member_path(info.name)
            source = archive.extractfile(info)
            if source is None:
                raise UnsafeArchiveError(f"unreadable tar member: {info.name!r}")
            with source:
                _write(source, target)
            written.append(target)
    return written
