"""Keep lyrics and non-commercial files out of git. Standard library only; prints locations, never text.

TEST-ONLY guard. Builds its lyric index from CC BY-NC sources for a pre-commit check; not for product/
training import.

Two checks over every file in each repo's index (tracked plus staged, exactly what a commit would hold):

1. **Lyrics.** Word 6-grams (and 10-character runs for Japanese) are built in memory from every karaoke
   chart's words and from RWC's lyric files. Any committed text that repeats one is reported as
   ``repo:path:line -> item id``. The matched words are never printed. N-grams with fewer than 3 distinct
   tokens ("la la la ...") and n-grams inside a song's own title or artist (credits) are ignored.
2. **Files.** Paths under ``out/``, ``.cache/`` or ``voices/``, anything named ``lights-on*``, audio, MIDI,
   SheetSage labs, reference TSVs, arrays and karaoke charts are refused.

    python -m open_transcriber.bakeoff.lyrics_guard --repos REPO [REPO ...] [--paths FILE ...]

Run it before every commit in either repo. Exit 0 when clean, 1 on any finding.
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable, Iterator

from . import paths
from .manifests import usdx_id

WORD_N = 6
CJK_N = 10
MIN_DISTINCT = 3

REFUSED_DIRS = ("out", ".cache", "voices")
REFUSED_SUFFIXES = {
    # audio and video
    ".wav", ".mp3", ".ogg", ".oga", ".flac", ".m4a", ".aac", ".aif", ".aiff", ".opus", ".wma", ".webm", ".mp4",
    # MIDI and karaoke
    ".mid", ".midi", ".kar", ".smf",
    # SheetSage labs, reference tables, arrays
    ".lab", ".tsv", ".npz", ".npy",
}
_WORD = re.compile(r"[^\W_]+")
_CHART_NOTE = re.compile(r"^[:*FRG]\s+-?\d+\s+-?\d+\s+-?\d+(\s|$)")

#: Known, allowed overlaps: (path in the repo, item id) -> why. Reported as allowed, never as findings.
ALLOWED_OVERLAPS = {
    ("songs/when-the-saints.yaml", "rwc-r012"):
        "public-domain words (pre-1929) in Copyist's own song file; RWC-R012 is a recording of the same song",
    ("songs/silent-night.yaml", "rwc-r013"):
        "public-domain words (1818) in Copyist's own song file; RWC-R013 is a recording of the same song",
}


def _is_cjk(ch: str) -> bool:
    code = ord(ch)
    return (0x3040 <= code <= 0x30FF or 0x3400 <= code <= 0x4DBF or 0x4E00 <= code <= 0x9FFF
            or 0xF900 <= code <= 0xFAFF or 0xFF66 <= code <= 0xFF9F)


def _normal(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    return text.replace("\u2019", "").replace("'", "").replace("`", "")


def streams(lines: Iterable[str]) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    """Word tokens and CJK characters, each with its 1-based line number."""
    words, cjk = [], []
    for number, line in enumerate(lines, 1):
        for token in _WORD.findall(_normal(line)):
            latin = "".join(ch for ch in token if not _is_cjk(ch))
            if latin:
                words.append((latin, number))
            cjk.extend((ch, number) for ch in token if _is_cjk(ch))
    return words, cjk


def ngrams(tokens: list[tuple[str, int]], n: int) -> Iterator[tuple[tuple[str, ...], int]]:
    """Informative n-grams (at least MIN_DISTINCT distinct tokens) with the line each starts on."""
    for i in range(len(tokens) - n + 1):
        gram = tuple(t for t, _ in tokens[i:i + n])
        if len(set(gram)) >= MIN_DISTINCT:
            yield gram, tokens[i][1]


@dataclass
class Index:
    """Lyric n-grams by item id, and credit n-grams that are allowed."""
    words: dict[tuple[str, ...], str]
    cjk: dict[tuple[str, ...], str]
    allowed: set[tuple[str, ...]]
    sources: int

    def add(self, item: str, lines: Iterable[str]) -> None:
        words, cjk = streams(lines)
        for gram, _ in ngrams(words, WORD_N):
            self.words.setdefault(gram, item)
        for gram, _ in ngrams(cjk, CJK_N):
            self.cjk.setdefault(gram, item)
        self.sources += 1

    def allow(self, credit: str) -> None:
        words, cjk = streams([credit])
        self.allowed.update(g for g, _ in ngrams(words, WORD_N))
        self.allowed.update(g for g, _ in ngrams(cjk, CJK_N))


def _usdx_lines(chart: Path) -> list[str]:
    from . import ultrastar  # local: stdlib only, but keep the import cheap

    return ultrastar.lyrics_text(ultrastar.parse(chart)).splitlines()


def _rwc_lyric_lines(path: Path) -> list[str]:
    """An RWC lyric file's lines after its boilerplate header block (up to the first blank line)."""
    lines = path.read_bytes().decode("cp932", errors="replace").splitlines()
    for i, line in enumerate(lines):
        if not line.strip():
            return lines[i + 1:]
    return lines


def build_index(data: Path | None = None) -> Index:
    """Lyric n-grams from the karaoke charts and RWC's lyric files under the judge's data folder."""
    data = Path(data or paths.data_dir())
    index = Index({}, {}, set(), 0)
    usdx_root = data / "usdx" / "src"
    for chart in sorted(usdx_root.glob("*/Creative Commons/*/*.txt")):
        if "licen" in chart.name.lower():
            continue
        from . import ultrastar

        header = ultrastar.read_header(chart)
        index.allow(header.get("TITLE", ""))
        index.allow(header.get("ARTIST", ""))
        index.add(usdx_id(chart.parent.name), _usdx_lines(chart))
    for lyric in sorted((data / "rwc" / "archive").glob("RWC-MDB-*_Lyrics/*.TXT")):
        item = "rwc-" + lyric.stem.split("-", 1)[1].lower()  # RM-P001 -> rwc-p001
        index.add(item, _rwc_lyric_lines(lyric))
    metadata = data / "rwc" / "annotations" / "metadata.csv"
    if metadata.is_file():
        with open(metadata, encoding="utf-8") as handle:
            for row in csv.DictReader(handle, delimiter=";"):
                index.allow(row.get("Title", ""))
                index.allow(row.get("Artist", ""))
    return index


@dataclass(frozen=True)
class Finding:
    where: str  # repo:path[:line]
    reason: str  # an item id or a file rule; never lyric text
    allowed: bool = False


def is_chart(text: str) -> bool:
    """True for UltraStar chart content: a #BPM header and at least 10 note lines."""
    lines = text.splitlines()
    has_bpm = any(line.upper().startswith("#BPM:") for line in lines[:60])
    notes = sum(1 for line in lines if _CHART_NOTE.match(line))
    return has_bpm and notes >= 10


def path_rule(path: str) -> str | None:
    """Why a path may not be committed, or None."""
    parts = PurePosixPath(path).parts
    if parts and parts[0] in REFUSED_DIRS:
        return f"{parts[0]}/ is never committed"
    if any(part.lower().startswith("lights-on") or part.lower().startswith("lights_on") for part in parts):
        return "the owner's lights-on material is never committed"
    if PurePosixPath(path).suffix.lower() in REFUSED_SUFFIXES:
        return f"{PurePosixPath(path).suffix.lower()} files are never committed"
    return None


def scan_text(where: str, text: str, index: Index, path: str | None = None) -> list[Finding]:
    """Lyric n-gram hits in one text, one finding per (line, item); ``path`` is checked against the allowlist."""
    words, cjk = streams(text.splitlines())
    hits: dict[tuple[int, str], None] = {}
    for table, tokens, n in ((index.words, words, WORD_N), (index.cjk, cjk, CJK_N)):
        for gram, line in ngrams(tokens, n):
            item = table.get(gram)
            if item and gram not in index.allowed:
                hits[(line, item)] = None
    return [Finding(f"{where}:{line}", item, (path, item) in ALLOWED_OVERLAPS) for line, item in sorted(hits)]


def index_files(repo: Path) -> list[tuple[str, str, str]]:
    """(mode, blob sha, path) for every entry in a repo's index."""
    out = subprocess.run(["git", "-C", str(repo), "ls-files", "-s", "-z"], check=True,
                         capture_output=True).stdout.decode("utf-8", "surrogateescape")
    entries = []
    for record in out.split("\0"):
        if not record:
            continue
        meta, path = record.split("\t", 1)
        mode, sha, _stage = meta.split()
        entries.append((mode, sha, path))
    return entries


def read_blobs(repo: Path, shas: list[str]) -> dict[str, bytes]:
    """Blob contents by sha, through one ``git cat-file --batch``."""
    if not shas:
        return {}
    process = subprocess.run(["git", "-C", str(repo), "cat-file", "--batch"], check=True,
                             input="\n".join(shas).encode() + b"\n", capture_output=True)
    data, blobs, pos = process.stdout, {}, 0
    for sha in shas:
        end = data.index(b"\n", pos)
        header = data[pos:end].split()
        size = int(header[2])
        blobs[sha] = data[end + 1:end + 1 + size]
        pos = end + 1 + size + 1
    return blobs


def _text(blob: bytes) -> str | None:
    if b"\0" in blob[:8192]:
        return None
    return blob.decode("utf-8", errors="replace")


def scan_repo(repo: Path, index: Index) -> list[Finding]:
    """Every finding in one repo's index."""
    repo = Path(repo)
    findings = []
    entries = [(mode, sha, path) for mode, sha, path in index_files(repo) if mode != "160000"]
    blobs = read_blobs(repo, sorted({sha for _, sha, _ in entries}))
    for _mode, sha, path in entries:
        where = f"{repo.name}:{path}"
        rule = path_rule(path)
        if rule:
            findings.append(Finding(where, rule))
            continue
        text = _text(blobs[sha])
        if text is None:
            continue
        if is_chart(text):
            findings.append(Finding(where, "karaoke chart content is never committed"))
            continue
        findings.extend(scan_text(where, text, index, path))
    return findings


def scan_paths(files: Iterable[Path], index: Index) -> list[Finding]:
    """Findings in extra files outside git (reports, messages)."""
    findings = []
    for file in files:
        text = _text(Path(file).read_bytes())
        if text is not None:
            findings.extend(scan_text(str(file), text, index))
    return findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repos", nargs="*", type=Path, default=[])
    parser.add_argument("--paths", nargs="*", type=Path, default=[], help="extra files to scan (reports)")
    parser.add_argument("--data", type=Path, default=None, help="the judge's data folder")
    parser.add_argument("--allow-missing-sources", action="store_true",
                        help="pass the lyric check when no charts or lyric files are present")
    args = parser.parse_args(argv)
    index = build_index(args.data)
    if index.sources == 0 and not args.allow_missing_sources:
        print("No lyric sources found under the judge's data folder; can't check for lyrics.", file=sys.stderr)
        return 1
    findings = []
    for repo in args.repos:
        findings += scan_repo(repo, index)
    findings += scan_paths(args.paths, index)
    bad = [f for f in findings if not f.allowed]
    allowed = sorted({f.where.split(":")[1] + " -> " + f.reason for f in findings if f.allowed})
    print(f"lyrics guard: {index.sources} lyric sources, {len(index.words) + len(index.cjk)} n-grams; "
          f"{len(args.repos)} repos, {len(args.paths)} extra files; {len(bad)} findings, "
          f"{len(findings) - len(bad)} allowed overlaps")
    for entry in allowed:
        print(f"  allowed: {entry}")
    for finding in bad:
        print(f"  {finding.where} -> {finding.reason}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
