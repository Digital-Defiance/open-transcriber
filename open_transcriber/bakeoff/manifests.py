"""Test-set manifests: ids, credits, hashes, licences and allowed use. No non-commercial content.

TEST-ONLY. RWC and the karaoke songs are CC BY-NC: evaluation/probe-eval only, never training. Product and
training code must not import this module; the import-boundary test enforces that.

Three sets, one JSON file each in ``manifests/``:

- ``rwc-pop``: RWC 2.0's 100 popular-music songs (SheetSage2's own test);
- ``rwc-other``: RWC 2.0's other 75 sung pieces (52 Genre, 15 Royalty-Free, 6 Classical, 2 Jazz);
- ``usdx``: UltraStar Deluxe's 39 Creative Commons karaoke songs (32 scored, 4 scored apart, 3 left out).

Commands (evaluation environment; the data lives under ``paths.data_dir()``):

    python -m open_transcriber.bakeoff.manifests --fetch     # download and extract what's missing
    python -m open_transcriber.bakeoff.manifests --build     # write manifests/*.json from the data folder
    python -m open_transcriber.bakeoff.manifests --verify    # re-hash every listed file: "OK: 214 items"
    python -m open_transcriber.bakeoff.manifests --licences  # each karaoke licence file's URLs, never its text

Commits may carry ids, titles and artists (credits), hashes, licences and counts. Never audio, annotations,
references, lyrics or licence-file text.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import struct
import subprocess
import sys
import unicodedata
import zipfile
from pathlib import Path

from . import fetch, paths

FORMAT = "copyist-judge-manifest/1"
MANIFEST_DIR = Path(__file__).with_name("manifests")
SETS = ("rwc-pop", "rwc-other", "usdx")
STATUSES = ("scored", "scored-apart", "excluded", "excluded-0a")

NC_USE = "testing only: never train on, ship or share; publish scores and credits only"
OPEN_USE = ("testing only (held out from training); the audio's licence would allow publishing our "
            "transcriptions, which stage 0a doesn't do")

RWC_ZIP = {
    "record": "https://zenodo.org/records/17177919",
    "doi": "10.5281/zenodo.17177919",
    "download": "https://zenodo.org/api/records/17177919/files/RWC-CGJPR.zip/content",
    "file": "RWC-CGJPR.zip",
    "bytes": 13418904710,
    "md5": "272309ed06121e6ed2ac5c8f55664baa",  # Zenodo's
    "sha256": "0a064310a3d233e0a8b0819bbb1a718c11bafa51cf28cfa97b96dd5614f7e48c",  # ours, 2026-10-19
}
RWC_ANNOTATIONS = {"url": "https://github.com/rwc-music/rwc-annotations",
                   "commit": "0a1a6c31dbe73a7f5d44f7caef8cd0999402a4c2"}
RWC_ARCHIVE = {"url": "https://github.com/rwc-music/rwc-annotations-archive",
               "commit": "994499597853664890b517711f97df9f47c0358a"}
USDX_REPO = {
    "url": "https://github.com/UltraStar-Deluxe/songs",
    "commit": "c9246c799050acec96a03183df4b85493d283d51",
    "download": "https://codeload.github.com/UltraStar-Deluxe/songs/tar.gz/c9246c799050acec96a03183df4b85493d283d51",
    "file": "songs-c9246c7.tar.gz",
    "bytes": 295694336,
    "sha256": "70249539a76e6cd2d9eb9316b987e28535d0186eb0a3aa823af5cee2146eb956",
}
USDX_FOLDER = "songs-c9246c799050acec96a03183df4b85493d283d51/Creative Commons"

RWC_COLLECTIONS = {"P": "RWC-P", "G": "RWC-G", "R": "RWC-R", "C": "RWC-C", "J": "RWC-J"}
RWC_NAMES = {
    "P": "Popular Music", "G": "Music Genre", "R": "Royalty-Free Music", "C": "Classical Music", "J": "Jazz Music",
}

# Hand-checked from each song's license.txt (URLs only; 2026-10-19), with the audit's exclusions:
# folder -> (status, audio licence, chart licence, in HookTheory's training split)
_JC = ("scored", "CC BY-NC 3.0", "CC BY-NC 3.0", False)
_JC_HT = ("scored-apart", "CC BY-NC 3.0", "CC BY-NC 3.0", True)
_NOT_STATED = "not stated separately"
USDX_SONGS: dict[str, tuple[str, str, str, bool]] = {
    "Dead Smiling Pirates - I": ("excluded", "CC BY-NC-ND 2.5", _NOT_STATED, False),
    "Fairy Bot Orchestra - Heaven can't wait": ("scored", "CC BY-NC-SA 3.0", _NOT_STATED, False),
    "Jonathan Coulton - A Talk with George": _JC,
    "Jonathan Coulton - Better": _JC,
    "Jonathan Coulton - Big Bad World One": _JC,
    "Jonathan Coulton - Chiron Beta Prime": _JC,
    "Jonathan Coulton - Code Monkey": _JC_HT,
    "Jonathan Coulton - Creepy Doll": _JC,
    "Jonathan Coulton - Dance, Soterios Johnson, Dance": _JC,
    "Jonathan Coulton - First of May": _JC,
    "Jonathan Coulton - Flickr": _JC,
    "Jonathan Coulton - Furry Old Lobster": _JC,
    "Jonathan Coulton - I Crush Everything": _JC,
    "Jonathan Coulton - I Feel Fantastic": _JC_HT,
    "Jonathan Coulton - I'm Your Moon": _JC,
    "Jonathan Coulton - Ikea": _JC,
    "Jonathan Coulton - Monkey Shines": _JC,
    "Jonathan Coulton - Mr. Fancy Pants": _JC,
    "Jonathan Coulton - My Beige Bear": _JC,
    "Jonathan Coulton - Not About You": _JC,
    "Jonathan Coulton - Re- Your Brains": _JC_HT,
    "Jonathan Coulton - Shop Vac": _JC_HT,
    "Jonathan Coulton - Skullcrusher Mountain": _JC,
    "Jonathan Coulton - That Spells DNA": _JC,
    "Jonathan Coulton - The Future Soon": _JC,
    "Jonathan Coulton - When You Go": _JC,
    "Joshua Morin - On the run": ("scored", "CC BY-SA 2.5", _NOT_STATED, False),
    "Pornophonique - Space Invaders": ("excluded", "CC BY-NC-ND 2.0", _NOT_STATED, False),
    "Shearer - 69": ("scored", "CC BY-NC-SA 2.0", _NOT_STATED, False),
    "Shearer - Can't stop it": ("scored", "CC BY-NC-SA 2.0", _NOT_STATED, False),
    "Shearer - Consequence of Dawn": ("scored", "CC BY-NC-SA 2.0", _NOT_STATED, False),
    "Shearer - In My Hand": ("scored", "CC BY-NC-SA 3.0", _NOT_STATED, False),
    "Shearer - Man Song": ("scored", "CC BY-NC-SA 2.0", _NOT_STATED, False),
    "Shearer - Stay with me": ("scored", "CC BY-NC-SA 2.0", _NOT_STATED, False),
    "Silver Note - Chasing Marks": ("scored", "CC BY 4.0", _NOT_STATED, False),
    "Silver Note - Sonic Rainboom VIP": ("scored", "CC BY 4.0", _NOT_STATED, False),
    "Steven Dunston - Northern Star": ("scored", "CC BY-NC-SA 2.5", _NOT_STATED, False),
    "Systemabsturz - Verdächtig": ("excluded-0a", "CC0 1.0", _NOT_STATED, False),
    "The Wasteland Wailers - Dare Master": ("scored", "CC BY-NC 3.0", "CC BY-NC 3.0", False),
}
USDX_REASONS = {
    "excluded": "no-derivatives licence: no right to make adaptations, and a transcription may count as one",
    "excluded-0a": "its chart has no pitches; lyric timing only, in stage 0b",
    "scored-apart": "in HookTheory's training split, which SheetSage2 trained on",
}
_LANGUAGES = {"englisch": "English", "english": "English", "german": "German", "deutsch": "German"}


def slug(text: str) -> str:
    """ASCII slug: accents dropped, lower case, runs of other characters to one hyphen."""
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", ascii_text.lower()).strip("-")


def usdx_id(folder: str) -> str:
    return "usdx-" + slug(folder)


def rwc_id(rwcid: str) -> str:
    """RWC_P001 -> rwc-p001."""
    return rwcid.replace("RWC_", "rwc-").lower()


# --- audio file facts (stdlib RIFF reader: PCM, IEEE float and WAVE_FORMAT_EXTENSIBLE) ---

def wav_info(path: Path) -> dict:
    """Rate, channels, bits, sample format, frames and seconds of a WAV file, from its header."""
    with open(path, "rb") as handle:
        riff, _, wave = struct.unpack("<4sI4s", handle.read(12))
        if riff != b"RIFF" or wave != b"WAVE":
            raise ValueError(f"{Path(path).name}: not a RIFF/WAVE file")
        fmt = None
        while True:
            head = handle.read(8)
            if len(head) < 8:
                raise ValueError(f"{Path(path).name}: no data chunk")
            chunk, size = struct.unpack("<4sI", head)
            if chunk == b"fmt ":
                body = handle.read(size + (size & 1))
                tag, channels, rate, _, align, bits = struct.unpack("<HHIIHH", body[:16])
                if tag == 0xFFFE and size >= 40:
                    tag = struct.unpack("<H", body[24:26])[0]
                fmt = {"format": {1: "pcm", 3: "float"}.get(tag, f"tag-{tag}"), "channels": channels,
                       "rate": rate, "bits": bits, "align": align}
            elif chunk == b"data":
                if fmt is None:
                    raise ValueError(f"{Path(path).name}: data before fmt")
                frames = size // fmt.pop("align")
                return {**fmt, "frames": frames, "seconds": round(frames / fmt["rate"], 6)}
            else:
                handle.seek(size + (size & 1), 1)


def file_record(path: Path, data: Path) -> dict:
    """``{path relative to the data folder, sha256, bytes}``."""
    return {"path": Path(path).relative_to(data).as_posix(), "sha256": fetch.sha256_file(path),
            "bytes": Path(path).stat().st_size}


def audio_record(path: Path, data: Path) -> dict:
    info = wav_info(path)
    return {**file_record(path, data), "rate": info["rate"], "channels": info["channels"], "bits": info["bits"],
            "sample_format": info["format"], "seconds": info["seconds"]}


# --- RWC ---

def _rwc_paths(data: Path) -> dict[str, Path]:
    root = data / "rwc"
    return {
        "zip": root / RWC_ZIP["file"],
        "audio": root / "audio",
        "annotations": root / "annotations",
        "archive": root / "archive",
        "metadata": root / "annotations" / "metadata.csv",
        "vocals": root / "annotations" / "01_annotations_preprocessed" / "MIDI_aligned" / "metadata_mapping_vocals.csv",
        "melody_map": root / "annotations" / "01_annotations_preprocessed" / "melody"
        / "metadata_mapping_melody_annotations.csv",
    }


def _read_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter=";"))


def rwc_sung_ids(data: Path) -> list[str]:
    """The 175 RWCIDs in the vocal mapping, in order."""
    return sorted({row["RWCID"] for row in _read_csv(_rwc_paths(data)["vocals"])})


def rwc_zip_members(zip_path: Path, rwcids: list[str]) -> dict[str, str]:
    """RWCID -> its zip member (``RWC_OpenAccess_WAV/RWC-<C>/<RWCID>.wav``); ValueError if one is missing."""
    with zipfile.ZipFile(zip_path) as archive:
        names = {i.filename for i in archive.infolist()}
    found = {}
    for rwcid in rwcids:
        name = f"RWC_OpenAccess_WAV/{RWC_COLLECTIONS[rwcid[4]]}/{rwcid}.wav"
        if name not in names:
            raise ValueError(f"{rwcid}: no {name} in the zip")
        found[rwcid] = name
    return found


def extract_rwc_audio(data: Path) -> list[Path]:
    """Extract the 175 sung pieces' WAVs from the zip to ``rwc/audio/RWC_<id>.wav`` (skips existing files)."""
    where = _rwc_paths(data)
    ids = rwc_sung_ids(data)
    members = rwc_zip_members(where["zip"], ids)
    todo = {members[i]: f"RWC_{i.split('_')[1]}.wav" for i in ids
            if not (where["audio"] / f"RWC_{i.split('_')[1]}.wav").exists()}
    if not todo:
        return []
    return fetch.extract_zip(where["zip"], where["audio"], members=list(todo), rename=todo)


def _rwc_item(rwcid: str, row: dict, data: Path, where: dict[str, Path], zip_member: str | None) -> dict:
    code = rwcid.split("_")[1]  # P001
    coll = RWC_COLLECTIONS[code[0]]
    annotations = [where["annotations"] / "01_annotations_preprocessed" / "MIDI_aligned" / coll / f"{rwcid}.mid"]
    if code[0] == "P":
        annotations += [
            where["annotations"] / "01_annotations_preprocessed" / "melody" / "RWC-P" / f"{rwcid}.csv",
            where["archive"] / "AIST_RWC-MDB-P-2001_VOCA_INST" / f"RM-{code}.VOCA_INST.TXT",
            where["archive"] / "AIST_RWC-MDB-P-2001_MIDI_SYNC" / f"RM-{code}.SMF_SYNC.MID",
        ]
    audio = audio_record(where["audio"] / f"RWC_{code}.wav", data)
    if zip_member:
        audio["zip_member"] = zip_member
    return {
        "id": rwc_id(rwcid),
        "source_id": rwcid,
        "collection": coll,
        "title": row["Title"],
        "artist": row["Artist"],
        "language": row["SingingLanguage"] or None,
        "singers": row["SingerInformation"] or None,
        "status": "scored",
        "cluster": rwc_id(rwcid),
        "audio": audio,
        "annotations": [file_record(p, data) for p in annotations],
        "reference": None,
    }


def build_rwc(data: Path) -> dict[str, dict]:
    """The two RWC manifests (``rwc-pop`` and ``rwc-other``) from the data folder."""
    where = _rwc_paths(data)
    metadata = {row["RWCID"]: row for row in _read_csv(where["metadata"])}
    ids = rwc_sung_ids(data)
    members = rwc_zip_members(where["zip"], ids) if where["zip"].exists() else {}
    shared = [file_record(where[k], data) for k in ("metadata", "vocals", "melody_map")]
    common = {
        "format": FORMAT,
        "source": {"audio": RWC_ZIP, "annotations": RWC_ANNOTATIONS, "archive": RWC_ARCHIVE},
        "licence": "CC BY-NC 4.0 (RWC 2.0 audio, rwc-annotations and the AIST archive)",
        "allowed_use": NC_USE,
        "credit": "RWC Music Database (Goto et al., 2002-2003), re-released as RWC 2.0 under CC BY-NC 4.0 "
                  "(Zenodo 10.5281/zenodo.17177919); annotations from rwc-music/rwc-annotations",
        "shared_annotations": shared,
    }
    pop = [i for i in ids if i.startswith("RWC_P")]
    other = [i for i in ids if not i.startswith("RWC_P")]
    out = {}
    for name, chosen, title, notes in (
        ("rwc-pop", pop, "RWC 2.0 Popular Music (RWC-P): all 100 songs",
         "SheetSage2's own test (RWC-Pop vocal and full melody). 80 Japanese, 20 English."),
        ("rwc-other", other, "RWC 2.0's other 75 sung pieces (52 Genre, 15 Royalty-Free, 6 Classical, 2 Jazz)",
         "The pieces with a vocal track in the June 2026 vocal mapping. SheetSage2 never reported on these. "
         "RWC-R012 and RWC-R013 are recordings of two songs Copyist's rules were tuned on (When the Saints, "
         "Silent Night), so P2 is also reported without them."),
    ):
        out[name] = {**common, "set": name, "title": title, "notes": notes,
                     "items": [_rwc_item(i, metadata[i], data, where, members.get(i)) for i in chosen]}
    return out


# --- UltraStar ---

def _usdx_root(data: Path) -> Path:
    return data / "usdx" / "src" / USDX_FOLDER


def ffmpeg_version() -> str:
    first = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, check=True).stdout.splitlines()[0]
    return first.split(" Copyright")[0]


def decode_usdx_audio(data: Path) -> list[Path]:
    """Decode each non-excluded song's ``#MP3`` file to float WAV at its own rate and channels (D9)."""
    from . import ultrastar

    written = []
    for folder, (status, *_rest) in USDX_SONGS.items():
        if status.startswith("excluded"):
            continue
        src_dir = _usdx_root(data) / folder
        header = ultrastar.read_header(ultrastar.chart_path(src_dir))
        source = src_dir / header["MP3"]
        target = data / "usdx" / "audio" / f"{usdx_id(folder)}.wav"
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.stem + ".part.wav")
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(source), "-map", "0:a:0",
                        "-c:a", "pcm_f32le", str(tmp)], check=True)
        tmp.replace(target)
        written.append(target)
    return written


def _language(tag: str | None) -> str:
    return _LANGUAGES.get((tag or "").strip().lower(), (tag or "").strip() or "English")


def build_usdx(data: Path) -> dict:
    """The karaoke manifest: all 39 songs, with status, licences, chart counts and decoded audio."""
    from . import ultrastar

    root = _usdx_root(data)
    folders = sorted(p.name for p in root.iterdir() if p.is_dir())
    if sorted(USDX_SONGS) != folders:
        raise ValueError("the karaoke folders don't match the hand-checked licence table")
    decoder = ffmpeg_version()
    items = []
    for folder in folders:
        status, audio_licence, chart_licence, hooktheory = USDX_SONGS[folder]
        chart_file = ultrastar.chart_path(root / folder)
        chart = ultrastar.parse(chart_file)
        header = chart.header
        source = root / folder / header["MP3"]
        item = {
            "id": usdx_id(folder),
            "source_id": folder,
            "collection": "UltraStar Deluxe songs: Creative Commons",
            "title": header.get("TITLE", ""),
            "artist": header.get("ARTIST", ""),
            "language": _language(header.get("LANGUAGE")),
            "singers": None,
            "status": status,
            "cluster": usdx_id(folder),
            "licence": {"audio": audio_licence, "chart": chart_licence},
            "allowed_use": NC_USE if "NC" in audio_licence else OPEN_USE,
            "hooktheory": hooktheory,
            "chart": {
                "bpm": chart.bpm,
                "gap_ms": round(chart.gap * 1000, 3),
                "notes": len(chart.notes),
                "pitched": sum(1 for n in chart.notes if n.pitched),
                "freestyle": sum(1 for n in chart.notes if not n.pitched),
                "distinct_pitches": len({n.pitch for n in chart.notes if n.pitched}),
                "lines": chart.lines,
            },
            "audio": None,
            "source_audio": file_record(source, data),
            "annotations": [file_record(chart_file, data), file_record(root / folder / "license.txt", data)],
            "reference": None,
        }
        if status in USDX_REASONS:
            item["reason"] = USDX_REASONS[status]
        wav = data / "usdx" / "audio" / f"{usdx_id(folder)}.wav"
        if not status.startswith("excluded"):
            item["audio"] = {**audio_record(wav, data), "decoder": decoder,
                             "decode": "ffmpeg -nostdin -v error -i SRC -map 0:a:0 -c:a pcm_f32le DST.wav"}
        items.append(item)
    return {
        "format": FORMAT,
        "set": "usdx",
        "title": "UltraStar Deluxe's Creative Commons karaoke songs (39; 32 scored, 4 scored apart)",
        "source": {"songs": USDX_REPO},
        "licence": "per song (see each item): 2 CC BY 4.0, 1 CC0, 1 CC BY-SA 2.5, 35 non-commercial",
        "allowed_use": "per song; non-commercial songs: " + NC_USE,
        "credit": "UltraStar Deluxe song repository (github.com/UltraStar-Deluxe/songs), each song's artist "
                  "and chart author per its license.txt",
        "notes": "Octaves in the charts are unreliable: scored by pitch class at 50 ms and 100 ms. Freestyle "
                 "notes have no pitch and are masked. Every chart is QA-checked before any system is scored.",
        "items": items,
    }


# --- loading, validation, verification ---

_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _check_file(entry: dict, where: str) -> None:
    if not isinstance(entry, dict) or not entry.get("path"):
        raise ValueError(f"{where}: missing path")
    if not _HEX64.match(str(entry.get("sha256", ""))):
        raise ValueError(f"{where}: missing or malformed sha256")


def validate(manifest: dict) -> dict:
    """Check a manifest's shape; return it. ``ValueError`` names the first problem."""
    for key in ("format", "set", "title", "source", "licence", "allowed_use", "items"):
        if key not in manifest:
            raise ValueError(f"manifest is missing {key!r}")
    if manifest["format"] != FORMAT:
        raise ValueError(f"unknown manifest format {manifest['format']!r}")
    seen = set()
    for item in manifest["items"]:
        ident = item.get("id")
        if not ident:
            raise ValueError("an item has no id")
        if ident in seen:
            raise ValueError(f"duplicate id {ident}")
        seen.add(ident)
        if item.get("status") not in STATUSES:
            raise ValueError(f"{ident}: unknown status {item.get('status')!r}")
        for key in ("title", "cluster", "annotations"):
            if key not in item:
                raise ValueError(f"{ident}: missing {key!r}")
        scored = not item["status"].startswith("excluded")
        if scored or item.get("audio"):
            _check_file(item.get("audio"), f"{ident} audio")
        for n, entry in enumerate(item["annotations"]):
            _check_file(entry, f"{ident} annotation {n}")
        for key in ("reference", "reference_full"):
            if item.get(key) is not None:
                _check_file(item[key], f"{ident} {key}")
        if manifest["set"] == "usdx" and not (item.get("licence") or {}).get("audio"):
            raise ValueError(f"{ident}: missing audio licence")
    return manifest


def load(name: str, directory: Path = MANIFEST_DIR) -> dict:
    """A committed manifest, validated."""
    return validate(json.loads((Path(directory) / f"{name}.json").read_text(encoding="utf-8")))


def load_all(directory: Path = MANIFEST_DIR) -> dict[str, dict]:
    return {name: load(name, directory) for name in SETS}


def write(manifest: dict, directory: Path = MANIFEST_DIR) -> Path:
    validate(manifest)
    path = Path(directory) / f"{manifest['set']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def file_entries(item: dict) -> list[dict]:
    """Every hashed file an item lists."""
    entries = [e for e in (item.get("audio"), item.get("source_audio"), item.get("reference"),
                           item.get("reference_full"), item.get("qa")) if e]
    return entries + list(item.get("annotations", []))


def verify(manifests: dict[str, dict], data: Path) -> list[str]:
    """Re-hash every file listed; return the problems (empty when all match)."""
    problems = []
    for manifest in manifests.values():
        for entry in manifest.get("shared_annotations", []):
            path = data / entry["path"]
            if not path.exists() or fetch.sha256_file(path) != entry["sha256"]:
                problems.append(f"{manifest['set']}: {entry['path']}")
        for item in manifest["items"]:
            for entry in file_entries(item):
                path = data / entry["path"]
                if not path.exists():
                    problems.append(f"{item['id']}: missing {entry['path']}")
                elif fetch.sha256_file(path) != entry["sha256"]:
                    problems.append(f"{item['id']}: hash differs for {entry['path']}")
    return problems


# --- licence files: URLs only ---

_URL = re.compile(r"https?://[^\s<>\"')\]]+")
_LABELS = re.compile(r"^\s*(mp3 file|audio file|video file|cover file|background file|background image|background|"
                     r"cover|ultrastar txt-song file|song|instrumental|vocals only|lyrics)\s*:", re.I)


def licence_urls(path: Path) -> list[tuple[str, str]]:
    """(section label, URL) pairs from a licence file. Labels come from a fixed list of file sections
    ("MP3 File:", "Cover File:", ...); every other piece of text is never kept."""
    from .ultrastar import decode

    label, pairs = "-", []
    for line in decode(Path(path).read_bytes()).splitlines():
        match = _LABELS.match(line)
        if match:
            label = match.group(1).title()
        for url in _URL.findall(line):
            pairs.append((label, url))
    return pairs


def _fetch_all(data: Path) -> None:
    rwc = _rwc_paths(data)
    fetch.download(RWC_ZIP["download"], rwc["zip"], size=RWC_ZIP["bytes"], md5=RWC_ZIP["md5"],
                   sha256=RWC_ZIP["sha256"])
    usdx_tar = data / "usdx" / USDX_REPO["file"]
    fetch.download(USDX_REPO["download"], usdx_tar, size=USDX_REPO["bytes"], sha256=USDX_REPO["sha256"])
    if not _usdx_root(data).exists():
        fetch.extract_tar(usdx_tar, data / "usdx" / "src")
    for target, pin in ((rwc["annotations"], RWC_ANNOTATIONS), (rwc["archive"], RWC_ARCHIVE)):
        if not target.exists():
            subprocess.run(["git", "clone", "-q", "--filter=blob:none", "--no-checkout", pin["url"], str(target)],
                           check=True)
        subprocess.run(["git", "-C", str(target), "checkout", "-q", pin["commit"]], check=True)
    paths.require_free_space(8 * 10**9, path=data)
    extract_rwc_audio(data)
    decode_usdx_audio(data)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--licences", action="store_true")
    parser.add_argument("--data", type=Path, default=None)
    args = parser.parse_args(argv)
    data = Path(args.data or paths.data_dir())
    if args.fetch:
        _fetch_all(data)
    if args.build:
        for manifest in (*build_rwc(data).values(), build_usdx(data)):
            print(f"wrote {write(manifest)} ({len(manifest['items'])} items)")
    if args.licences:
        for folder in sorted(USDX_SONGS):
            print(f"== {folder}")
            for label, url in licence_urls(_usdx_root(data) / folder / "license.txt"):
                print(f"   {label}: {url}")
    if args.verify:
        manifests = load_all()
        problems = verify(manifests, data)
        for problem in problems:
            print("BAD", problem)
        if problems:
            return 1
        print(f"OK: {sum(len(m['items']) for m in manifests.values())} items")
    return 0


if __name__ == "__main__":
    sys.exit(main())
