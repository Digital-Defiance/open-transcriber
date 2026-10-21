"""Reference builders for evaluation, written to the data folder and never committed.

TEST-ONLY. Builds references from RWC/karaoke (CC BY-NC) annotations for evaluation/probe-eval only, never
training. Product and training code must not import this module.

- **RWC vocal, V1 (primary)**: the piece's rows in RWC 2.0's vocal mapping (``metadata_mapping_vocals.csv``)
  on its aligned MIDI. ``MIDITrackNo`` indexes the SMF track list, and the track's name must equal
  ``MIDITrackName`` (stripped) or the build aborts. A row whose comment says it's back vocals, not vocal, not
  the main melody or nothing is left out, unless that would leave the piece with no rows (RWC-P026, whose
  only row is the main melody except in the chorus): then the rows are kept and the piece is flagged.
  Octave-exact pitch = note + s·SemitoneOffset, with the sign s fixed by an automatic check against RWC-P's
  F0 annotations.
- **RWC-Pop full, F1 (primary)**: V1 plus the instrumental solos: for each ``s:<instrument>`` segment of the
  AIST VOCA_INST annotation, one track (by tier: a name with SOLO; else the instrument's General MIDI family;
  else any track that isn't drums or vocal), the one with the most onsets in the segment (ties: higher median
  pitch, then lower index), its notes with onsets inside the segment, as a 20 ms skyline.
- **Diagnostic variants**, scored only in G0: V2 (all rows), V3 (the 2006 MIDI_SYNC files, V1's tracks by name),
  V4 (V1 inside VOCA_INST vocal segments), F2 (F1 without the skyline), F3 (V4 plus F1's solos),
  F4 (V1 plus every note of tracks named SOLO).
- **Karaoke**: every chart note; freestyle masked (``ultrastar.py``).

Each reference is a TSV (``note_id onset offset midi midi_sung mask source``) with a provenance JSON.

    python -m open_transcriber.bakeoff.refs --build-all     # evaluation environment
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from . import fetch, paths, rwc_midi

EXCLUDE_ROW = re.compile(r"back ?vocals?|not vocal|not the main|nothing\?", re.IGNORECASE)
#: Comments report octave inconsistencies: left out of octave-exact means only.
OCTAVE_EXACT_EXCLUDED = ("rwc-p003", "rwc-p031")
VOCAL_LABELS = ("m", "f", "g", "r")
SKYLINE_SECONDS = 0.020
COLUMNS = ("note_id", "onset", "offset", "midi", "midi_sung", "mask", "source")

#: VOCA_INST label typos, normalised.
LABEL_FIXES = {"e;ectricguitar": "electricguitar", "chelo": "cello", "malimba": "marimba", "okarina": "ocarina",
               "synthesizers": "synthesizer"}
_PIANO, _CHROM, _ORGAN, _GUITAR = range(0, 8), range(8, 16), range(16, 24), range(24, 32)
_BASS, _STRINGS, _ENSEMBLE, _BRASS = range(32, 40), range(40, 48), range(48, 56), range(56, 64)
_REED, _PIPE, _SYNTH = range(64, 72), range(72, 80), range(80, 104)
#: General MIDI program families for each solo instrument named in VOCA_INST (tier 2).
GM_FAMILIES: dict[str, tuple[range, ...]] = {
    "electricguitar": (_GUITAR,), "acousticguitar": (_GUITAR,), "slideguitar": (_GUITAR,),
    "guitarandbassriff": (_GUITAR, _BASS), "electricbass": (_BASS,), "bass": (_BASS,), "synthbass": (_BASS,),
    "synthesizer": (_SYNTH, _PIANO), "synthlead": (_SYNTH, _PIANO), "keyboard": (_PIANO, _SYNTH, _ORGAN),
    "synthstrings": (_ENSEMBLE, _SYNTH), "strings": (_STRINGS, _ENSEMBLE), "cello": (_STRINGS,),
    "brass": (_BRASS,), "trumpet": (_BRASS,), "synthbrass": (_BRASS, _SYNTH),
    "sax": (range(64, 68),), "synthsax": (range(64, 68), _SYNTH), "oboe": (_REED,),
    "accordion": (_ORGAN,), "harmonica": (_ORGAN,), "organ": (_ORGAN,),
    "bell": (_CHROM, range(112, 113)), "marimba": (_CHROM,), "ocarina": (_PIPE,), "pipe": (_PIPE,),
}


@dataclass(frozen=True)
class Row:
    rwcid: str
    track: int
    name: str
    offset: int
    excluded: bool
    comment: str  # annotation text: kept in memory, never written to anything committed


@dataclass(frozen=True)
class Note:
    onset: float
    offset: float
    midi: int
    midi_sung: int
    mask: str
    source: str


def _data_paths(data: Path) -> dict[str, Path]:
    root = data / "rwc"
    pre = root / "annotations" / "01_annotations_preprocessed"
    return {"aligned": pre / "MIDI_aligned", "vocals": pre / "MIDI_aligned" / "metadata_mapping_vocals.csv",
            "f0": pre / "melody" / "RWC-P", "voca_inst": root / "archive" / "AIST_RWC-MDB-P-2001_VOCA_INST",
            "midi_sync": root / "archive" / "AIST_RWC-MDB-P-2001_MIDI_SYNC", "refs": data / "refs"}


def mapping_rows(data: Path) -> dict[str, list[Row]]:
    """Vocal-mapping rows per RWCID, with the exclusion rule applied (and P026's exception)."""
    with open(_data_paths(data)["vocals"], encoding="utf-8") as handle:
        raw = list(csv.DictReader(handle, delimiter=";"))
    rows: dict[str, list[Row]] = defaultdict(list)
    for r in raw:
        rows[r["RWCID"]].append(Row(r["RWCID"], int(r["MIDITrackNo"]), r["MIDITrackName"].strip(),
                                    int(r["SemitoneOffset"]), bool(EXCLUDE_ROW.search(r["comment"] or "")),
                                    r["comment"] or ""))
    return dict(rows)


def included(rows: list[Row]) -> tuple[list[Row], bool]:
    """(the rows V1 uses, flagged) — all rows are kept and flagged when the rule would leave none."""
    kept = [r for r in rows if not r.excluded]
    return (kept, False) if kept else (list(rows), True)


def track_notes(midi: rwc_midi.MidiFile, row: Row) -> list[rwc_midi.MidiNote]:
    """The notes of a mapping row's track; ValueError when the track's name doesn't match the mapping."""
    if row.track >= len(midi.tracks):
        raise ValueError(f"{row.rwcid}: track {row.track} doesn't exist")
    track = midi.tracks[row.track]
    if track.name.strip() != row.name:
        raise ValueError(f"{row.rwcid}: track {row.track} is named {track.name.strip()!r}, the mapping says "
                         f"{row.name!r}")
    return track.notes


def vocal_notes(midi: rwc_midi.MidiFile, rows: list[Row], sign: int) -> list[Note]:
    notes = []
    for row in rows:
        for n in track_notes(midi, row):
            notes.append(Note(n.onset, n.offset, n.pitch, n.pitch + sign * row.offset, "", f"track:{row.track}"))
    return sorted(notes, key=lambda n: (n.onset, n.midi, n.source))


def hz_to_midi(hz: float) -> float:
    return 69.0 + 12.0 * math.log2(hz / 440.0)


def read_f0(path: Path) -> list[tuple[float, float]]:
    """RWC-P's F0 annotation: (time, Hz) rows; 0 Hz is unvoiced."""
    out = []
    with open(path, encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter=";"):
            out.append((float(row["t"]), float(row["f0"])))
    return out


def f0_offset(f0: list[tuple[float, float]], notes: list[rwc_midi.MidiNote]) -> float | None:
    """Median of (F0 in MIDI - MIDI note) over voiced annotation frames inside the notes."""
    import bisect

    times = [t for t, _ in f0]
    diffs = []
    for n in notes:
        i = bisect.bisect_left(times, n.onset)
        while i < len(f0) and f0[i][0] < n.offset:
            if f0[i][1] > 0:
                diffs.append(hz_to_midi(f0[i][1]) - n.pitch)
            i += 1
    return statistics.median(diffs) if diffs else None


def sign_check(medians: dict[str, tuple[float | None, int]], tolerance: float = 1.0) -> int:
    """+1 or -1: the sign s with |median - s*offset| <= 1 semitone for every piece; ValueError otherwise."""
    for sign in (1, -1):
        bad = [rid for rid, (median, offset) in medians.items()
               if median is None or abs(median - sign * offset) > tolerance]
        if not bad:
            return sign
    plus = [rid for rid, (m, o) in medians.items() if m is None or abs(m - o) > tolerance]
    minus = [rid for rid, (m, o) in medians.items() if m is None or abs(m + o) > tolerance]
    raise ValueError(f"SemitoneOffset sign check failed: + fails on {len(plus)} pieces ({', '.join(plus[:8])}), "
                     f"- fails on {len(minus)} ({', '.join(minus[:8])})")


def voca_inst(path: Path, end: float) -> list[tuple[float, float, str]]:
    """VOCA_INST segments (start, end, normalised label). Each line starts a segment that runs to the next."""
    marks = []
    for line in Path(path).read_bytes().decode("cp932", errors="replace").splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split()
        try:
            time = float(parts[0])
        except ValueError:
            continue
        label = parts[1] if len(parts) > 1 else ""
        label = "".join(ch for ch in label if ch.isascii()).lower().replace("s;", "s:", 1)
        if label.startswith("s:"):
            label = "s:" + LABEL_FIXES.get(label[2:], label[2:])
        marks.append((time, label))
    marks.sort()
    return [(t, marks[i + 1][0] if i + 1 < len(marks) else end, label) for i, (t, label) in enumerate(marks)]


def skyline(notes: list[rwc_midi.MidiNote], window: float = SKYLINE_SECONDS) -> list[rwc_midi.MidiNote]:
    """Of notes starting within ``window`` of a group's first onset, keep the highest (then the earliest)."""
    out, group = [], []
    for n in sorted(notes, key=lambda n: (n.onset, -n.pitch)):
        if group and n.onset - group[0].onset > window:
            out.append(max(group, key=lambda g: (g.pitch, -g.onset)))
            group = []
        group.append(n)
    if group:
        out.append(max(group, key=lambda g: (g.pitch, -g.onset)))
    return out


def _in(notes, start, end):
    return [n for n in notes if start <= n.onset < end]


def solo_track(midi: rwc_midi.MidiFile, instrument: str, start: float, end: float,
               vocal_tracks: set[int]) -> tuple[int, int] | None:
    """(tier, track index) for a solo segment, or None when no track plays in it."""
    families = GM_FAMILIES.get(instrument, ())
    candidates = [t for t in midi.tracks if t.index not in vocal_tracks and not t.is_drums and _in(t.notes, start, end)]
    tiers = [
        [t for t in candidates if "SOLO" in t.name.upper()],
        [t for t in candidates if t.program is not None and any(t.program in f for f in families)],
        candidates,
    ]
    for tier, tracks in enumerate(tiers, 1):
        if tracks:
            def key(t):
                inside = _in(t.notes, start, end)
                return (len(inside), statistics.median(n.pitch for n in inside), -t.index)
            return tier, max(tracks, key=key).index
    return None


def solo_notes(midi: rwc_midi.MidiFile, segments: list[tuple[float, float, str]], vocal_tracks: set[int],
               use_skyline: bool = True) -> tuple[list[Note], list[dict]]:
    """The solos of F1 (or F2 without the skyline), and a log of the choices."""
    notes, log = [], []
    for start, end, label in segments:
        if not label.startswith("s"):
            continue
        instrument = label[2:] if label.startswith("s:") else ""
        choice = solo_track(midi, instrument, start, end, vocal_tracks)
        if choice is None:
            log.append({"start": start, "end": end, "instrument": instrument, "tier": None, "track": None, "notes": 0})
            continue
        tier, index = choice
        chosen = _in(midi.tracks[index].notes, start, end)
        if use_skyline:
            chosen = skyline(chosen)
        for n in chosen:
            notes.append(Note(n.onset, n.offset, n.pitch, n.pitch, "", f"solo:{index}:{instrument or '?'}"))
        log.append({"start": start, "end": end, "instrument": instrument, "tier": tier, "track": index,
                    "notes": len(chosen)})
    return notes, log


def write_tsv(path: Path, notes: list[Note]) -> dict:
    """Write a reference TSV (sorted by onset); return its file record relative to the data folder's parent."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(notes, key=lambda n: (n.onset, n.midi, n.source))
    lines = ["\t".join(COLUMNS)]
    for i, n in enumerate(ordered):
        lines.append(f"{i}\t{n.onset:.6f}\t{n.offset:.6f}\t{n.midi}\t{n.midi_sung}\t{n.mask}\t{n.source}")
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    tmp.replace(path)
    return {"sha256": fetch.sha256_file(path), "notes": len(ordered)}


def _record(path: Path, data: Path, info: dict) -> dict:
    return {"path": Path(path).relative_to(data).as_posix(), "sha256": info["sha256"]}


@dataclass
class Built:
    references: dict[str, dict[str, dict]]  # item -> name -> file record
    summary: dict


def build_rwc(data: Path, manifests: dict[str, dict]) -> Built:
    """Every RWC reference and variant; returns file records and aggregate counts (numbers only)."""
    where = _data_paths(data)
    rows = mapping_rows(data)
    pop = [i for i in manifests["rwc-pop"]["items"]]
    other = [i for i in manifests["rwc-other"]["items"]]
    midi_cache: dict[str, rwc_midi.MidiFile] = {}

    def aligned(rwcid: str) -> rwc_midi.MidiFile:
        if rwcid not in midi_cache:
            midi_cache[rwcid] = rwc_midi.read(where["aligned"] / f"RWC-{rwcid[4]}" / f"{rwcid}.mid")
        return midi_cache[rwcid]

    # The sign of SemitoneOffset, from RWC-P's F0 annotations.
    medians = {}
    for item in pop:
        rid = item["source_id"]
        kept, _ = included(rows[rid])
        notes = [n for r in kept for n in track_notes(aligned(rid), r)]
        offsets = {r.offset for r in kept}
        medians[rid] = (f0_offset(read_f0(where["f0"] / f"{rid}.csv"), notes), max(offsets))
    sign = sign_check(medians)

    refs: dict[str, dict[str, dict]] = defaultdict(dict)
    summary = {"sign": sign, "sign_check": {}, "sets": {}, "flagged": [], "excluded_rows": 0,
               "zero_length_dropped": 0, "duplicates_collapsed": 0, "solo": {"segments": 0, "seconds": 0.0,
                                                                              "tiers": {1: 0, 2: 0, 3: 0, "none": 0}}}
    nonzero = [abs(m - o) for m, o in medians.values() if m is not None and o != 0]
    summary["sign_check"] = {"pieces": len(medians), "pieces_with_offset": len(nonzero),
                             "max_abs_error_semitones": round(max(nonzero), 3) if nonzero else None}
    counts = defaultdict(int)
    for item in pop + other:
        rid, ident = item["source_id"], item["id"]
        midi = aligned(rid)
        counts_key = "rwc-pop" if item in pop else "rwc-other"
        kept, flagged = included(rows[rid])
        if flagged:
            summary["flagged"].append(ident)
        summary["excluded_rows"] += 0 if flagged else sum(1 for r in rows[rid] if r.excluded)
        summary["zero_length_dropped"] += midi.counts["zero_length_dropped"]
        summary["duplicates_collapsed"] += midi.counts["duplicates_collapsed"]
        v1 = vocal_notes(midi, kept, sign)
        name = "rwc-pop-vocal" if counts_key == "rwc-pop" else "rwc-other-vocal"
        info = write_tsv(where["refs"] / name / f"{ident}.tsv", v1)
        refs[ident]["vocal"] = _record(where["refs"] / name / f"{ident}.tsv", data, info)
        counts[f"{name}:notes"] += info["notes"]
        counts[f"{name}:items"] += 1
        counts[f"{name}:empty"] += info["notes"] == 0
        v2 = vocal_notes(midi, rows[rid], sign)
        write_tsv(where["refs"] / "variants" / "V2" / f"{ident}.tsv", v2)
        provenance = {"item": ident, "rows": [{"track": r.track, "name": r.name, "offset": r.offset,
                                               "excluded": r.excluded, "used": r in kept} for r in rows[rid]],
                      "flagged": flagged, "midi_counts": midi.counts, "sign": sign, "vocal_notes": len(v1)}
        if counts_key == "rwc-pop":
            vocal_tracks = {r.track for r in rows[rid]}
            duration = item["audio"]["seconds"]
            segments = voca_inst(where["voca_inst"] / f"RM-{rid[4:]}.VOCA_INST.TXT", duration)
            solos, log = solo_notes(midi, segments, vocal_tracks)
            solos_plain, _ = solo_notes(midi, segments, vocal_tracks, use_skyline=False)
            full = v1 + solos
            info = write_tsv(where["refs"] / "rwc-pop-full" / f"{ident}.tsv", full)
            refs[ident]["full"] = _record(where["refs"] / "rwc-pop-full" / f"{ident}.tsv", data, info)
            counts["rwc-pop-full:notes"] += info["notes"]
            counts["rwc-pop-full:solo_notes"] += len(solos)
            counts["rwc-pop-full:songs_with_solos"] += bool(solos)
            for entry in log:
                summary["solo"]["segments"] += 1
                summary["solo"]["seconds"] += entry["end"] - entry["start"]
                summary["solo"]["tiers"][entry["tier"] or "none"] += 1
            vocal_segments = [(s, e) for s, e, label in segments if label[:1] in VOCAL_LABELS]
            v4 = [n for n in v1 if any(s <= n.onset < e for s, e in vocal_segments)]
            write_tsv(where["refs"] / "variants" / "V4" / f"{ident}.tsv", v4)
            write_tsv(where["refs"] / "variants" / "F2" / f"{ident}.tsv", v1 + solos_plain)
            write_tsv(where["refs"] / "variants" / "F3" / f"{ident}.tsv", v4 + solos)
            named_solo = [Note(n.onset, n.offset, n.pitch, n.pitch, "", f"solo:{t.index}:named")
                          for t in midi.tracks if "SOLO" in t.name.upper() and t.index not in vocal_tracks
                          for n in t.notes]
            write_tsv(where["refs"] / "variants" / "F4" / f"{ident}.tsv", v1 + named_solo)
            v3_status = _write_v3(where, rid, ident, kept, sign)
            provenance.update({"solo_segments": log, "full_notes": len(full), "v4_notes": len(v4), "v3": v3_status})
        jobs_json(where["refs"] / "provenance" / f"{ident}.json", provenance)
    summary["sets"] = dict(counts)
    summary["solo"]["seconds"] = round(summary["solo"]["seconds"], 1)
    summary["solo"]["tiers"] = {str(k): v for k, v in summary["solo"]["tiers"].items()}
    return Built(dict(refs), summary)


def _write_v3(where: dict[str, Path], rid: str, ident: str, kept: list[Row], sign: int) -> str:
    """V3: the 2006 MIDI_SYNC file, V1's tracks found by name (V1's index when names repeat)."""
    path = where["midi_sync"] / f"RM-{rid[4:]}.SMF_SYNC.MID"
    if not path.exists():
        return "no MIDI_SYNC file"
    midi = rwc_midi.read(path)
    notes = []
    for row in kept:
        matches = [t for t in midi.tracks if t.name.strip() == row.name]
        if not matches:
            return f"track {row.name!r} not found"
        track = next((t for t in matches if t.index == row.track), matches[0])
        notes += [Note(n.onset, n.offset, n.pitch, n.pitch + sign * row.offset, "", f"track:{track.index}")
                  for n in track.notes]
    write_tsv(where["refs"] / "variants" / "V3" / f"{ident}.tsv", notes)
    return "ok"


def jobs_json(path: Path, data: dict) -> None:
    from .jobs import write_json

    write_json(path, data)


def build_usdx(data: Path, manifest: dict, lyrics_dir: Path) -> Built:
    """Karaoke references (every chart note, freestyle masked) and the words files for the lyrics condition."""
    from . import ultrastar

    where = _data_paths(data)
    refs: dict[str, dict[str, dict]] = defaultdict(dict)
    counts = defaultdict(int)
    for item in manifest["items"]:
        if item["status"].startswith("excluded"):
            continue
        chart = ultrastar.parse(data / item["annotations"][0]["path"])
        rows = ultrastar.reference_rows(chart)
        notes = [Note(r["onset"], r["offset"], r["midi"], r["midi_sung"], r["mask"], r["source"]) for r in rows]
        path = where["refs"] / "usdx" / f"{item['id']}.tsv"
        info = write_tsv(path, notes)
        refs[item["id"]]["vocal"] = _record(path, data, info)
        words = ultrastar.write_lyrics(chart, Path(lyrics_dir) / f"{item['id']}.txt")
        key = "usdx-32" if item["status"] == "scored" else "usdx-ht4"
        counts[f"{key}:items"] += 1
        counts[f"{key}:notes"] += len(notes)
        counts[f"{key}:freestyle"] += sum(1 for n in notes if n.mask)
        counts[f"{key}:lyric_lines"] += words["lines"]
    return Built(dict(refs), {"sets": dict(counts)})


def update_manifests(manifests: dict[str, dict], refs: dict[str, dict[str, dict]]) -> None:
    for manifest in manifests.values():
        for item in manifest["items"]:
            found = refs.get(item["id"])
            if not found:
                continue
            item["reference"] = found["vocal"]
            if "full" in found:
                item["reference_full"] = found["full"]


def main(argv: list[str] | None = None) -> int:
    from . import manifests as manifest_module

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--build-all", action="store_true")
    parser.add_argument("--summary", type=Path, default=None, help="write the aggregate counts as JSON")
    args = parser.parse_args(argv)
    if not args.build_all:
        parser.error("nothing to do; use --build-all")
    data = paths.data_dir()
    manifests = manifest_module.load_all()
    rwc = build_rwc(data, manifests)
    usdx = build_usdx(data, manifests["usdx"], paths.out_dir() / "lyrics")
    update_manifests(manifests, {**rwc.references, **usdx.references})
    for manifest in manifests.values():
        manifest_module.write(manifest)
    summary = {"rwc": rwc.summary, "usdx": usdx.summary}
    print(json.dumps(summary, indent=1, default=str))
    if args.summary:
        args.summary.write_text(json.dumps(summary, indent=1, default=str) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
