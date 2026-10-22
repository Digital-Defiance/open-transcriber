"""Run MuScriptor (Kyutai and Mirelo) over the dev sets and drop its melody line onto the scoreboard.

MuScriptor is a general multi-instrument transcriber: its ``transcribe`` CLI writes one Standard MIDI File
per input, with every instrument on its own track and a tempo map. The scoreboard scores the *vocal line*, so
:func:`melody_line` reads the MIDI, turns ticks into seconds through the file's tempo map, and reduces the
polyphony to a single monophonic line the same way the board's ``mono_bp`` reduces Basic Pitch: keep notes in
the singing range, resolve overlaps by velocity with a small cost for large leaps, merge same-pitch
fragments, and drop very short notes. The result is written as a board drop-in TSV (``onset offset midi``), so
the board picks it up with no code change.

We score MuScriptor on both frozen dev sets, in all three model sizes, on the full mixture and on the Demucs
vocal stem: six systems named ``muscriptor-<size>-<mix|stem>``, landing in
``out/judge/systems/muscriptor-<size>-<mix|stem>/<id>.tsv``.

**Licence (testing only).** MuScriptor's code is MIT but its released weights are CC BY-NC 4.0
(non-commercial). It is a yardstick only: never run on the owner's songs, never shipped, and for the
non-commercial test songs only scores and credits are published, never audio or transcriptions. Downloading
the official weights requires accepting the CC BY-NC 4.0 licence on Hugging Face (a genuine owner decision).

A tiny standard-library SMF reader is used on purpose, so the converter has no MIDI-library dependency and can
be tested against a MIDI file synthesised in the test.

Content was rephrased for compliance with licensing restrictions.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

from open_transcriber.bakeoff import paths
from open_transcriber.board.game import (
    SIZES,
    SOURCES,
    Timing,
    load_timings,
    save_timings,
    source_path,
    write_system_tsv,
)

# Re-export the shared pieces callers expect from a board runner module.
__all__ = [
    "SIZES",
    "SOURCES",
    "Timing",
    "config_name",
    "read_midi_notes",
    "melody_line",
    "convert_midi",
    "convert_dir",
    "source_path",
    "write_system_tsv",
    "load_timings",
    "save_timings",
]

#: The singing range MuScriptor's melody line is kept within (same as ``mono_bp``): A2..C6.
MELODY_LO = 45
MELODY_HI = 84


def config_name(size: str, source: str) -> str:
    """The board system name for a (size, source) run, e.g. ``muscriptor-small-stem``."""
    if size not in SIZES:
        raise ValueError(f"unknown MuScriptor size {size!r}; expected one of {SIZES}")
    if source not in SOURCES:
        raise ValueError(f"unknown source {source!r}; expected one of {SOURCES}")
    return f"muscriptor-{size}-{source}"


# --- a small standard-library SMF reader ------------------------------------------------------------------

@dataclass
class MidiNote:
    """One MIDI note after tempo resolution: onset and offset in seconds, integer pitch, velocity."""

    onset: float
    offset: float
    midi: int
    velocity: int


def _read_vlq(data: bytes, pos: int) -> tuple[int, int]:
    """Read a MIDI variable-length quantity at ``pos``; return (value, next position)."""
    value = 0
    while True:
        byte = data[pos]
        pos += 1
        value = (value << 7) | (byte & 0x7F)
        if not byte & 0x80:
            return value, pos


def read_midi_notes(path: Path) -> list[MidiNote]:
    """Read every note of a Standard MIDI File, with ticks turned into seconds through its tempo map.

    Supports format 0 and 1 (several tracks sharing one tempo map), the common case MuScriptor writes.
    Note-on with velocity 0 is treated as note-off. Drum notes (channel 9) are skipped, since they are not a
    melody. The returned notes are sorted by onset.
    """
    raw = Path(path).read_bytes()
    if raw[:4] != b"MThd":
        raise ValueError(f"{path}: not a Standard MIDI File (missing MThd header)")
    (_hdr_len,) = struct.unpack(">I", raw[4:8])
    _fmt, _ntrk, division = struct.unpack(">HHH", raw[8:14])
    if division & 0x8000:
        raise ValueError(f"{path}: SMPTE time division is not supported")
    ticks_per_beat = division
    pos = 8 + _hdr_len

    # Collect (abs_tick, kind, pitch, velocity) events and tempo changes across all tracks.
    note_events: list[tuple[int, str, int, int]] = []
    tempo_events: list[tuple[int, int]] = [(0, 500000)]  # (abs_tick, microseconds per beat); default 120 BPM
    while pos < len(raw):
        if raw[pos : pos + 4] != b"MTrk":
            break
        (track_len,) = struct.unpack(">I", raw[pos + 4 : pos + 8])
        pos += 8
        end = pos + track_len
        abs_tick = 0
        running_status = 0
        while pos < end:
            delta, pos = _read_vlq(raw, pos)
            abs_tick += delta
            status = raw[pos]
            if status & 0x80:
                pos += 1
                running_status = status
            else:
                status = running_status
            event = status & 0xF0
            channel = status & 0x0F
            if status == 0xFF:  # meta
                meta_type = raw[pos]
                pos += 1
                length, pos = _read_vlq(raw, pos)
                if meta_type == 0x51 and length == 3:  # set tempo
                    tempo = int.from_bytes(raw[pos : pos + 3], "big")
                    tempo_events.append((abs_tick, tempo))
                pos += length
            elif status in (0xF0, 0xF7):  # sysex
                length, pos = _read_vlq(raw, pos)
                pos += length
            elif event in (0x80, 0x90):  # note off / note on
                pitch = raw[pos]
                velocity = raw[pos + 1]
                pos += 2
                if channel == 9:  # percussion: never a melody
                    continue
                if event == 0x90 and velocity > 0:
                    note_events.append((abs_tick, "on", pitch, velocity))
                else:
                    note_events.append((abs_tick, "off", pitch, 0))
            elif event in (0xA0, 0xB0, 0xE0):  # 2-byte channel messages
                pos += 2
            elif event in (0xC0, 0xD0):  # 1-byte channel messages
                pos += 1
            else:
                pos += 1
        pos = end

    tempo_events.sort()

    def tick_to_seconds(tick: int) -> float:
        seconds = 0.0
        last_tick = 0
        tempo = tempo_events[0][1]
        for ev_tick, ev_tempo in tempo_events:
            if ev_tick >= tick:
                break
            seconds += (ev_tick - last_tick) / ticks_per_beat * (tempo / 1_000_000)
            last_tick = ev_tick
            tempo = ev_tempo
        seconds += (tick - last_tick) / ticks_per_beat * (tempo / 1_000_000)
        return seconds

    # Pair note-on with the next note-off of the same pitch (stable within a pitch's event order).
    open_notes: dict[int, list[tuple[int, int]]] = {}
    notes: list[MidiNote] = []
    for abs_tick, kind, pitch, velocity in sorted(note_events, key=lambda e: (e[0], e[1] != "on")):
        if kind == "on":
            open_notes.setdefault(pitch, []).append((abs_tick, velocity))
        else:
            stack = open_notes.get(pitch)
            if not stack:
                continue
            on_tick, on_vel = stack.pop(0)
            onset = tick_to_seconds(on_tick)
            offset = tick_to_seconds(abs_tick)
            if offset > onset:
                notes.append(MidiNote(onset, offset, int(pitch), int(on_vel)))
    notes.sort(key=lambda n: (n.onset, n.offset))
    return notes


# --- melody reduction (mirrors the board's mono_bp, velocity in place of amplitude) -----------------------

def melody_line(notes: list[MidiNote], *, min_dur: float = 0.06, merge_gap: float = 0.03,
                jump: float = 0.04, lo: int = MELODY_LO, hi: int = MELODY_HI) -> list[tuple[float, float, int]]:
    """Reduce MuScriptor's polyphony to one melody line, as ``mono_bp`` does for Basic Pitch.

    Keep notes inside the singing range and above the minimum duration, prefer the louder note when two
    overlap (with a cost for large leaps from the previous pitch), merge same-pitch fragments across small
    gaps, and drop the very short leftovers. Velocity stands in for Basic Pitch's amplitude.
    """
    kept = [n for n in notes if lo <= n.midi <= hi and n.offset - n.onset >= min_dur]
    kept.sort(key=lambda n: (n.onset, -n.velocity))
    line: list[list[float]] = []  # [onset, offset, midi, velocity]
    for n in kept:
        if line and n.onset < line[-1][1] - 0.02:
            prev = line[-2][2] if len(line) > 1 else line[-1][2]
            cur = line[-1]
            score_new = n.velocity - jump * abs(n.midi - prev)
            score_cur = cur[3] - jump * abs(cur[2] - prev)
            if score_new > score_cur and n.onset - cur[0] > 0.05:
                line[-1] = [cur[0], n.onset, cur[2], cur[3]]
                line.append([n.onset, n.offset, n.midi, n.velocity])
            elif score_new > score_cur:
                line[-1] = [n.onset, n.offset, n.midi, n.velocity]
            continue
        if line and n.midi == line[-1][2] and n.onset - line[-1][1] < merge_gap and n.velocity <= line[-1][3] * 1.15:
            line[-1][1] = max(line[-1][1], n.offset)
            line[-1][3] = max(line[-1][3], n.velocity)
            continue
        line.append([n.onset, n.offset, n.midi, n.velocity])
    rows = sorted((a, b, int(p)) for a, b, p, _ in line if b - a >= min_dur)
    return rows


def convert_midi(midi_path: Path, dest: Path) -> list[tuple[float, float, int]]:
    """Read a MuScriptor MIDI, extract the melody line, and write a board drop-in TSV; return the rows."""
    rows = melody_line(read_midi_notes(midi_path))
    write_system_tsv(rows, dest)
    return rows


def convert_dir(raw_dir: Path, system_dir: Path, items: list[str]) -> list[str]:
    """Convert every ``<item>.mid`` in ``raw_dir`` to ``<item>.tsv`` in ``system_dir``; return items done."""
    done = []
    for item in items:
        midi = raw_dir / f"{item}.mid"
        if not midi.exists():
            midi = raw_dir / f"{item}.midi"
        if not midi.exists():
            continue
        convert_midi(midi, system_dir / f"{item}.tsv")
        done.append(item)
    return done
