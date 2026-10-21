"""RWC's MIDI files through mido: a tempo map, and each track's notes in seconds.

TEST-ONLY. RWC is CC BY-NC (evaluation/probe-eval only, never training). Not for product/training import.

Only note, program, tempo and track-name events are read. Text and lyric events (RWC's track 0 carries
them) are never read or printed.

Pairing follows pretty_midi, which SheetSage2's environment ships: a note-off (or a note-on with velocity 0)
closes every open note of that channel and pitch that started on an earlier tick, and a note started on the
same tick stays open (it's the next note). Unlike pretty_midi, when a note-off finds only same-tick notes to
close, those are zero-length notes: they're dropped and counted instead of staying open. Exact duplicates
(same track, onset tick and pitch) are collapsed to one note and counted.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class MidiNote:
    track: int
    channel: int
    pitch: int
    velocity: int
    onset: float
    offset: float
    onset_tick: int
    offset_tick: int


@dataclass
class MidiTrack:
    index: int
    name: str
    program: int | None
    channels: set[int] = field(default_factory=set)
    notes: list[MidiNote] = field(default_factory=list)

    @property
    def is_drums(self) -> bool:
        """Channel 10 (index 9), or a name with DRUM or PERC in it."""
        upper = self.name.upper()
        return 9 in self.channels or "DRUM" in upper or "PERC" in upper


@dataclass
class MidiFile:
    tracks: list[MidiTrack]
    ticks_per_beat: int
    tempos: list[tuple[int, int]]  # (tick, microseconds per beat)
    counts: dict[str, int]


class TempoMap:
    """Ticks to seconds through every ``set_tempo`` event (default 120 BPM before the first)."""

    def __init__(self, tempos: list[tuple[int, int]], ticks_per_beat: int):
        changes = sorted(tempos)
        if not changes or changes[0][0] != 0:
            changes = [(0, 500000)] + changes
        self.ticks_per_beat = ticks_per_beat
        self.ticks, self.tempos, self.seconds = [], [], []
        elapsed = 0.0
        for i, (tick, tempo) in enumerate(changes):
            if i:
                prev_tick, prev_tempo = changes[i - 1]
                elapsed += (tick - prev_tick) * prev_tempo / 1e6 / ticks_per_beat
            if self.ticks and self.ticks[-1] == tick:  # a later change at the same tick wins
                self.tempos[-1], self.seconds[-1] = tempo, elapsed
                continue
            self.ticks.append(tick)
            self.tempos.append(tempo)
            self.seconds.append(elapsed)

    def time(self, tick: int) -> float:
        import bisect

        i = bisect.bisect_right(self.ticks, tick) - 1
        return self.seconds[i] + (tick - self.ticks[i]) * self.tempos[i] / 1e6 / self.ticks_per_beat


def read(path: Path) -> MidiFile:
    """All tracks of a MIDI file, notes in seconds."""
    import mido

    mf = mido.MidiFile(str(path))
    tempos = []
    raw_tracks = []
    for index, track in enumerate(mf.tracks):
        tick = 0
        name, program, channels, events = "", None, set(), []
        for message in track:
            tick += message.time
            kind = message.type
            if kind == "set_tempo":
                tempos.append((tick, message.tempo))
            elif kind == "track_name" and not name:
                name = message.name
            elif kind == "program_change":
                if program is None:
                    program = message.program
                channels.add(message.channel)
            elif kind in ("note_on", "note_off"):
                channels.add(message.channel)
                events.append((tick, kind, message.channel, message.note, message.velocity))
        raw_tracks.append((index, name, program, channels, events))
    tempo_map = TempoMap(tempos, mf.ticks_per_beat)
    counts = {"zero_length_dropped": 0, "duplicates_collapsed": 0, "unterminated": 0}
    tracks = []
    for index, name, program, channels, events in raw_tracks:
        open_notes: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
        pairs: list[tuple[int, int, int, int, int]] = []
        for tick, kind, channel, pitch, velocity in events:
            key = (channel, pitch)
            if kind == "note_on" and velocity > 0:
                open_notes[key].append((tick, velocity))
                continue
            started = open_notes.get(key, [])
            close = [(s, v) for s, v in started if s != tick]
            keep = [(s, v) for s, v in started if s == tick]
            for start, vel in close:
                pairs.append((start, tick, channel, pitch, vel))
            if close:
                # pretty_midi: a note started on this very tick is the next note; it stays open
                open_notes[key] = keep
            else:
                # nothing earlier to close: the same-tick notes are zero-length; drop and count them
                counts["zero_length_dropped"] += len(keep)
                open_notes[key] = []
        counts["unterminated"] += sum(len(v) for v in open_notes.values())
        seen, notes = set(), []
        for start, end, channel, pitch, vel in sorted(pairs):
            if (start, pitch) in seen:
                counts["duplicates_collapsed"] += 1
                continue
            seen.add((start, pitch))
            notes.append(MidiNote(index, channel, pitch, vel, tempo_map.time(start), tempo_map.time(end), start, end))
        tracks.append(MidiTrack(index, name, program, channels, notes))
    return MidiFile(tracks, mf.ticks_per_beat, sorted(tempos), counts)
