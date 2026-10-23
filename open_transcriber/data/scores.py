"""A data-only, seeded sampler of copyist-song/0 melodies for training data.

This writes small, singable lead sheets covering a wide musical range (all 24
keys, five meters, a 50-180 tempo span, and a spread of rhythmic and melodic
features) so the transcriber can be trained on melodies whose every note label
is exact. It is NOT the product's melody writer: it never imports
``copyist.ruleswrite`` and reimplements its own bar-length logic. Pitch
spelling and ABC assembly reuse ``copyist.abcplan`` (shared helpers, not the
rule writer).

Determinism: every number comes from a single ``random.Random(seed)``, so the
same seed always yields byte-identical YAML. ``write_batch`` runs each song
through Copyist's validators and the held-out guard, re-rolling deterministically
on a FAIL or a leak, and writes into a gitignored cache directory. Generated
songs are training data and are never committed.
"""
from __future__ import annotations

import random
import tempfile
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

from music21 import pitch as m21pitch

from copyist import abcplan
from copyist.lyrics import note_target
from copyist.song import load_song
from copyist.validate import Finding, is_finished, validate

from . import heldout as heldout_mod

UNIT = Fraction(1, 8)  # every song is written in eighth-note units (L:1/8)

# All 24 keys: 12 major (uppercase letter) and 12 minor (letter + "m").
MAJOR_KEYS = ("C", "G", "D", "A", "E", "B", "F#", "C#", "F", "Bb", "Eb", "Ab")
MINOR_KEYS = ("Am", "Em", "Bm", "F#m", "C#m", "G#m", "D#m", "Dm", "Gm", "Cm", "Fm", "Bbm")
ALL_KEYS = MAJOR_KEYS + MINOR_KEYS

METERS = ("2/4", "3/4", "4/4", "6/8", "12/8")

# Voice ranges as (low, high) note names; the melody is clamped into one of these.
VOICE_RANGES = {
    "soprano": ("C4", "A5"),
    "alto": ("G3", "E5"),
    "tenor": ("C3", "A4"),
    "bass": ("E2", "D4"),
}

# Diatonic triads by 0-based scale degree (root, third, fifth indices into the scale).
_TRIAD_DEGREES = ((0, 2, 4), (1, 3, 5), (2, 4, 6), (3, 5, 0), (4, 6, 1), (5, 0, 2), (6, 1, 3))
# Chord quality suffix for each diatonic degree, major vs minor key.
_MAJ_QUALITY = ("", "m", "m", "", "", "m", "dim")
_MIN_QUALITY = ("m", "dim", "", "m", "m", "", "")
# Short, singable diatonic progressions (1-based scale degrees), looped per song.
PROGRESSIONS = (
    (1, 5, 6, 4),
    (1, 4, 5, 1),
    (1, 6, 4, 5),
    (1, 4, 1, 5),
    (1, 5, 1, 4),
    (1, 6, 2, 5),
)


@dataclass
class SamplerConfig:
    """Knobs for the sampler: the musical range and how often each feature appears."""
    keys: tuple[str, ...] = ALL_KEYS
    meters: tuple[str, ...] = METERS
    tempo_range: tuple[int, int] = (50, 180)
    voice_ranges: dict[str, tuple[str, str]] = field(default_factory=lambda: dict(VOICE_RANGES))
    bars_range: tuple[int, int] = (6, 12)
    # Feature-toggle probabilities (per song or per note as noted in scores.py).
    p_sixteenths: float = 0.5
    p_dotted: float = 0.5
    p_syncopation: float = 0.45
    p_ties: float = 0.4
    p_pickup: float = 0.4
    p_rests: float = 0.5
    p_melisma: float = 0.5        # per song: does it use any melisma
    melisma_len: tuple[int, int] = (2, 4)  # notes per melisma syllable, inclusive
    p_patter: float = 0.4         # a run of repeated-pitch quick notes
    p_long_hold: float = 0.5      # a bar-filling held note somewhere
    p_octave_leap: float = 0.4
    p_chromatic: float = 0.45     # chromatic neighbour/passing tones


DEFAULT = SamplerConfig()


@dataclass
class SampledSong:
    """A generated song: its YAML text and a record of what it contains."""
    yaml_text: str
    meta: dict


def _bar_units(meter: str) -> Fraction:
    """A bar's written length in UNIT (eighth-note) steps. Local; does not import ruleswrite."""
    num, den = (int(x) for x in meter.split("/"))
    return Fraction(num, den) / UNIT


def _scale_pcs(key: str) -> list[int]:
    """Pitch classes of the key's scale (major, or natural minor for a minor key)."""
    root = (abcplan.NATURAL[abcplan._key_parts(key)[0]]
            + {"#": 1, "b": -1, "": 0}[abcplan._key_parts(key)[1]]) % 12
    parts = abcplan._key_parts(key)
    minor = parts[2].lower().startswith("m")
    steps = (0, 2, 3, 5, 7, 8, 10) if minor else (0, 2, 4, 5, 7, 9, 11)
    return [(root + s) % 12 for s in steps]


def _is_minor(key: str) -> bool:
    return abcplan._key_parts(key)[2].lower().startswith("m")


def _chord_symbol(key: str, degree_1: int) -> str:
    """Diatonic chord symbol for a 1-based scale degree in the key."""
    scale = _scale_pcs(key)
    i = (degree_1 - 1) % 7
    root_pc = scale[i]
    quality = (_MIN_QUALITY if _is_minor(key) else _MAJ_QUALITY)[i]
    letter, alter, _oct = abcplan._spell(60 + root_pc, key)
    root = letter + abcplan.ACC_TEXT.get(alter, "").replace("=", "")
    return root + quality


def _deg_midi(scale: list[int], degree_0: int, target: int, low: int, high: int) -> int:
    """MIDI note for a 0-based scale degree, nearest ``target`` and clamped to [low, high]."""
    pc = scale[degree_0 % 7]
    base = target - (target % 12) + pc
    m = min((base - 12, base, base + 12), key=lambda x: abs(x - target))
    while m < low:
        m += 12
    while m > high:
        m -= 12
    return m


def _rhythm_cells(meter: str, cfg: SamplerConfig, rng: random.Random) -> list[list[Fraction]]:
    """Candidate one-bar rhythm patterns (lists of UNIT-lengths summing to the bar)."""
    bar = _bar_units(meter)
    compound = meter.endswith("/8")
    cells: list[list[Fraction]] = []
    if compound:
        # Groups of three eighths (a dotted-quarter beat).
        groups = int(bar) // 3
        cells.append([Fraction(3)] * groups)                       # dotted-quarter beats
        cells.append([Fraction(2), Fraction(1)] * groups)          # quarter + eighth per beat
        cells.append([Fraction(1), Fraction(1), Fraction(1)] * groups)  # three eighths per beat
        if cfg.p_dotted and rng.random() < cfg.p_dotted:
            cells.append(([Fraction(2), Fraction(1)] + [Fraction(3)] * (groups - 1)) if groups > 1
                         else [Fraction(2), Fraction(1)])
    else:
        # Simple meter: fill `bar` eighths.
        n = int(bar)
        cells.append([Fraction(2)] * (n // 2) + ([Fraction(1)] * (n % 2)))  # quarters
        cells.append([Fraction(1)] * n)                                      # all eighths
        cells.append([Fraction(2)] * (n // 2 - 1) + [Fraction(1), Fraction(1)] + ([Fraction(2)] if n % 2 == 0 and n >= 4 else []))
        if cfg.p_dotted and rng.random() < cfg.p_dotted and n >= 4:
            cells.append([Fraction(3), Fraction(1)] + [Fraction(2)] * ((n - 4) // 2))  # dotted quarter + eighth
        if cfg.p_sixteenths and rng.random() < cfg.p_sixteenths and n >= 2:
            cells.append([Fraction(1, 2), Fraction(1, 2), Fraction(1)] + [Fraction(2)] * ((n - 2) // 2))  # two sixteenths
        if cfg.p_syncopation and rng.random() < cfg.p_syncopation and n >= 4:
            cells.append([Fraction(1), Fraction(2), Fraction(1)] + [Fraction(2)] * ((n - 4) // 2))  # off-beat accent
    # Keep only cells that exactly fill the bar.
    cells = [c for c in cells if sum(c, Fraction(0)) == bar]
    return cells or [[Fraction(1)] * int(bar)] if not compound else cells or [[Fraction(3)] * (int(bar) // 3)]


@dataclass
class _Slot:
    """One rhythmic slot in the melody: a note (with pitch) or a rest."""
    length: Fraction
    is_rest: bool
    midi: int | None = None
    tie_to_next: bool = False


def _build_melody(key: str, meter: str, cfg: SamplerConfig, rng: random.Random,
                  low: int, high: int, meta: dict) -> tuple[list[list[_Slot]], list[str]]:
    """Build bars of slots and a parallel per-bar chord symbol list."""
    scale = _scale_pcs(key)
    bar_units = _bar_units(meter)
    n_bars = rng.randint(*cfg.bars_range)
    progression = PROGRESSIONS[rng.randrange(len(PROGRESSIONS))]

    pickup = cfg.p_pickup and rng.random() < cfg.p_pickup
    meta["pickup"] = bool(pickup)

    use_rests = cfg.p_rests and rng.random() < cfg.p_rests
    use_ties = cfg.p_ties and rng.random() < cfg.p_ties
    use_melisma = cfg.p_melisma and rng.random() < cfg.p_melisma
    use_patter = cfg.p_patter and rng.random() < cfg.p_patter
    use_long_hold = cfg.p_long_hold and rng.random() < cfg.p_long_hold
    use_octave = cfg.p_octave_leap and rng.random() < cfg.p_octave_leap
    use_chromatic = cfg.p_chromatic and rng.random() < cfg.p_chromatic

    cells = _rhythm_cells(meter, cfg, rng)
    middle = (low + high) // 2
    prev_midi = _deg_midi(scale, 0, middle, low, high)

    # Plan the bar skeletons (lengths + rest flags) first.
    bars_lengths: list[list[tuple[Fraction, bool]]] = []
    for bi in range(n_bars):
        if pickup and bi == 0:
            # A pickup bar: one short note worth one UNIT.
            bars_lengths.append([(Fraction(1), False)])
            continue
        if pickup and bi == n_bars - 1:
            # Last bar completes the pickup: fill the rest of a full bar.
            rem = bar_units - Fraction(1)
            bars_lengths.append([(rem, False)])
            continue
        if use_long_hold and bi == n_bars // 2:
            bars_lengths.append([(bar_units, False)])  # one bar-long held note
            continue
        if use_patter and bi == max(1, n_bars // 3):
            # A patter run: all shortest notes available in this meter.
            step = Fraction(1)
            count = int(bar_units / step)
            bars_lengths.append([(step, False)] * count)
            meta["patter"] = True
            continue
        cell = list(rng.choice(cells))
        slots = [(length, False) for length in cell]
        if use_rests and len(slots) > 1 and rng.random() < 0.5:
            idx = rng.randrange(len(slots))
            slots[idx] = (slots[idx][0], True)
            meta["rests"] = True
        bars_lengths.append(slots)

    # Record which rhythmic shapes appear.
    for bl in bars_lengths:
        for length, _rest in bl:
            if length == Fraction(1, 2):
                meta["sixteenths"] = True
            if length == Fraction(3):
                meta["dotted"] = True if not meter.endswith("/8") else meta.get("dotted", False)

    # Whether this song will carry a melisma (the run is chosen later, in
    # syllable-slot space, once ties are settled).
    meta["_use_melisma"] = bool(use_melisma)

    # Now assign pitches.
    bars: list[list[_Slot]] = []
    chords: list[str] = []
    note_index = 0

    for bi, bl in enumerate(bars_lengths):
        degree_1 = progression[bi % len(progression)]
        chords.append(_chord_symbol(key, degree_1))
        root_deg0 = (degree_1 - 1) % 7
        chord_degs = list(_TRIAD_DEGREES[root_deg0])
        bar_slots: list[_Slot] = []
        pos = Fraction(0)
        for (length, is_rest) in bl:
            if is_rest:
                bar_slots.append(_Slot(length, True))
                pos += length
                continue
            strong = pos == 0 or (bar_units >= 4 and pos == bar_units / 2)
            if strong:
                deg0 = rng.choice(chord_degs)
                aim = prev_midi
            else:
                # Stepwise or small leap from the previous note.
                move = rng.choice((-2, -1, 1, 2)) if not (use_octave and rng.random() < 0.15) else rng.choice((-7, 7))
                deg0 = None
                aim = prev_midi + move
            if deg0 is None:
                # Snap aim to nearest scale degree.
                target = _nearest_scale_midi(scale, aim, low, high)
            else:
                target = _deg_midi(scale, deg0, aim, low, high)
            if use_octave and note_index > 0 and rng.random() < 0.08:
                cand = target + 12 if target + 12 <= high else target - 12
                if low <= cand <= high:
                    target = cand
                    meta["octave_leap"] = True
            # Chromatic passing/neighbour tone on a weak slot.
            if use_chromatic and not strong and rng.random() < 0.18:
                alt = target + rng.choice((-1, 1))
                if low <= alt <= high and (alt % 12) not in scale:
                    target = alt
                    meta["chromatic"] = True
            bar_slots.append(_Slot(length, False, midi=target))
            prev_midi = target
            note_index += 1
        bars.append(bar_slots)

    # Ties: tie the last note of a bar to the first note of the next when they
    # share a pitch. Only when the tie's two notes are truly adjacent (no rest
    # between them), so music21 reads a tie, not two onsets across a rest.
    if use_ties:
        for bi in range(len(bars) - 1):
            cur, nxt = bars[bi], bars[bi + 1]
            if not cur or not nxt:
                continue
            last, first = cur[-1], nxt[0]
            if (not last.is_rest and not first.is_rest
                    and last.midi == first.midi and not last.tie_to_next):
                last.tie_to_next = True
                meta["ties"] = True
                break

    return bars, chords


def _syllable_slot_count(bars: list[list["_Slot"]]) -> int:
    """Notes that can take a syllable: onsets, not tie continuations or rests.

    Mirrors copyist.lyrics.lyric_slots so the generated lyric line fits exactly.
    """
    return len(_syllable_slots(bars))


def _syllable_slots(bars: list[list["_Slot"]]) -> list["_Slot"]:
    """The _Slot objects that take a syllable, in order (onsets, not tie-stops/rests)."""
    flat = [s for bar in bars for s in bar]
    out = []
    prev_tied = False
    for s in flat:
        if s.is_rest:
            prev_tied = False  # a rest ends any tie; its partner would be invalid
            continue
        if not prev_tied:
            out.append(s)
        prev_tied = s.tie_to_next
    return out


def _choose_melisma(bars: list[list["_Slot"]], scale: list[int], low: int, high: int,
                    cfg: SamplerConfig, rng: random.Random) -> tuple[int, int] | None:
    """Pick a melisma run over the syllable slots and vary its held pitches.

    Returns ``(start_slot, length)`` in syllable-slot index space, or None if the
    song is too short. The held notes are re-pitched so none repeats the pitch of
    the note before it (SoulX slurs same-pitch notes into one), keeping them
    singable as one syllable.
    """
    slots = _syllable_slots(bars)
    mlen = rng.randint(*cfg.melisma_len)
    if len(slots) < mlen + 2:
        return None
    start = rng.randrange(0, len(slots) - mlen)
    run = slots[start:start + mlen]
    for k in range(1, len(run)):
        prev, cur = run[k - 1], run[k]
        if cur.midi == prev.midi:
            up = prev.midi + 2
            down = prev.midi - 2
            cand = up if up <= high else down
            cur.midi = _nearest_scale_midi(scale, cand, low, high)
            if cur.midi == prev.midi:  # last resort: a clamped semitone step
                cur.midi = max(low, min(high, prev.midi + (1 if prev.midi < high else -1)))
    return start, mlen


def _nearest_scale_midi(scale: list[int], target: int, low: int, high: int) -> int:
    best = None
    for m in range(low, high + 1):
        if m % 12 in scale:
            if best is None or abs(m - target) < abs(best - target):
                best = m
    return best if best is not None else max(low, min(high, target))


def _length_text(length: Fraction) -> str:
    """ABC length text for a note/rest in UNIT steps (unit = 1/8)."""
    if length == 1:
        return ""
    if length.denominator == 1:
        return str(length.numerator)
    if length.numerator == 1:
        return f"/{length.denominator}"
    return f"{length.numerator}/{length.denominator}"


def _note_abc(key: str, midi: int, length: Fraction, tie: bool, bar_state: dict) -> str:
    letter, alter, octave = abcplan._spell(midi, key)
    ka = abcplan.key_accidentals(key)
    explicit = alter != ka.get(letter, 0) or bar_state.get(letter) not in (None, alter)
    acc = abcplan.ACC_TEXT.get(alter, "") if explicit else ""
    bar_state[letter] = alter
    name = letter if octave <= 4 else letter.lower()
    marks = "," * (4 - octave) if octave < 4 else "'" * (octave - 5) if octave > 5 else ""
    return f"{acc}{name}{marks}{_length_text(length)}" + ("-" if tie else "")


def _melody_abc(bars: list[list[_Slot]], chords: list[str], key: str) -> str:
    """Render bars of slots to a Copyist melody ABC string."""
    out_bars = []
    last_chord = None
    for bi, bar in enumerate(bars):
        bar_state: dict[str, int] = {}
        tokens = []
        chord = chords[bi]
        placed_chord = False
        for s in bar:
            prefix = ""
            if not placed_chord and chord != last_chord:
                prefix = f'"{chord}"'
                last_chord = chord
                placed_chord = True
            elif not placed_chord:
                placed_chord = True
            if s.is_rest:
                tokens.append(prefix + "z" + _length_text(s.length))
            else:
                tokens.append(prefix + _note_abc(key, s.midi, s.length, s.tie_to_next, bar_state))
        out_bars.append(" ".join(tokens))
    # Four bars per line for readability.
    lines = []
    for i in range(0, len(out_bars), 4):
        chunk = out_bars[i:i + 4]
        terminator = " |]" if i + 4 >= len(out_bars) else " |"
        lines.append(" | ".join(chunk) + terminator)
    return "\n".join(lines)


# ---------------------------------------------------------------- lyrics

def _load_pool() -> list[list[str]]:
    """Each pool file as a list of words (comment lines dropped, lowercased, apostrophes kept)."""
    pool_dir = Path(__file__).resolve().parent / "lyrics"
    poems = []
    for path in sorted(pool_dir.glob("*.txt")):
        words = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("#") or not line.strip():
                continue
            for w in line.split():
                clean = "".join(ch for ch in w.lower() if ch.isalpha() or ch == "'")
                if clean:
                    words.append(clean)
        if words:
            poems.append(words)
    return poems


_POOL: list[list[str]] | None = None


def _pool() -> list[list[str]]:
    global _POOL
    if _POOL is None:
        _POOL = _load_pool()
    return _POOL


def _is_singable(word: str) -> bool:
    """True if Copyist's own pronunciation path resolves ``word`` (so SoulX never fails on it).

    Uses the SAME lookup ``copyist sing`` uses (``copyist.sing.phones_for``: CMU dict plus Copyist's
    pronunciation table), so a word that passes here is guaranteed singable end to end. Falls back to the
    ``pronouncing`` dictionary if ``copyist.sing`` is unavailable for any reason.
    """
    try:
        from copyist.sing import phones_for

        return phones_for(word) is not None
    except Exception:  # pragma: no cover - copyist.sing is always importable here
        import pronouncing

        return bool(pronouncing.phones_for_word(word))


_SINGABLE_POOL: list[list[str]] | None = None


def _singable_pool() -> list[list[str]]:
    """The lyric pool filtered to WHOLE words that Copyist can pronounce (cached, deterministic order).

    Only dictionary-safe words survive, so the generated ``w:`` line never produces an out-of-dictionary
    token for ``copyist sing``. A poem with no singable words is dropped; if nothing survives (never, with the
    public-domain pool) a tiny all-known fallback keeps generation working.
    """
    global _SINGABLE_POOL
    if _SINGABLE_POOL is None:
        filtered = []
        for poem in _pool():
            words = [w for w in poem if _is_singable(w)]
            if words:
                filtered.append(words)
        _SINGABLE_POOL = filtered or [["la", "love", "day", "night", "sing", "heart"]]
    return _SINGABLE_POOL


def _syllabify(word: str) -> list[str]:
    """Split a known word into syllable pieces for the melody's notes.

    The pieces are rejoined by ``copyist sing`` (hyphen-connected syllables of one word), so the ORIGINAL
    word is what SoulX pronounces — the pieces themselves need not be dictionary words. Uses the CMU syllable
    count when known, else a vowel-group heuristic.
    """
    import pronouncing
    phones = pronouncing.phones_for_word(word)
    if phones:
        n = max(1, len(pronouncing.stresses(phones[0])))
    else:
        groups = 0
        prev_vowel = False
        for ch in word:
            v = ch in "aeiouy"
            if v and not prev_vowel:
                groups += 1
            prev_vowel = v
        n = max(1, groups)
    if n == 1 or len(word) <= n:
        return [word]
    # Slice letters into n roughly equal chunks.
    size = len(word) / n
    pieces = []
    start = 0
    for i in range(1, n + 1):
        end = round(i * size)
        pieces.append(word[start:end])
        start = end
    return [p for p in pieces if p] or [word]


def _lyric_line(n_onsets: int, holds: set[int], rng: random.Random) -> str:
    """A `w:`-syntax lyric line whose note_target equals n_onsets.

    One syllable per onset, except ``holds`` (melisma continuations) get a `_` hold instead of a new
    syllable. Syllables of ONE word are hyphen-joined so ``copyist sing`` rejoins them into the original,
    dictionary-safe word; a space starts the next word. Only singable words are used, so SoulX never hits a
    missing pronunciation.
    """
    n_syllables = n_onsets - len(holds)  # holds take notes but carry no new syllable
    poem = rng.choice(_singable_pool())
    one_syllable = [w for w in poem if len(_syllabify(w)) == 1] or ["la"]
    # A flat list of (piece, is_last_of_word), choosing words so the line NEVER ends mid-word: a word is
    # only started if it fits whole in the remaining syllable budget, otherwise a one-syllable filler is used.
    chunks: list[tuple[str, bool]] = []
    wi = rng.randrange(len(poem))
    while len(chunks) < n_syllables:
        remaining = n_syllables - len(chunks)
        word = poem[wi % len(poem)]
        wi += 1
        pieces = _syllabify(word)
        if len(pieces) > remaining:
            # Not enough notes left to finish this word cleanly: fill with one-syllable words instead, so no
            # dangling hyphen (a half-word) is ever emitted. Copyist rejoins only complete words.
            pieces = [one_syllable[wi % len(one_syllable)]]
        for i, piece in enumerate(pieces):
            chunks.append((piece, i == len(pieces) - 1))

    out_tokens: list[str] = []
    syl_cursor = 0
    for onset in range(n_onsets):
        if onset in holds:
            out_tokens.append("_")
            continue
        piece, is_last = chunks[syl_cursor]
        # A non-final syllable of a word ends with a hyphen so Copyist rejoins the word for pronunciation.
        out_tokens.append(piece if is_last else piece + "-")
        syl_cursor += 1
    return " ".join(out_tokens)


# ---------------------------------------------------------------- assembly

def _tempo_for(meter: str, cfg: SamplerConfig, rng: random.Random) -> int:
    lo, hi = cfg.tempo_range
    return rng.randint(lo, hi)


def _render_yaml(key: str, meter: str, tempo: int, voice: str, voice_range: tuple[str, str],
                 melody_abc: str, lyric_line: str, seed: int) -> str:
    low, high = voice_range
    lines = [
        "format: copyist-song/0",
        f"title: Study {seed}",
        f"key: {key}",
        f"meter: {meter}",
        "unit: 1/8",
        f"tempo: {tempo}",
        "voice:",
        f"  name: {voice.capitalize()}",
        f"  range: [{low}, {high}]",
        "# Generated training melody (open_transcriber.data.scores); public-domain lyric pool.",
        "melody: |",
    ]
    for ml in melody_abc.splitlines():
        lines.append(f"  {ml}")
    lines.append("lyrics:")
    lines.append(f"  - {lyric_line}")
    return "\n".join(lines) + "\n"


def sample_song(seed: int, cfg: SamplerConfig = DEFAULT) -> SampledSong:
    """Generate one deterministic copyist-song/0 melody from ``seed``."""
    rng = random.Random(seed)
    key = rng.choice(cfg.keys)
    meter = rng.choice(cfg.meters)
    tempo = _tempo_for(meter, cfg, rng)
    voice = rng.choice(list(cfg.voice_ranges))
    vr = cfg.voice_ranges[voice]
    low = m21pitch.Pitch(vr[0]).midi
    high = m21pitch.Pitch(vr[1]).midi

    meta: dict = {
        "seed": seed, "key": key, "meter": meter, "tempo": tempo, "voice": voice,
        "minor": _is_minor(key),
        "sixteenths": False, "dotted": False, "syncopation": False, "ties": False,
        "pickup": False, "rests": False, "melisma": False, "patter": False,
        "octave_leap": False, "chromatic": False,
    }

    bars, chords = _build_melody(key, meter, cfg, rng, low, high, meta)
    # syncopation flag: an off-beat accent cell was used (detected by a 1,2,... pattern start).
    for bar in bars:
        lengths = [s.length for s in bar]
        if len(lengths) >= 3 and lengths[0] == Fraction(1) and lengths[1] == Fraction(2):
            meta["syncopation"] = True
            break

    holds: set[int] = set()
    if meta.pop("_use_melisma", False):
        run = _choose_melisma(bars, _scale_pcs(key), low, high, cfg, rng)
        if run is not None:
            start, mlen = run
            holds = {start + k for k in range(1, mlen)}
            meta["melisma"] = True
            meta["melisma_len"] = mlen

    onset_count = _syllable_slot_count(bars)
    lyric_line = _lyric_line(onset_count, holds, rng)

    melody_abc = _melody_abc(bars, chords, key)
    yaml_text = _render_yaml(key, meter, tempo, voice, vr, melody_abc, lyric_line, seed)
    meta["onsets"] = onset_count
    return SampledSong(yaml_text=yaml_text, meta=meta)


def validate_song_text(yaml_text: str) -> list[Finding]:
    """Write the YAML to a temp file, load it and run Copyist's validators."""
    path = Path(tempfile.mktemp(suffix=".yaml"))
    path.write_text(yaml_text, encoding="utf-8")
    try:
        return validate(load_song(path))
    finally:
        path.unlink(missing_ok=True)


def _passes(yaml_text: str, heldout) -> tuple[bool, str]:
    """True if the YAML validates (no FAIL) and is not a held-out leak."""
    path = Path(tempfile.mktemp(suffix=".yaml"))
    path.write_text(yaml_text, encoding="utf-8")
    try:
        song = load_song(path)
    except Exception as e:  # a malformed melody: treat as a fail, re-roll
        path.unlink(missing_ok=True)
        return False, f"load error: {e}"
    try:
        if not is_finished(validate(song)):
            return False, "validator FAIL"
        if heldout is not None:
            result = heldout_mod.is_held_out(song, heldout)
            if result.is_leak:
                return False, f"held-out leak: {result.reason}"
        return True, ""
    finally:
        path.unlink(missing_ok=True)


def default_out_dir() -> Path:
    return heldout_mod.COPYIST_ROOT / ".cache" / "t1" / "scores"


def write_batch(out_dir: Path | None = None, n: int = 100, start_seed: int = 0,
                cfg: SamplerConfig = DEFAULT, heldout=None, max_attempts: int = 40) -> list[Path]:
    """Generate ``n`` validated, non-leaking songs into ``out_dir``.

    Seeds run ``start_seed``..``start_seed+n-1``. A song that FAILs validation or
    leaks a held-out tune is re-rolled deterministically with
    ``seed + 10_000_000 * attempt`` up to ``max_attempts``. Files are named
    ``song-<seed>.yaml``. Default out_dir is the gitignored ``.cache/t1/scores``.
    """
    if out_dir is None:
        out_dir = default_out_dir()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if heldout is None:
        heldout = heldout_mod.load_heldout()

    written: list[Path] = []
    for i in range(n):
        base_seed = start_seed + i
        chosen_text = None
        for attempt in range(max_attempts):
            seed = base_seed + 10_000_000 * attempt
            text = sample_song(seed, cfg).yaml_text
            ok, _reason = _passes(text, heldout)
            if ok:
                chosen_text = text
                break
        if chosen_text is None:
            raise RuntimeError(f"Could not generate a valid song for seed {base_seed} "
                               f"in {max_attempts} attempts.")
        path = out_dir / f"song-{base_seed}.yaml"
        path.write_text(chosen_text, encoding="utf-8")
        written.append(path)
    return written
