"""The fixed GO/NO-GO decision for the encoder bake-off (pure, tested; no model execution).

Applies the pre-registered rule exactly (``PREREGISTRATION.md``): a GO needs one music encoder that, in at
least two of the three seeds, beats the Basic-Pitch-only control by >=3.0 absolute on real-dev octave-exact
melody F1 and by >=5.0 on synthetic, without losing more than 2.0 points mean on beat/downbeat/chord/key
versus the best candidate/control, and fits the Mac. Ties break lexicographically. Whisper is added only if
it adds >=2.0 on the repeated-pitch/lyric-onset subset and loses <=0.5 overall. Otherwise NO-GO.
"""

from __future__ import annotations

from dataclasses import dataclass, field

REAL_MELODY_THRESHOLD = 3.0
SYNTHETIC_MELODY_THRESHOLD = 5.0
MULTITASK_LOSS_LIMIT = 2.0
TIE_MELODY = 1.0
TIE_MULTITASK = 0.5
WHISPER_MIN_GAIN = 2.0
WHISPER_MAX_LOSS = 0.5
SEEDS_REQUIRED = 2

#: The secondary tasks whose mean must not drop more than MULTITASK_LOSS_LIMIT vs the best.
MULTITASK_KEYS = ("beat_f1", "downbeat_f1", "chord", "key")
#: The primary real-dev melody metric used for the +3.0 gate and the tie-break.
REAL_MELODY_KEY = "oct_on50_f1"


@dataclass
class EncoderResult:
    """One encoder's results across seeds. Scores are 0..100 (points)."""
    encoder: str
    real_melody: dict[int, float]        # seed -> real-dev octave-exact melody F1 (points)
    synthetic_melody: dict[int, float]   # seed -> synthetic melody F1 (points)
    multitask_mean: dict[int, float]     # seed -> mean of MULTITASK_KEYS (points)
    fits_mac: bool = True
    size_rank: float = 0.0               # smaller/faster = smaller number (tie-break)

    def seeds(self) -> list[int]:
        return sorted(self.real_melody)

    def mean_real_melody(self) -> float:
        vals = list(self.real_melody.values())
        return sum(vals) / len(vals) if vals else 0.0

    def mean_multitask(self) -> float:
        vals = list(self.multitask_mean.values())
        return sum(vals) / len(vals) if vals else 0.0


@dataclass
class Decision:
    go: bool
    winner: str | None = None
    add_whisper: bool = False
    reasons: list[str] = field(default_factory=list)
    qualifying: list[str] = field(default_factory=list)


def _seeds_passing(candidate: EncoderResult, control: EncoderResult,
                   best_multitask_by_seed: dict[int, float]) -> int:
    """How many seeds meet all per-seed gates for this candidate."""
    passing = 0
    for seed in candidate.seeds():
        real_gain = candidate.real_melody.get(seed, 0.0) - control.real_melody.get(seed, 0.0)
        synth_gain = candidate.synthetic_melody.get(seed, 0.0) - control.synthetic_melody.get(seed, 0.0)
        multitask_loss = best_multitask_by_seed.get(seed, 0.0) - candidate.multitask_mean.get(seed, 0.0)
        if (real_gain >= REAL_MELODY_THRESHOLD and synth_gain >= SYNTHETIC_MELODY_THRESHOLD
                and multitask_loss <= MULTITASK_LOSS_LIMIT):
            passing += 1
    return passing


def _best_multitask_by_seed(results: list[EncoderResult], control: EncoderResult) -> dict[int, float]:
    best: dict[int, float] = {}
    for r in list(results) + [control]:
        for seed, v in r.multitask_mean.items():
            best[seed] = max(best.get(seed, float("-inf")), v)
    return best


def decide(candidates: list[EncoderResult], control: EncoderResult,
           whisper_subset_gain: float | None = None, whisper_overall_loss: float | None = None) -> Decision:
    """Apply the pre-registered GO/NO-GO rule. ``candidates`` are the music encoders; ``control`` is the
    Basic-Pitch-only probe. Whisper gains/losses, if given, drive the add-on rule."""
    best_mt = _best_multitask_by_seed(candidates, control)
    qualifying = []
    reasons = []
    for cand in candidates:
        n_pass = _seeds_passing(cand, control, best_mt)
        if not cand.fits_mac:
            reasons.append(f"{cand.encoder}: excluded (does not fit the Mac limit)")
            continue
        if n_pass >= SEEDS_REQUIRED:
            qualifying.append(cand)
            reasons.append(f"{cand.encoder}: passes in {n_pass}/{len(cand.seeds())} seeds")
        else:
            reasons.append(f"{cand.encoder}: passes in only {n_pass}/{len(cand.seeds())} seeds")

    if not qualifying:
        return Decision(go=False, winner=None, reasons=reasons, qualifying=[])

    winner = _lexicographic_winner(qualifying)
    add_whisper = False
    if whisper_subset_gain is not None and whisper_overall_loss is not None:
        add_whisper = (whisper_subset_gain >= WHISPER_MIN_GAIN and whisper_overall_loss <= WHISPER_MAX_LOSS)
        reasons.append(f"whisper: gain {whisper_subset_gain:+.2f} / loss {whisper_overall_loss:+.2f} -> "
                       f"{'add' if add_whisper else 'skip'}")
    return Decision(go=True, winner=winner.encoder, add_whisper=add_whisper, reasons=reasons,
                    qualifying=[c.encoder for c in qualifying])


def _lexicographic_winner(qualifying: list[EncoderResult]) -> EncoderResult:
    """Highest real-dev melody; within 1 pt, highest multi-task mean; within 0.5 pt, smaller/faster."""
    best = qualifying[0]
    for cand in qualifying[1:]:
        if cand.mean_real_melody() > best.mean_real_melody() + TIE_MELODY:
            best = cand
        elif abs(cand.mean_real_melody() - best.mean_real_melody()) <= TIE_MELODY:
            if cand.mean_multitask() > best.mean_multitask() + TIE_MULTITASK:
                best = cand
            elif abs(cand.mean_multitask() - best.mean_multitask()) <= TIE_MULTITASK:
                if cand.size_rank < best.size_rank:
                    best = cand
    return best
