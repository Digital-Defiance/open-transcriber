"""Frame-level sung-note transcriber: a small recurrent head over FROZEN Basic Pitch posteriors and a
log-mel of the Demucs vocal.

This is the richer model the sung-data experiment calls for (docs/transcriber-plan.md; DECISIONS 22 Oct),
and the deliberate step up from the per-note keep/drop head in :mod:`open_transcriber.data.pilot`, which did
NOT beat Basic Pitch on real sung vocals. The question is whether a model trained on REALISTIC SUNG data
(SoulX sings a sampler score, mixed over the band, then Demucs-separated — the clean-licence stand-in for
real singing) moves the real-dev melody number toward the GAME specialist bar.

The design (small enough to train on an M4 Max):

* **Input per frame** (on Basic Pitch's 86 fps / 88-bin note grid): the frozen BP ``note`` posterior (88) and
  ``onset`` posterior (88), plus a log-mel of the Demucs vocal resampled onto the same frame grid
  (``MEL_BANDS`` bands). Nothing in the input is learned — Basic Pitch stays frozen, as the plan requires.
* **Model** (:class:`FrameModel`): input projection → a 1-D temporal context (bidirectional GRU) → two
  per-frame heads, a note-activation head (88) and an onset head (88). A handful of hundred-thousand
  parameters; trains on CPU for the smoke test and on the GPU for the pilot.
* **Labels**: the exact melody MIDI from the sampler score (known by construction), rasterised onto the same
  frame grid — a frame is active between a note's onset and offset, and its first frame is an onset frame.
* **Decoding** (:func:`decode_notes`): Basic-Pitch-style note creation — start a note where the onset head
  peaks above ``onset_thresh`` and the note head is active, extend while the note head stays above
  ``frame_thresh``. One monophonic melody line is taken (the argmax-active pitch per frame region), matching
  the ``mono_bp`` baseline's job.

The honest success test is the SAME real recordings (RWC-other-dev, karaoke-dev) with the strict scorer and a
fair cross-fit onset shift (:mod:`open_transcriber.data.pilot` already does this for the baseline). Judging is
NEVER on our own tidy renders.

CPU-only and deterministic (a fixed seed is recorded). The real feature extraction (Basic Pitch posteriors,
Demucs, SoulX singing) runs in granted GPU windows; the plumbing here is proven end to end on CPU with small
or synthetic inputs first.

Content was rephrased for compliance with licensing restrictions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from open_transcriber.bakeoff import notes_io

#: Basic Pitch's note grid: 88 semitones from MIDI 21 (A0), at this frame rate.
FPS = 86.0
HOP = 1.0 / FPS
N_BINS = 88
MIDI_LO = 21  # A0, Basic Pitch's lowest note bin
#: Log-mel bands of the Demucs vocal added beside the BP posteriors.
MEL_BANDS = 32
#: Feature width per frame: BP note + BP onset + log-mel.
FEATURE_DIM = N_BINS + N_BINS + MEL_BANDS
#: The singing range the melody line is kept within (same as the board's ``mono_bp``).
MELODY_LO = 45
MELODY_HI = 84


# --- features and labels --------------------------------------------------------------------------------

def frames_from_posteriors(note: np.ndarray, onset: np.ndarray, mel: np.ndarray | None) -> np.ndarray:
    """Stack the frozen BP ``note`` (T, 88) and ``onset`` (T, 88) posteriors with a log-mel (T, MEL_BANDS).

    All three are trimmed/padded to a common frame count. ``mel`` may be ``None`` (zeros), so the plumbing
    runs before a vocal stem is available.
    """
    note = np.asarray(note, dtype=np.float32)
    onset = np.asarray(onset, dtype=np.float32)
    t = min(len(note), len(onset))
    if mel is None:
        mel = np.zeros((t, MEL_BANDS), dtype=np.float32)
    else:
        mel = np.asarray(mel, dtype=np.float32)
        t = min(t, len(mel))
    note, onset, mel = note[:t], onset[:t], mel[:t]
    if note.shape[1] != N_BINS or onset.shape[1] != N_BINS:
        raise ValueError(f"BP posteriors must have {N_BINS} note bins, got {note.shape}, {onset.shape}")
    if mel.shape[1] != MEL_BANDS:
        raise ValueError(f"log-mel must have {MEL_BANDS} bands, got {mel.shape}")
    return np.concatenate([note, onset, mel], axis=1).astype(np.float32)


def log_mel(wav: np.ndarray, sr: int, n_frames: int, n_mels: int = MEL_BANDS,
            fmin: float = 50.0, fmax: float = 8000.0) -> np.ndarray:
    """A simple log-mel of a mono waveform, resampled to ``n_frames`` on the BP grid. Pure numpy (no librosa).

    Short-time power spectrum on a hop that matches the BP frame rate, a triangular mel filterbank, log, then
    linear interpolation onto exactly ``n_frames`` so it lines up frame-for-frame with the BP posteriors.
    """
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    if wav.size == 0:
        return np.zeros((n_frames, n_mels), dtype=np.float32)
    peak = float(np.max(np.abs(wav)))
    if peak > 1.0:  # a wav read at integer scale; bring it to [-1, 1] so the power spectrum stays in range
        wav = wav / peak
    win = 1024
    hop = max(1, int(round(sr * HOP)))
    pad = win // 2
    wav = np.pad(wav, (pad, pad), mode="reflect" if wav.size > pad else "constant")
    n = 1 + (len(wav) - win) // hop if len(wav) >= win else 1
    window = np.hanning(win).astype(np.float32)
    framed = np.zeros((n, win), dtype=np.float32)
    for i in range(n):
        seg = wav[i * hop: i * hop + win]
        framed[i, : len(seg)] = seg
    spec = np.abs(np.fft.rfft(framed.astype(np.float64) * window, axis=1)) ** 2  # (n, win/2+1), float64
    spec = np.nan_to_num(spec, nan=0.0, posinf=np.finfo(np.float64).max, neginf=0.0)
    fb = _mel_filterbank(sr, win, n_mels, fmin, fmax).astype(np.float64)          # (n_mels, win/2+1)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        energy = np.nan_to_num(spec @ fb.T, nan=0.0, posinf=np.finfo(np.float64).max, neginf=0.0)
    mel = np.log(np.maximum(energy, 1e-8)).astype(np.float32)  # (n, n_mels)
    # Resample along time to exactly n_frames.
    if n == n_frames:
        out = mel
    else:
        src = np.linspace(0.0, 1.0, num=n)
        dst = np.linspace(0.0, 1.0, num=n_frames)
        out = np.stack([np.interp(dst, src, mel[:, b]) for b in range(n_mels)], axis=1).astype(np.float32)
    # Per-band standardisation keeps the scale comparable with the [0,1] BP posteriors.
    mu, sd = out.mean(0, keepdims=True), out.std(0, keepdims=True) + 1e-5
    return ((out - mu) / sd).astype(np.float32)


def _mel_filterbank(sr: int, n_fft: int, n_mels: int, fmin: float, fmax: float) -> np.ndarray:
    def hz_to_mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel_to_hz(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    n_freqs = n_fft // 2 + 1
    freqs = np.linspace(0, sr / 2, n_freqs)
    mels = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    pts = mel_to_hz(mels)
    fb = np.zeros((n_mels, n_freqs), dtype=np.float32)
    for m in range(1, n_mels + 1):
        lo, ctr, hi = pts[m - 1], pts[m], pts[m + 1]
        left = (freqs - lo) / max(ctr - lo, 1e-6)
        right = (hi - freqs) / max(hi - ctr, 1e-6)
        fb[m - 1] = np.clip(np.minimum(left, right), 0.0, None)
    return fb


def align_truth_to_audio(truth: notes_io.Notes, bp_note: np.ndarray, thresh: float = 0.5
                         ) -> notes_io.Notes:
    """Map the exact score-time MIDI onto the sung-audio timeline using the BP note-posterior voiced span.

    The labels are exact in PITCH by construction, but their TIME is score time: SoulX does not sing at the
    written tempo (it stretches, pauses and has a lead-in latency), so a note at score second ``t`` is not at
    audio second ``t``. Training on score-time labels teaches the model mislabeled frames (diagnosed: BP
    passthrough scores ~1.6% against raw labels but ~13% against time-aligned labels).

    The fix used here is a per-clip linear rescale of label time [0, label_dur] onto the BP-voiced span
    [t0, t1] (where BP's note posterior first/last exceeds ``thresh``). It is deterministic, needs no extra
    model, and recovers most of the lost signal. Pitches are untouched.
    """
    strong = np.where(bp_note.max(axis=1) >= thresh)[0]
    if len(strong) < 2 or truth.offset.size == 0:
        return truth
    t0, t1 = float(strong.min()) / FPS, float(strong.max()) / FPS
    ld = float(truth.offset.max())
    if ld <= 1e-6 or t1 <= t0:
        return truth
    scale = (t1 - t0) / ld

    def m(x: float) -> float:
        return t0 + x * scale

    onset = np.array([m(float(o)) for o in truth.onset])
    offset = np.array([max(m(float(f)), m(float(o)) + notes_io.MIN_DURATION)
                       for o, f in zip(truth.onset, truth.offset)])
    return notes_io._validate(notes_io.Notes(onset, offset, truth.midi.copy()), False, "aligned")


def rasterise_labels(truth: notes_io.Notes, n_frames: int) -> tuple[np.ndarray, np.ndarray]:
    """Exact melody MIDI → per-frame (note_active, onset) label arrays, both (n_frames, 88).

    A note is active in the bins covering [onset, offset); its first frame is an onset frame. Pitches outside
    the 88-bin grid are clipped in.
    """
    note = np.zeros((n_frames, N_BINS), dtype=np.float32)
    onset = np.zeros((n_frames, N_BINS), dtype=np.float32)
    for o, f, m in zip(truth.onset, truth.offset, truth.midi):
        b = int(round(m)) - MIDI_LO
        if not 0 <= b < N_BINS:
            b = min(max(b, 0), N_BINS - 1)
        fo = int(np.floor(o * FPS))
        ff = max(fo + 1, int(np.ceil(f * FPS)))
        fo = min(max(fo, 0), n_frames - 1)
        ff = min(max(ff, fo + 1), n_frames)
        note[fo:ff, b] = 1.0
        onset[fo, b] = 1.0
    return note, onset


# --- decoding -------------------------------------------------------------------------------------------

def decode_notes(note: np.ndarray, onset: np.ndarray, frame_thresh: float = 0.3,
                 onset_thresh: float = 0.5, min_frames: int = 3, onset_shift: float = 0.0,
                 lo: int = MELODY_LO, hi: int = MELODY_HI) -> notes_io.Notes:
    """Basic-Pitch-style note creation from per-frame note/onset activations, taking one monophonic line.

    For each frame, the candidate pitch is the argmax active note bin within [lo, hi]. A note starts where the
    onset head peaks above ``onset_thresh`` (or the active pitch changes) and ends when the note head drops
    below ``frame_thresh`` for that pitch. Notes shorter than ``min_frames`` are dropped.
    """
    note = np.asarray(note, dtype=np.float32)
    onset = np.asarray(onset, dtype=np.float32)
    t = min(len(note), len(onset))
    if t == 0:
        return notes_io._validate(notes_io.Notes(np.zeros(0), np.zeros(0), np.zeros(0)), False, "frames")
    lo_b, hi_b = lo - MIDI_LO, hi - MIDI_LO
    rows: list[tuple[float, float, float]] = []
    cur_pitch = -1
    cur_start = 0
    for i in range(t):
        active = note[i, lo_b:hi_b + 1]
        best = int(np.argmax(active))
        best_val = float(active[best])
        pitch = best + lo_b if best_val >= frame_thresh else -1
        # A re-onset only counts where the onset head is a local peak above threshold, so a held note's
        # wobbling onset posterior does not shatter it into many fragments.
        onset_hit = False
        if pitch >= 0 and float(onset[i, pitch]) >= onset_thresh:
            prev_o = float(onset[i - 1, pitch]) if i > 0 else 0.0
            onset_hit = float(onset[i, pitch]) >= prev_o
        changed = pitch != cur_pitch
        if cur_pitch >= 0 and (pitch < 0 or changed or (onset_hit and i - cur_start >= min_frames)):
            if i - cur_start >= min_frames:
                rows.append((cur_start * HOP, i * HOP, float(cur_pitch + MIDI_LO)))
            cur_pitch = -1
        if pitch >= 0 and cur_pitch < 0:
            cur_pitch, cur_start = pitch, i
    if cur_pitch >= 0 and t - cur_start >= min_frames:
        rows.append((cur_start * HOP, t * HOP, float(cur_pitch + MIDI_LO)))
    # Merge adjacent same-pitch notes separated by a tiny gap (decode seams, not true re-articulations).
    merged: list[tuple[float, float, float]] = []
    for o, f, p in sorted(rows):
        if merged and merged[-1][2] == p and o - merged[-1][1] < 0.05:
            merged[-1] = (merged[-1][0], max(merged[-1][1], f), p)
        else:
            merged.append((o, f, p))
    rows = merged
    if onset_shift:
        rows = [(max(o + onset_shift, 0.0), max(f + onset_shift, max(o + onset_shift, 0.0) + notes_io.MIN_DURATION), p)
                for o, f, p in rows]
    rows.sort()
    if not rows:
        return notes_io._validate(notes_io.Notes(np.zeros(0), np.zeros(0), np.zeros(0)), False, "frames")
    return notes_io._validate(
        notes_io.Notes(np.array([r[0] for r in rows]), np.array([r[1] for r in rows]),
                       np.array([r[2] for r in rows])), False, "frames")


# --- the model ------------------------------------------------------------------------------------------

class FrameModel:
    """A small bidirectional-GRU frame model over the frozen features, with note and onset heads.

    Deliberately modest (a pilot, sized for the M4 Max): an input projection, one BiGRU layer, two linear
    heads. Trains on sequence windows; predicts the whole sequence at once.
    """

    def __init__(self, hidden: int = 64, seed: int = 0):
        import torch

        self.seed = seed
        self.hidden = hidden
        torch.manual_seed(seed)
        self.proj = torch.nn.Sequential(torch.nn.Linear(FEATURE_DIM, hidden), torch.nn.ReLU())
        self.gru = torch.nn.GRU(hidden, hidden, batch_first=True, bidirectional=True)
        self.note_head = torch.nn.Linear(2 * hidden, N_BINS)
        self.onset_head = torch.nn.Linear(2 * hidden, N_BINS)
        self._modules = [self.proj, self.gru, self.note_head, self.onset_head]

    def _params(self):
        for m in self._modules:
            yield from m.parameters()

    def _forward(self, x):
        import torch  # noqa: F401

        h = self.proj(x)
        h, _ = self.gru(h)
        return self.note_head(h), self.onset_head(h)

    def fit(self, seqs: list[tuple[np.ndarray, np.ndarray, np.ndarray]], epochs: int = 20,
            lr: float = 1e-3, window: int = 512, batch: int = 8) -> dict:
        """Train on a list of (features (T,F), note_label (T,88), onset_label (T,88)) sequences.

        Sequences are chopped into fixed windows so every batch is a tensor. Onsets are rare, so the onset
        loss is positively weighted.
        """
        import torch

        if not seqs:
            return {"epochs": 0, "n": 0, "loss": None}
        torch.manual_seed(self.seed)
        wins: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for x, yn, yo in seqs:
            t = min(len(x), len(yn), len(yo))
            x, yn, yo = x[:t], yn[:t], yo[:t]
            for s in range(0, max(1, t), window):
                xs, yns, yos = x[s:s + window], yn[s:s + window], yo[s:s + window]
                if len(xs) < 1:
                    continue
                pad = window - len(xs)
                if pad:
                    xs = np.pad(xs, ((0, pad), (0, 0)))
                    yns = np.pad(yns, ((0, pad), (0, 0)))
                    yos = np.pad(yos, ((0, pad), (0, 0)))
                wins.append((xs, yns, yos))
        if not wins:
            return {"epochs": 0, "n": 0, "loss": None}
        X = torch.from_numpy(np.stack([w[0] for w in wins]).astype(np.float32))
        YN = torch.from_numpy(np.stack([w[1] for w in wins]).astype(np.float32))
        YO = torch.from_numpy(np.stack([w[2] for w in wins]).astype(np.float32))
        onset_pos = float(YO.sum())
        onset_neg = float(YO.numel() - onset_pos)
        ow = torch.tensor([onset_neg / onset_pos]) if onset_pos > 0 else torch.tensor([1.0])
        note_pos = float(YN.sum())
        note_neg = float(YN.numel() - note_pos)
        # Only ~1 of 88 bins is active per active frame, so without a positive weight the note head
        # collapses to all-zero (the onset head already gets one). Cap it so a loud window can't dominate.
        nw = torch.tensor([min(note_neg / note_pos, 50.0)]) if note_pos > 0 else torch.tensor([1.0])
        note_loss = torch.nn.BCEWithLogitsLoss(pos_weight=nw)
        onset_loss = torch.nn.BCEWithLogitsLoss(pos_weight=ow)
        opt = torch.optim.Adam(self._params(), lr=lr)
        n = len(wins)
        last = None
        for _ in range(epochs):
            perm = torch.randperm(n)
            total = 0.0
            for b in range(0, n, batch):
                idx = perm[b:b + batch]
                opt.zero_grad()
                pn, po = self._forward(X[idx])
                loss = note_loss(pn, YN[idx]) + onset_loss(po, YO[idx])
                loss.backward()
                opt.step()
                total += float(loss.item()) * len(idx)
            last = total / n
        return {"epochs": epochs, "n": n, "windows": n, "loss": last}

    def predict(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Per-frame note and onset probabilities for one sequence (T, F)."""
        import torch

        if len(features) == 0:
            return np.zeros((0, N_BINS), np.float32), np.zeros((0, N_BINS), np.float32)
        with torch.no_grad():
            x = torch.from_numpy(features.astype(np.float32)).unsqueeze(0)
            pn, po = self._forward(x)
            return torch.sigmoid(pn)[0].numpy(), torch.sigmoid(po)[0].numpy()

    def save(self, path: Path) -> None:
        import torch

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"seed": self.seed, "hidden": self.hidden,
                    "proj": self.proj.state_dict(), "gru": self.gru.state_dict(),
                    "note_head": self.note_head.state_dict(), "onset_head": self.onset_head.state_dict()}, path)

    @classmethod
    def load(cls, path: Path) -> "FrameModel":
        import torch

        blob = torch.load(path, map_location="cpu", weights_only=False)
        model = cls(hidden=blob.get("hidden", 64), seed=blob.get("seed", 0))
        model.proj.load_state_dict(blob["proj"])
        model.gru.load_state_dict(blob["gru"])
        model.note_head.load_state_dict(blob["note_head"])
        model.onset_head.load_state_dict(blob["onset_head"])
        return model


@dataclass
class FrameDataset:
    """A frame-level training set: a list of per-clip (features, note_label, onset_label) and provenance."""

    seqs: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = field(default_factory=list)
    songs: list[str] = field(default_factory=list)
    seed: int = 0

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {}
        for i, (x, yn, yo) in enumerate(self.seqs):
            arrays[f"x{i}"] = x
            arrays[f"yn{i}"] = yn
            arrays[f"yo{i}"] = yo
        np.savez(path, n=len(self.seqs), songs=np.array(self.songs, dtype=object), seed=self.seed, **arrays)

    @classmethod
    def load(cls, path: Path) -> "FrameDataset":
        blob = np.load(path, allow_pickle=True)
        n = int(blob["n"])
        seqs = [(blob[f"x{i}"], blob[f"yn{i}"], blob[f"yo{i}"]) for i in range(n)]
        return cls(seqs=seqs, songs=list(blob["songs"]), seed=int(blob["seed"]))
