"""Data-engine pilot: a small head on frozen Basic Pitch notes, trained on Copyist renders.

The owner's goal is "write music, then train a network to hear it". This is the smallest honest test of that
recipe:

1. Generate a few validated sampler songs (:mod:`open_transcriber.data.scores`).
2. Render each with Copyist's band so the melody line has an exact MIDI label and an isolated stem
   (``copyist build`` → ``out/<slug>/stems/melody.wav`` and ``stems/midi/melody.mid``).
3. Run Basic Pitch on the stem (and the mix) to get candidate notes — the frozen feature the plan builds on.
4. Build a per-candidate training table: features of each Basic Pitch note, and a label of whether it matches
   a true melody note (onset within ``MATCH_ONSET`` s, same pitch class).
5. Train a tiny classifier head (:class:`Head`) that scores each candidate; keep the accepted ones.

**The success test is real recordings, not our tidy renders.** A head trained on clean sampled-instrument
melodies is evaluated on the RWC-other and karaoke dev sets' Basic Pitch outputs and scored with the strict
scorer, against the honest baseline ``basic-pitch one line``. The pilot only earns "it works" if that
real-dev ``pc_on50`` moves up toward the GAME ~46 target — moving the number on renders alone proves nothing
(the clean-render-vs-messy-real gap).

This module is CPU-only and deterministic (a fixed seed is recorded in every output). Training on the full
data set is a separate GPU step; here the plumbing runs end to end on CPU on a tiny pilot.

Content was rephrased for compliance with licensing restrictions.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from open_transcriber.bakeoff import notes_io

#: A Basic Pitch candidate counts as a true-positive label if its onset is within this of a true melody note.
MATCH_ONSET = 0.05
#: The singing range the melody line is kept within (same as the board's ``mono_bp``).
MELODY_LO = 45
MELODY_HI = 84
#: Per-candidate feature names, in order. Kept small and interpretable for a pilot head.
FEATURES = ("pitch_norm", "dur", "amp", "gap_prev", "gap_next", "overlap_prev", "leap_prev", "in_range")


@dataclass
class Candidate:
    """One Basic Pitch note candidate with its features and (when training) its match label."""

    onset: float
    offset: float
    midi: int
    amp: float
    features: np.ndarray
    label: int = -1  # 1 matches a true melody note, 0 does not, -1 unknown (inference)


def load_bp_notes(path: Path) -> list[tuple[float, float, int, float]]:
    """Read a Basic Pitch notes TSV (``onset offset midi [amp]``) into tuples, sorted by onset.

    The band-parts/``bp_notes`` TSV has three columns; the raw ``basic-pitch-raw/<id>/notes.tsv`` has an
    ``amplitude`` column. Both are accepted; a missing amplitude defaults to 1.0.
    """
    rows: list[tuple[float, float, int, float]] = []
    text = Path(path).read_text(encoding="utf-8").splitlines()
    header = None
    for line in text:
        line = line.strip()
        if not line:
            continue
        if line.lower().startswith(("onset", "start")):
            header = line.lower().split("\t")
            continue
        parts = line.split("\t") if "\t" in line else line.split(",")
        if len(parts) < 3:
            continue
        onset, offset, midi = float(parts[0]), float(parts[1]), int(float(parts[2]))
        amp = 1.0
        if header and "amplitude" in header:
            ai = header.index("amplitude")
            if ai < len(parts):
                amp = float(parts[ai])
        elif len(parts) >= 4:
            try:
                amp = float(parts[3])
            except ValueError:
                amp = 1.0
        if offset > onset:
            rows.append((onset, offset, midi, amp))
    rows.sort()
    return rows


def candidate_features(notes: list[tuple[float, float, int, float]]) -> list[Candidate]:
    """Turn raw Basic Pitch notes into feature-carrying candidates (no labels yet)."""
    cands: list[Candidate] = []
    for i, (onset, offset, midi, amp) in enumerate(notes):
        dur = offset - onset
        prev = notes[i - 1] if i > 0 else None
        nxt = notes[i + 1] if i + 1 < len(notes) else None
        gap_prev = onset - prev[1] if prev else 1.0
        gap_next = nxt[0] - offset if nxt else 1.0
        overlap_prev = 1.0 if (prev and onset < prev[1] - 0.02) else 0.0
        leap_prev = abs(midi - prev[2]) if prev else 0.0
        in_range = 1.0 if MELODY_LO <= midi <= MELODY_HI else 0.0
        feats = np.array([
            (midi - 60) / 24.0,          # pitch_norm
            min(dur, 2.0),               # dur (s, capped)
            float(amp),                  # amp
            min(max(gap_prev, -0.5), 1.0),   # gap_prev
            min(max(gap_next, -0.5), 1.0),   # gap_next
            overlap_prev,                # overlap_prev
            min(leap_prev, 24.0) / 24.0, # leap_prev
            in_range,                    # in_range
        ], dtype=np.float32)
        cands.append(Candidate(onset, offset, midi, amp, feats))
    return cands


def label_candidates(cands: list[Candidate], truth: notes_io.Notes) -> list[Candidate]:
    """Label each candidate 1 if it matches a true melody note (±MATCH_ONSET s, same pitch class), else 0.

    Greedy one-to-one: each true note is used once, by the nearest unclaimed candidate in onset.
    """
    true_rows = sorted(zip(truth.onset.tolist(), (truth.midi.astype(int) % 12).tolist()))
    used = [False] * len(cands)
    for t_onset, t_pc in true_rows:
        best, best_d = -1, MATCH_ONSET + 1e-9
        for i, c in enumerate(cands):
            if used[i] or (c.midi % 12) != t_pc:
                continue
            d = abs(c.onset - t_onset)
            if d <= best_d:
                best, best_d = i, d
        if best >= 0:
            used[best] = True
    for i, c in enumerate(cands):
        c.label = 1 if used[i] else 0
    return cands


def feature_matrix(cands: list[Candidate]) -> np.ndarray:
    return np.stack([c.features for c in cands]) if cands else np.zeros((0, len(FEATURES)), np.float32)


# --- the head -------------------------------------------------------------------------------------------

class Head:
    """A tiny torch MLP over per-candidate features that scores each Basic Pitch note to keep or drop.

    Deliberately small (a pilot, not the final 5-10M model): it proves the train/eval plumbing and whether a
    learned keep-decision on frozen Basic Pitch moves the real-dev number. Trains on CPU in seconds.
    """

    def __init__(self, hidden: int = 16, seed: int = 0):
        import torch

        self.seed = seed
        torch.manual_seed(seed)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(len(FEATURES), hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, 1),
        )

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 200, lr: float = 0.05) -> dict:
        import torch

        if len(X) == 0:
            return {"epochs": 0, "n": 0, "loss": None}
        torch.manual_seed(self.seed)
        xt = torch.from_numpy(X.astype(np.float32))
        yt = torch.from_numpy(y.astype(np.float32)).unsqueeze(1)
        pos = float(y.sum())
        neg = float(len(y) - pos)
        pos_weight = torch.tensor([neg / pos]) if pos > 0 else torch.tensor([1.0])
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        last = None
        for _ in range(epochs):
            opt.zero_grad()
            out = self.net(xt)
            loss = loss_fn(out, yt)
            loss.backward()
            opt.step()
            last = float(loss.item())
        return {"epochs": epochs, "n": int(len(X)), "loss": last}

    def scores(self, X: np.ndarray) -> np.ndarray:
        import torch

        if len(X) == 0:
            return np.zeros((0,), np.float32)
        with torch.no_grad():
            logits = self.net(torch.from_numpy(X.astype(np.float32)))
            return torch.sigmoid(logits).squeeze(1).numpy()

    def save(self, path: Path) -> None:
        import torch

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state": self.net.state_dict(), "seed": self.seed, "features": list(FEATURES)}, path)

    @classmethod
    def load(cls, path: Path) -> "Head":
        import torch

        blob = torch.load(path, map_location="cpu", weights_only=False)
        head = cls(seed=blob.get("seed", 0))
        head.net.load_state_dict(blob["state"])
        return head


def apply_head(notes: list[tuple[float, float, int, float]], head: Head, threshold: float = 0.5,
               onset_shift: float = 0.0) -> notes_io.Notes:
    """Keep the Basic Pitch notes the head accepts (score ≥ threshold), apply a constant onset shift."""
    cands = candidate_features(notes)
    if not cands:
        return notes_io._validate(notes_io.Notes(np.zeros(0), np.zeros(0), np.zeros(0)), False, "head")
    keep = head.scores(feature_matrix(cands)) >= threshold
    rows = [(c.onset + onset_shift, c.offset + onset_shift, float(c.midi))
            for c, k in zip(cands, keep) if k]
    rows.sort()
    if not rows:
        return notes_io._validate(notes_io.Notes(np.zeros(0), np.zeros(0), np.zeros(0)), False, "head")
    return notes_io._validate(
        notes_io.Notes(np.array([r[0] for r in rows]), np.array([r[1] for r in rows]),
                       np.array([r[2] for r in rows])), False, "head")


@dataclass
class Dataset:
    """The pilot training table: stacked candidate features and match labels, with provenance."""

    X: np.ndarray
    y: np.ndarray
    songs: list[str] = field(default_factory=list)
    seed: int = 0

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, X=self.X, y=self.y, songs=np.array(self.songs), seed=self.seed)

    @classmethod
    def load(cls, path: Path) -> "Dataset":
        blob = np.load(path, allow_pickle=True)
        return cls(X=blob["X"], y=blob["y"], songs=list(blob["songs"]), seed=int(blob["seed"]))
