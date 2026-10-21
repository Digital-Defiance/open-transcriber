"""Turn cached encoder features + by-construction label JSON into the tensors train/eval consume.

The probe training loop (``train.py``) consumes ``{"layers","bp","targets"}`` dicts of tensors; the evaluator
(``evaluate.py``) consumes ``(head_outputs, labels, item)`` tuples. This module builds both from:

- an ``Extractor``'s per-layer features (``early``/``middle``/``final``, each ``(frames, dim)`` at the
  encoder's native rate), stacked and resampled to the common 50 Hz grid;
- Basic Pitch posteriors (summarised to the model's ``bp_dim`` channels, resampled to 50 Hz);
- per-frame targets built **entirely from the score label JSON** (``labels.py``). Nothing here reads audio.

Real-dev clips are resolved through ``dev_splits`` and any sealed TEST-partition id is refused.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .. import dev_splits
from .model import BP_POSTERIOR_DIM, N_KEY, N_PITCH, TARGET_RATE_HZ

#: The declared layer order (matches ``extractor.LAYER_NAMES`` and ``model.stack_layers``).
LAYER_ORDER = ("early", "middle", "final")
#: Mapping from a chord-quality string to the 3-way head class.
_QUALITY_CLASS = {"maj": 0, "min": 1, "no-chord": 2}


def load_labels(path: Path | str) -> dict:
    """Load a ``labels.py`` JSON record as the plain dict the eval scorer consumes."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def frame_count(duration: float, rate: float = TARGET_RATE_HZ) -> int:
    """Number of 50 Hz frames spanning ``duration`` seconds (at least one frame)."""
    return max(1, int(round(float(duration) * rate)))


def _resample_np(arr: np.ndarray, target_frames: int) -> np.ndarray:
    """Resample a ``(frames, dim)`` array along time to ``target_frames`` with linear interpolation."""
    x = torch.as_tensor(np.asarray(arr, dtype=np.float32)).unsqueeze(0)  # (1, frames, dim)
    y = torch.nn.functional.interpolate(x.transpose(1, 2), size=target_frames, mode="linear",
                                        align_corners=False)
    return y.transpose(1, 2).squeeze(0).numpy().astype(np.float32)


def stack_layers_50hz(features: dict[str, np.ndarray], target_frames: int,
                      order=LAYER_ORDER) -> np.ndarray:
    """Stack an extractor's per-layer arrays, resample each to 50 Hz / ``target_frames`` -> (n_layers, T, dim).

    Resampling to ``target_frames`` (derived from the label duration at 50 Hz) time-aligns every encoder's
    native frame rate onto the common grid, so the encoder's ``native_rate_hz`` is handled by the caller.
    """
    layers = [_resample_np(features[name], target_frames) for name in order]
    return np.stack(layers, axis=0).astype(np.float32)


def summarise_bp(posteriors: dict[str, np.ndarray], target_frames: int,
                 bp_dim: int = BP_POSTERIOR_DIM) -> np.ndarray:
    """Summarise Basic Pitch note/onset/contour posteriors to ``bp_dim`` channels at 50 Hz.

    The model's tested interface uses ``bp_dim`` channels (default 3: a per-frame max activation of
    contour/note/onset). The full posterior stack can be swapped in by widening ``bp_dim`` and this summary.
    """
    channels = []
    for key in ("contour", "note", "onset")[:bp_dim]:
        arr = np.asarray(posteriors[key], dtype=np.float32)
        if arr.ndim == 2:
            chan = arr.max(axis=1, keepdims=True)       # (frames, 1): strongest activation per frame
        else:
            chan = arr.reshape(-1, 1)
        channels.append(chan)
    summary = np.concatenate(channels, axis=1)          # (frames, bp_dim)
    return _resample_np(summary, target_frames)


def _pulse_target(times, target_frames: int, rate: float = TARGET_RATE_HZ) -> np.ndarray:
    """A per-frame 0/1 pulse train: 1 at the frame nearest each event time."""
    out = np.zeros(target_frames, dtype=np.float32)
    for t in times:
        idx = int(round(float(t) * rate))
        if 0 <= idx < target_frames:
            out[idx] = 1.0
    return out


def build_targets(labels: dict, target_frames: int, rate: float = TARGET_RATE_HZ) -> dict[str, torch.Tensor]:
    """Per-frame targets for every head (one unbatched clip), built from the score label JSON only.

    Heads: pitch (0..127 class), silence (0/1), onset/offset (0/1 pulses), beat/downbeat (0/1 pulses),
    chord_root (0..12 class), chord_quality (0..2 class), key (scalar 0..23).
    """
    pitch = np.zeros(target_frames, dtype=np.int64)
    silence = np.ones(target_frames, dtype=np.float32)      # silent unless a note covers the frame
    onset = np.zeros(target_frames, dtype=np.float32)
    offset = np.zeros(target_frames, dtype=np.float32)
    for n in labels.get("notes", []):
        a = int(round(float(n["onset"]) * rate))
        b = int(round(float(n["offset"]) * rate))
        midi = int(n["midi"]) % N_PITCH
        for i in range(max(a, 0), min(b, target_frames)):
            pitch[i] = midi
            silence[i] = 0.0
        if 0 <= a < target_frames:
            onset[a] = 1.0
        last = b - 1
        if 0 <= last < target_frames:
            offset[last] = 1.0

    beat = _pulse_target(labels.get("beats", []), target_frames, rate)
    downbeat = _pulse_target(labels.get("downbeats", []), target_frames, rate)

    chord_root = np.full(target_frames, 12, dtype=np.int64)          # 12 = no-chord
    chord_quality = np.full(target_frames, _QUALITY_CLASS["no-chord"], dtype=np.int64)
    for c in labels.get("chords", []):
        a = int(round(float(c["onset"]) * rate))
        b = int(round(float(c["offset"]) * rate))
        rpc = int(c["root_pc"])
        root_class = rpc if 0 <= rpc < 12 else 12
        q_class = _QUALITY_CLASS.get(c["quality"], _QUALITY_CLASS["no-chord"])
        for i in range(max(a, 0), min(b, target_frames)):
            chord_root[i] = root_class
            chord_quality[i] = q_class

    key_class = int(labels.get("key_class", 0)) % N_KEY
    return {
        "pitch": torch.from_numpy(pitch),
        "silence": torch.from_numpy(silence),
        "onset": torch.from_numpy(onset),
        "offset": torch.from_numpy(offset),
        "beat": torch.from_numpy(beat),
        "downbeat": torch.from_numpy(downbeat),
        "chord_root": torch.from_numpy(chord_root),
        "chord_quality": torch.from_numpy(chord_quality),
        "key": torch.tensor(key_class, dtype=torch.int64),
    }


def _clip_tensors(features: dict[str, np.ndarray], bp_post: dict[str, np.ndarray], labels: dict,
                  bp_dim: int = BP_POSTERIOR_DIM, order=LAYER_ORDER) -> dict:
    """Build the (unbatched) layers/bp/targets tensors for one clip, all frame-aligned at 50 Hz."""
    target_frames = frame_count(labels.get("duration", 0.0))
    layers = stack_layers_50hz(features, target_frames, order=order)                # (n_layers, T, dim)
    bp = summarise_bp(bp_post, target_frames, bp_dim=bp_dim)                          # (T, bp_dim)
    targets = build_targets(labels, target_frames)
    return {
        "layers": torch.from_numpy(layers),
        "bp": torch.from_numpy(bp),
        "targets": targets,
    }


def _stack_batch(clip_tensors: list[dict]) -> dict:
    """Stack a list of unbatched clip tensors (same frame count) into one training batch dict."""
    layers = torch.stack([c["layers"] for c in clip_tensors], dim=0)   # (B, n_layers, T, dim)
    bp = torch.stack([c["bp"] for c in clip_tensors], dim=0)           # (B, T, bp_dim)
    targets = {}
    for head in clip_tensors[0]["targets"]:
        targets[head] = torch.stack([c["targets"][head] for c in clip_tensors], dim=0)
    return {"layers": layers, "bp": bp, "targets": targets}


def build_train_batches(clips: list[dict], extractor, bp, order=LAYER_ORDER) -> list[dict]:
    """Build training batches from clips. Each clip is ``{"wav": path, "labels": dict-or-path}``.

    Clips are grouped by frame count so a batch stacks cleanly (each clip's own frame count is set by its
    label duration). Returns a list of ``{"layers","bp","targets"}`` batch dicts. Labels come from the score.
    """
    bp_dim = getattr(bp, "bp_dim", BP_POSTERIOR_DIM)
    by_frames: dict[int, list[dict]] = {}
    for clip in clips:
        labels = clip["labels"] if isinstance(clip["labels"], dict) else load_labels(clip["labels"])
        feats = extractor.features(Path(clip["wav"]))
        post = bp.posteriors(Path(clip["wav"]))
        tensors = _clip_tensors(feats, post, labels, bp_dim=bp_dim, order=order)
        by_frames.setdefault(tensors["layers"].shape[1], []).append(tensors)
    return [_stack_batch(group) for group in by_frames.values()]


def build_eval_clips(clips: list[dict], trained_model, extractor, bp,
                     order=LAYER_ORDER) -> list[tuple[dict, dict, str]]:
    """Run a trained probe over real-dev clips -> ``[(head_outputs_numpy, labels, item_id)]`` for the scorer.

    Each clip is ``{"id": str, "wav": path, "labels": dict-or-path}``. Any sealed test-partition id is
    refused. Labels come from the score. The model is run per clip (batch size 1) in eval mode.
    """
    ids = [str(c["id"]) for c in clips]
    _guard_real_dev_ids(ids)
    bp_dim = getattr(bp, "bp_dim", BP_POSTERIOR_DIM)
    out_clips: list[tuple[dict, dict, str]] = []
    trained_model.eval()
    for clip in clips:
        labels = clip["labels"] if isinstance(clip["labels"], dict) else load_labels(clip["labels"])
        feats = extractor.features(Path(clip["wav"]))
        post = bp.posteriors(Path(clip["wav"]))
        tensors = _clip_tensors(feats, post, labels, bp_dim=bp_dim, order=order)
        with torch.no_grad():
            out = trained_model(tensors["layers"].unsqueeze(0), tensors["bp"].unsqueeze(0))
        head_np = {k: v.squeeze(0).detach().cpu().numpy() for k, v in out.items()}
        out_clips.append((head_np, labels, str(clip["id"])))
    return out_clips


def _guard_real_dev_ids(ids: list[str]) -> None:
    """Refuse any sealed TEST-partition id (RWC-Pop, RWC-other test, USDX test) or a COTB id."""
    sealed = dev_splits.test_partition_ids()
    leaked = sorted(set(ids) & sealed)
    if leaked:
        raise ValueError(f"refusing to assemble sealed test-partition ids: {leaked}")
    if any(str(i).lower().startswith("cotb") for i in ids):
        raise ValueError("refusing to assemble COTB ids during the bake-off")
