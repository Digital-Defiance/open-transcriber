"""Encoder load-and-extract smoke test (bake-off, GPU window 2026-10-21).

THROWAWAY smoke test, not part of the bake-off package. For one candidate encoder at a time it:
  - loads the model in this (ear) torch env, preferring mps, recording CPU-fallback if mps is unavailable;
  - runs it on a few already-rendered sampler clips;
  - records: loads-on-mps (yes/no/cpu-fallback), output feature shape + frame rate, which declared
    early/middle/final layers are actually available, seconds-per-clip, peak memory, and any MPS op failure.

No weights are downloaded here; each ``load_*`` helper is called by a tiny per-encoder driver that has
already fetched the weights with its own licence-checked path. Results are printed as one JSON line per clip
so the caller can tabulate them. RWC/karaoke/owner data are never touched; only .cache/ear/sampler clips.
"""

from __future__ import annotations

import gc
import json
import time
import traceback
from pathlib import Path

import numpy as np
import torch

SAMPLER = Path("/Volumes/Code/copyist/.cache/ear/sampler/clips")


def pick_clips(n: int = 5) -> list[Path]:
    """Five clips from distinct scores (one variant each) for a bit of variety."""
    seen: dict[str, Path] = {}
    for wav in sorted(SAMPLER.glob("*/audio.wav")):
        score = wav.parent.name.split("-v")[0]
        if score not in seen:
            seen[score] = wav
        if len(seen) >= n:
            break
    return list(seen.values())


def best_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def peak_mem_mb(device: str) -> float:
    if device == "mps":
        try:
            return torch.mps.current_allocated_memory() / 1e6
        except Exception:
            return float("nan")
    return float("nan")


def summarize_feature(x) -> dict:
    """Shape + dtype of a tensor/array, moved to CPU."""
    if isinstance(x, torch.Tensor):
        x = x.detach().to("cpu")
        return {"shape": list(x.shape), "dtype": str(x.dtype)}
    arr = np.asarray(x)
    return {"shape": list(arr.shape), "dtype": str(arr.dtype)}


def run_smoke(name: str, load_fn, extract_fn, clips: list[Path], native_rate_hz: float | None) -> None:
    """load_fn() -> (model, device_used, note); extract_fn(model, device, wav_path) -> dict[layer]->feature."""
    rec = {"encoder": name}
    requested = best_device()
    rec["requested_device"] = requested
    try:
        t0 = time.time()
        model, device_used, note = load_fn(requested)
        rec["load_seconds"] = round(time.time() - t0, 2)
        rec["device_used"] = device_used
        rec["cpu_fallback"] = (requested == "mps" and device_used != "mps")
        rec["load_note"] = note
    except Exception as e:  # pragma: no cover - window diagnostics
        rec["load_error"] = f"{type(e).__name__}: {e}"
        rec["load_traceback"] = traceback.format_exc()[-1500:]
        print(json.dumps(rec))
        return
    print(json.dumps({**rec, "event": "loaded"}))

    for wav in clips:
        clip_rec = {"encoder": name, "clip": wav.parent.name, "device_used": rec["device_used"]}
        try:
            if rec["device_used"] == "mps":
                torch.mps.empty_cache()
            t0 = time.time()
            feats = extract_fn(model, rec["device_used"], wav)
            clip_rec["extract_seconds"] = round(time.time() - t0, 3)
            clip_rec["peak_mem_mb"] = round(peak_mem_mb(rec["device_used"]), 1)
            clip_rec["layers"] = {k: summarize_feature(v) for k, v in feats.items()}
            if native_rate_hz:
                clip_rec["declared_native_rate_hz"] = native_rate_hz
                any_layer = next(iter(feats.values()))
                frames = summarize_feature(any_layer)["shape"][-2] if len(summarize_feature(any_layer)["shape"]) >= 2 else None
                clip_rec["frames"] = frames
        except Exception as e:  # pragma: no cover - window diagnostics
            clip_rec["extract_error"] = f"{type(e).__name__}: {e}"
            clip_rec["extract_traceback"] = traceback.format_exc()[-1500:]
        print(json.dumps(clip_rec))
    del model
    gc.collect()
    if rec["device_used"] == "mps":
        torch.mps.empty_cache()
