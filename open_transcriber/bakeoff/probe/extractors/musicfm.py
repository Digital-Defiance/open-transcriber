"""MusicFM FMA-checkpoint extractor (MIT) — the reference implementation.

Per the 21 October smoke test: load the FMA checkpoint (MIT, Creative-Commons-trained) into
``.cache/ear/encoders/musicfm/``, run on ``mps`` (CPU fallback), resample audio to 24 kHz mono, and read
``get_latent(wav, layer_ix)`` for hidden layers 3/7/12 -> early/middle/final, each ``(frames, 1024)`` at
25 Hz, time-aligned across the three layers (all equal length, tracking the clip length).

Heavy imports (torch, torchaudio, soundfile, the MusicFM package) are lazy so this module stays import-cheap.
The MusicFM package uses absolute ``musicfm.*`` imports, so its clone's parent goes on ``sys.path``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from ... import paths
from ..extractor import ExtractorDescriptor

#: The FMA 25 Hz checkpoint's hidden-state indices for early/middle/final (0..12 available).
LAYER_IX = {"early": 3, "middle": 7, "final": 12}
SR = 24000
FEATURE_DIM = 1024
NATIVE_RATE_HZ = 25.0


def encoders_root() -> Path:
    return paths.copyist_root() / ".cache" / "ear" / "encoders"


def musicfm_data_dir() -> Path:
    return encoders_root() / "musicfm" / "data"


def _model_hash(ckpt: Path) -> str:
    """A short, stable id for the checkpoint so the feature cache key tracks the weights."""
    if not ckpt.exists():
        return ""
    stat = ckpt.stat()
    return f"{stat.st_size}-{int(stat.st_mtime)}"


class MusicFMExtractor:
    """MusicFM FMA checkpoint, hidden layers 3/7/12 at 25 Hz, dim 1024, on mps (CPU fallback)."""

    def __init__(self, device: str | None = None):
        self._requested = device or "mps"
        self._model = None
        self._device = None
        data = musicfm_data_dir()
        ckpt = data / "pretrained_fma.pt"
        self._desc = ExtractorDescriptor(
            name="musicfm", code_licence="MIT", weight_licence="MIT",
            native_rate_hz=NATIVE_RATE_HZ, feature_dim=FEATURE_DIM,
            device=self._requested, model_hash=_model_hash(ckpt),
            note="FMA checkpoint; hidden 3/7/12 time-aligned at 25 Hz",
        )
        if not self._desc.free_and_allowed:
            raise PermissionError("musicfm licence is not free/allowed; refusing to load")

    def descriptor(self) -> ExtractorDescriptor:
        return self._desc

    def _load(self):
        if self._model is not None:
            return
        repo = encoders_root()
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))  # so `import musicfm.*` resolves the clone
        from musicfm.model.musicfm_25hz import MusicFM25Hz  # noqa: E402

        data = musicfm_data_dir()
        model = MusicFM25Hz(stat_path=str(data / "fma_stats.json"),
                            model_path=str(data / "pretrained_fma.pt"))
        device = self._requested
        try:
            model.to(device)
        except Exception:  # mps unsupported op / no mps -> CPU
            device = "cpu"
            model.to(device)
        model.eval()
        self._model, self._device = model, device

    def _load_audio(self, wav_path: Path):
        import soundfile as sf
        import torch

        audio, sr = sf.read(str(wav_path), dtype="float32", always_2d=True)
        mono = audio.mean(axis=1)
        if sr != SR:
            import torchaudio
            mono = torchaudio.functional.resample(torch.from_numpy(mono), sr, SR).numpy()
        return torch.from_numpy(mono).unsqueeze(0)  # (1, samples)

    def features(self, wav_path: Path) -> dict[str, np.ndarray]:
        import torch

        self._load()
        x = self._load_audio(Path(wav_path)).to(self._device)
        out: dict[str, np.ndarray] = {}
        with torch.no_grad():
            for name, ix in LAYER_IX.items():
                emb = self._model.get_latent(x, layer_ix=ix)        # (1, frames, 1024)
                out[name] = emb.squeeze(0).detach().to("cpu").numpy().astype(np.float32)
        return out
