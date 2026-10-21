"""Dasheng-base extractor (Apache-2.0) — final-layer only.

The 21 October smoke test showed Dasheng's final output is a clean time-aligned ``(frames, 768)`` at 25 Hz,
but the only intermediate signal available (``.blocks`` forward hooks) is in the flattened ``(f t)`` patch
space and is NOT time-aligned to the final (e.g. 248 vs 752 frames for the same clip). Dasheng has no public
API that returns a time-aligned early/middle hidden state. Per the plan and the prereg rule, this extractor
therefore ships **final-only**: it returns the final representation for all declared layers and records
``"final-only"``. It never emits the misaligned block-hook layers.

Heavy imports (torch, torchaudio, soundfile, dasheng) are lazy so this module stays import-cheap.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ... import paths
from ..extractor import ExtractorDescriptor

SR = 16000          # Dasheng consumes 16 kHz mono
FEATURE_DIM = 768
NATIVE_RATE_HZ = 25.0


def encoders_root() -> Path:
    return paths.copyist_root() / ".cache" / "ear" / "encoders"


class DashengExtractor:
    """Dasheng base, final (frames,768) @ 25 Hz on mps (CPU fallback). Final-only (no aligned early/middle)."""

    def __init__(self, device: str | None = None):
        self._requested = device or "mps"
        self._model = None
        self._device = None
        self._desc = ExtractorDescriptor(
            name="dasheng-base", code_licence="Apache-2.0", weight_licence="Apache-2.0",
            native_rate_hz=NATIVE_RATE_HZ, feature_dim=FEATURE_DIM,
            device=self._requested, model_hash="dasheng_base",
            note="final-only (no time-aligned early/middle via the real API)",
        )
        if not self._desc.free_and_allowed:
            raise PermissionError("dasheng licence is not free/allowed; refusing to load")

    def descriptor(self) -> ExtractorDescriptor:
        return self._desc

    def _load(self):
        if self._model is not None:
            return
        import dasheng

        model = dasheng.dasheng_base()
        device = self._requested
        try:
            model.to(device)
        except Exception:
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
        with torch.no_grad():
            final = self._model(x)                              # (1, frames, 768)
        arr = final.squeeze(0).detach().to("cpu").numpy().astype(np.float32)
        # Final-only: the same time-aligned representation stands in for every declared layer. The layer mix
        # learns to collapse to it; we never return the misaligned block-hook outputs.
        return {name: arr for name in self._desc.layer_names}
