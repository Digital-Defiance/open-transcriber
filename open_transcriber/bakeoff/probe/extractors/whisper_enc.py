"""Whisper base.en encoder extractor (MIT) — CPU only, auxiliary ear.

Per the 21 October smoke test: ``whisper.load_model("base.en")`` (MIT, cached under ``.cache/whisper/``),
**CPU only** (mps is blocked by an unimplemented SparseMPS op in the positional-embedding path; CPU is
~0.12 s/clip, a non-issue). Encoder blocks 1/3/6 (1-indexed in the brief; captured as the 0-indexed outputs
1/3/5 and the post-LN encoder output) give early/middle/final, each ``(1500, 512)`` at 50 Hz over the fixed
30 s window. Whisper is an auxiliary ear, added only under the prereg Whisper rule.

Heavy imports (torch, whisper) are lazy so this module stays import-cheap.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ... import paths
from ..extractor import ExtractorDescriptor

#: Encoder block outputs captured as early/middle; final is the post-LN encoder output. base.en has 6 blocks.
EARLY, MIDDLE = 1, 3     # 0-indexed block outputs, matching the smoke driver
FEATURE_DIM = 512
NATIVE_RATE_HZ = 50.0


def whisper_cache() -> Path:
    return paths.copyist_root() / ".cache" / "whisper"


class WhisperEncExtractor:
    """Whisper base.en encoder, blocks 1/3/final -> (1500,512) @ 50 Hz, CPU only. Auxiliary per the prereg rule."""

    def __init__(self, device: str | None = None):
        # Whisper's encoder hits an unimplemented SparseMPS op; the smoke test pins it to CPU.
        self._device = "cpu"
        self._model = None
        self._desc = ExtractorDescriptor(
            name="whisper-base-en", code_licence="MIT", weight_licence="MIT",
            native_rate_hz=NATIVE_RATE_HZ, feature_dim=FEATURE_DIM,
            device="cpu", model_hash="whisper-base.en",
            note="auxiliary ear (prereg Whisper rule); CPU only (mps blocked by SparseMPS)",
        )
        if not self._desc.free_and_allowed:
            raise PermissionError("whisper licence is not free/allowed; refusing to load")

    def descriptor(self) -> ExtractorDescriptor:
        return self._desc

    def _load(self):
        if self._model is not None:
            return
        import whisper

        model = whisper.load_model("base.en", download_root=str(whisper_cache()))
        model.to("cpu")
        model.eval()
        self._model = model

    def features(self, wav_path: Path) -> dict[str, np.ndarray]:
        import torch
        import whisper

        self._load()
        audio = whisper.load_audio(str(Path(wav_path)))      # 16 kHz mono float32
        audio = whisper.pad_or_trim(audio)                   # fixed 30 s window -> 3000 mel frames
        mel = whisper.log_mel_spectrogram(audio).to("cpu")
        enc = self._model.encoder
        caps: dict[int, torch.Tensor] = {}
        handles = []
        for idx, block in enumerate(enc.blocks):
            handles.append(block.register_forward_hook(
                lambda _m, _i, out, idx=idx: caps.__setitem__(idx, out.detach())))
        try:
            with torch.no_grad():
                final = enc(mel.unsqueeze(0))                # (1, 1500, 512); native 50 Hz over 30 s
        finally:
            for h in handles:
                h.remove()
        return {
            "early": caps[EARLY].squeeze(0).to("cpu").numpy().astype(np.float32),
            "middle": caps[MIDDLE].squeeze(0).to("cpu").numpy().astype(np.float32),
            "final": final.squeeze(0).to("cpu").numpy().astype(np.float32),
        }
