"""The frozen-encoder ``Extractor`` protocol and a deterministic fake for CPU tests.

Every encoder in the bake-off is wrapped as an ``Extractor`` that declares its layers, native frame rate and
feature dims, and returns early/middle/final layer features for an audio file. Real extractors load heavy
weights and run on the GPU; they are implemented in the GPU window. All CPU-testable downstream code (the
probe body, heads, training loop, decode, decision) depends only on this protocol, so it runs against the
``FakeExtractor`` with no model download.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

#: The three layer positions every encoder declares (a learned 3-way mix combines them).
LAYER_NAMES = ("early", "middle", "final")

#: Licence families allowed for training/shipping (model weights MIT; upstream MIT/Apache/BSD/CC0/CC BY).
_ALLOWED_FAMILIES = ("MIT", "Apache-2.0", "BSD", "CC0", "CC BY")


def _licence_allowed(licence: str) -> bool:
    """A licence is allowed only if it names an allowed family and carries no NC / ND restriction."""
    text = (licence or "").strip()
    upper = text.upper()
    if "-NC" in upper or " NC" in upper or "-ND" in upper:   # non-commercial / no-derivatives: rejected
        return False
    return any(text.startswith(a) for a in _ALLOWED_FAMILIES)


@dataclass(frozen=True)
class ExtractorDescriptor:
    """What an encoder is and how its features are shaped."""
    name: str
    code_licence: str
    weight_licence: str
    native_rate_hz: float          # the encoder's own frame rate (resampled to 50 Hz downstream)
    feature_dim: int               # feature width per layer
    layer_names: tuple[str, ...] = LAYER_NAMES

    @property
    def free_and_allowed(self) -> bool:
        return _licence_allowed(self.code_licence) and _licence_allowed(self.weight_licence)


@runtime_checkable
class Extractor(Protocol):
    def descriptor(self) -> ExtractorDescriptor: ...

    def features(self, wav_path: Path) -> dict[str, np.ndarray]:
        """Features per declared layer: {layer_name: array of shape (frames, feature_dim)} at native_rate_hz."""
        ...


class FakeExtractor:
    """A deterministic tiny extractor for tests: features are a seeded function of the file path and layer.

    No audio is read; the number of frames is derived from an optional ``frames`` argument so tests can make
    short sequences. Produces the declared layers/shapes so the whole downstream stack is CPU-testable.
    """

    def __init__(self, name: str = "fake", native_rate_hz: float = 25.0, feature_dim: int = 8,
                 frames: int = 20):
        self._desc = ExtractorDescriptor(
            name=name, code_licence="MIT", weight_licence="MIT",
            native_rate_hz=native_rate_hz, feature_dim=feature_dim,
        )
        self._frames = frames

    def descriptor(self) -> ExtractorDescriptor:
        return self._desc

    def features(self, wav_path: Path) -> dict[str, np.ndarray]:
        out = {}
        for i, layer in enumerate(self._desc.layer_names):
            seed = abs(hash((str(wav_path), layer))) % (2**32)
            rng = np.random.default_rng(seed)
            arr = rng.standard_normal((self._frames, self._desc.feature_dim)).astype(np.float32)
            # a tiny per-layer offset so the three layers are distinguishable
            out[layer] = arr + float(i)
        return out
